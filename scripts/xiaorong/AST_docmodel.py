import os
import sys
import time
import threading
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np



dataset_dir = "../dataset/final_dataset/"
train_path = os.path.join(dataset_dir, "train.jsonl")
val_path = os.path.join(dataset_dir, "val.jsonl")
test_path = os.path.join(dataset_dir, "test.jsonl")
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(output_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
history_path = os.path.join(output_dir, f"AST_training_log_{timestamp}.json")

# 使用方式：
# train_list, val_list, test_list = JSONLDataManager.load_datasets(train_path, val_path, test_path)
# 自动检测显卡，如果有 NVIDIA 显卡则使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")


# ===== 关键：在 import transformers 之前设置 HF 镜像源 =====
# 国内环境无法直接访问 huggingface.co，必须使用 hf-mirror.com
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# 统一设置 cache 路径
my_cache_path = "/root/autodl-tmp/water/hf_cache/"

# GraphCodeBERT 的 HuggingFace 模型 ID（不是本地路径）
graphcodebert_model_id = "microsoft/graphcodebert-base"
roberta_base_model_id = "roberta-base"


class LoadingSpinner:
    """在慢速操作（如模型/分词器加载）期间显示动态旋转进度指示器"""

    SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # braille 动画
    # SPINNER_CHARS = "|/-\\"  # 备选：经典 ASCII 旋转

    def __init__(self, message="加载中"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = None

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            char = self.SPINNER_CHARS[idx % len(self.SPINNER_CHARS)]
            sys.stdout.write(f"\r  {char} {self.message} ... {elapsed:.1f}s")
            sys.stdout.flush()
            idx += 1
            self._stop_event.wait(0.1)

    def start(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, success_msg=None):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        elapsed = time.time() - self._start_time
        if success_msg:
            sys.stdout.write(f"\r  ✅ {success_msg} ({elapsed:.1f}s)\n")
        else:
            sys.stdout.write(f"\r  ✅ {self.message} 完成 ({elapsed:.1f}s)\n")
        sys.stdout.flush()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.stop()
        else:
            self._stop_event.set()
            if self._thread:
                self._thread.join()
            elapsed = time.time() - self._start_time
            sys.stdout.write(f"\r  ❌ {self.message} 失败 ({elapsed:.1f}s)\n")
            sys.stdout.flush()
        return False

def load_with_progress(load_fn, description):
    """带进度显示的加载包装器

    Args:
        load_fn: 无参可调用对象，返回加载结果
        description: 加载描述文字

    Returns:
        加载结果
    """
    with LoadingSpinner(description) as spinner:
        result = load_fn()
    return result

class JSONLDataManager:
    @staticmethod
    def load_jsonl(file_path):
        """
        读取单个 JSONL 文件并返回数据列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        return data

    @staticmethod
    def load_datasets(train_path, val_path, test_path):
        """
        从固定目录读取训练集、验证集和测试集
        """
        train_data = JSONLDataManager.load_jsonl(train_path)
        val_data = JSONLDataManager.load_jsonl(val_path)
        test_data = JSONLDataManager.load_jsonl(test_path)

        print(f"✅ 数据加载完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条, 测试集 {len(test_data)} 条")
        return train_data, val_data, test_data

class AuditCollator:
    def __init__(self, tk_code, tk_nl, max_len=512):
        self.tk_code = tk_code
        self.tk_nl = tk_nl
        self.max_len = max_len

    def __call__(self, batch):
        # 提取文本数据
        codes = [item['code'] for item in batch]
        comments = [item['comment'] for item in batch]
        
        # 提取特征与标签
        # score_avg 是训练特征（而非预测目标），用于辅助异常检测
        #  "score_style": 0.475, "score_robustness": 0.425, "score_correctness": 0.5, "score_attack_surface": 0.0
        feat_score = torch.tensor([[
                                        item.get('score_style', 0.0),
                                        item.get('score_robustness', 0.0),
                                        item.get('score_correctness', 0.0),
                                        item.get('score_attack_surface', 0.0)
                                    ] for item in batch]).float()
        labels_a = torch.tensor([item.get('is_bug', 0) for item in batch]).long()

        # Tokenization
        tk_args = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.max_len,
            "return_tensors": "pt"
        }
        
        batch_code = self.tk_code(codes, **tk_args)
        batch_nl = self.tk_nl(comments, **tk_args)

        # 提取语言标签，用于按语言分类统计指标
        langs = [item.get('lang', 'unknown') for item in batch]

        return {
            "code": batch_code,
            "nl": batch_nl,
            "feat_score": feat_score,
            "label_a": labels_a,
            "lang": langs
        }


class EarlyStopping:
    """早停法：当验证指标在连续 patience 轮内没有改善时停止训练
    
    Args:
        patience: 容忍验证指标不改善的轮数（默认 3）
        min_delta: 判定为"改善"所需的最小变化量（默认 0.0）
        mode: 'min' 表示指标越小越好（如 loss），'max' 表示越大越好（如 accuracy）
    """
    def __init__(self, patience=3, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _is_improvement(self, current_score):
        if self.best_score is None:
            return True
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        else:  # mode == 'max'
            return current_score > self.best_score + self.min_delta

    def __call__(self, current_score):
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"  ⚠️ 早停计数器: {self.counter}/{self.patience} (最佳: {self.best_score:.4f}, 当前: {current_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                print("  🛑 触发早停！训练提前终止。")


class CodeAuditDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: 包含 {"code": "...", "comment": "...", "score": 0.8, "is_anomaly": 1} 的列表
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CodeAuditSystem(nn.Module):
    def __init__(self, model_dim=768):
        super().__init__()
        self.code_encoder = load_with_progress(
            lambda: AutoModel.from_pretrained(graphcodebert_model_id, cache_dir=my_cache_path),
            f"加载模型: {graphcodebert_model_id}"
        )
        self.nl_encoder = load_with_progress(
            lambda: AutoModel.from_pretrained(roberta_base_model_id, cache_dir=my_cache_path),
            f"加载模型: {roberta_base_model_id}"
        )
        
        # 语义对齐层
        self.proj = nn.ModuleDict({
            'code': nn.Linear(model_dim, model_dim),
            'nl': nn.Linear(model_dim, model_dim)
        })
        
        # 交互与融合 (简化版以保持清爽)
        self.attn = nn.MultiheadAttention(model_dim, 8, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim)
        )
        
        # 评分特征嵌入层：将 score_avg 从标量映射到低维向量，作为输入特征注入
        self.score_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 32)
        )
        
        # 分类头（score_avg 作为特征拼接到 latent 中，仅预测 is_bug）
        self.classifier = nn.Sequential(
            nn.Linear(model_dim + 32, 128),  # 768 + 32 = 800
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.anomaly_head = nn.Linear(128, 2)

    def forward(self, code_inputs, nl_inputs, feat_score=None):
        """
        Args:
            code_inputs: 代码分词结果
            nl_inputs: 注释分词结果
            feat_score: (B,) 或 (B,1) 的评分特征，训练时必传，推理时可选
        """
        # 编码与投影
        h_c = self.proj['code'](self.code_encoder(**code_inputs).last_hidden_state)
        h_nl = self.proj['nl'](self.nl_encoder(**nl_inputs).last_hidden_state)

        # 跨模态交互 (代码向注释对齐)
        ctx, _ = self.attn(h_c, h_nl, h_nl)
        
        # 融合与全局池化
        fused = self.fusion(torch.cat([h_c, ctx], dim=-1))
        feat = (fused[:, 0, :] + fused.mean(dim=1)) / 2  # CLS + Mean, shape: (B, model_dim)
        
        # 注入评分特征
        if feat_score is not None:
            # 1. 健壮性检查：如果 batch size 为 1 且变成了 1D 张量 (4,)，将其升维回 (1, 4)
            if feat_score.dim() == 1:
                feat_score = feat_score.unsqueeze(0)  # (4,) -> (1, 4)
            # 2. 映射维度：(B, 4) -> (B, 32)
            score_vec = self.score_embed(feat_score)  
            # 3. 特征拼接：(B, model_dim) + (B, 32) -> (B, model_dim + 32)
            feat = torch.cat([feat, score_vec], dim=-1)
        else:
            # 推理阶段无评分特征时，用零向量占位
            score_vec = torch.zeros(feat.size(0), 32, device=feat.device)
            feat = torch.cat([feat, score_vec], dim=-1)
        
        latent = self.classifier(feat)
        return self.anomaly_head(latent)

    def compute_loss(self, anomaly_pred, labels_anomaly):
        loss_a = nn.CrossEntropyLoss()(anomaly_pred, labels_anomaly)
        return loss_a
    

class AuditPipeline:
    def __init__(self, model, tokenizers, device="cuda"):
        self.model = model.to(device)
        self.tk_code, self.tk_nl = tokenizers
        self.device = device
        self.scaler = GradScaler()

    def __call__(self, code, comment, feat_score=None):
        """推理：一行代码调用，feat_score 为可选的评分特征"""
        self.model.eval()
        tks = self._prepare_inputs(code, comment)
        with torch.no_grad(), autocast(device_type=self.device.type):
            logits = self.model(**tks, feat_score=feat_score)
        
        status = torch.argmax(logits, dim=-1).item()
        return {"is_bug": bool(status)}

    def train_step(self, batch, optimizer):
        """训练：内部处理所有显卡搬运和梯度更新"""
        self.model.train()
        optimizer.zero_grad()
        
        # 搬运数据
        c_in = {k: v.to(self.device) for k, v in batch['code'].items()}
        n_in = {k: v.to(self.device) for k, v in batch['nl'].items()}
        f_s = batch['feat_score'].to(self.device)
        l_a = batch['label_a'].to(self.device)

        with autocast(device_type=self.device.type):
            a_pred = self.model(c_in, n_in, feat_score=f_s)
            loss = self.model.compute_loss(a_pred, l_a)

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        return loss.item()

    def _prepare_inputs(self, code, comment):
        """内部工具：处理输入"""
        args = {"return_tensors": "pt", "padding": "max_length", "max_length": 512, "truncation": True}
        return {
            "code_inputs": {k: v.to(self.device) for k, v in self.tk_code(code, **args).items()},
            "nl_inputs": {k: v.to(self.device) for k, v in self.tk_nl(comment, **args).items()}
        }
    def evaluate(self, dataloader):
        self.model.eval()
        all_anomaly_preds = []
        all_anomaly_labels = []
        all_langs = []
        total_val_loss = 0.0
        num_batches = 0

        print("开始验证评估...")
        with torch.no_grad(), autocast(device_type=self.device.type):
            for batch in dataloader:
                # 搬运数据到 GPU
                c_in = {k: v.to(self.device) for k, v in batch['code'].items()}
                n_in = {k: v.to(self.device) for k, v in batch['nl'].items()}
                f_s = batch['feat_score'].to(self.device)
                l_a = batch['label_a'].to(self.device)
                
                # 模型预测
                a_pred = self.model(c_in, n_in, feat_score=f_s)

                # 计算验证损失
                loss = self.model.compute_loss(a_pred, l_a)
                total_val_loss += loss.item()
                num_batches += 1

                # 收集结果
                all_anomaly_preds.extend(torch.argmax(a_pred, dim=-1).cpu().numpy())
                all_anomaly_labels.extend(batch['label_a'].numpy())
                all_langs.extend(batch['lang'])

        # 计算总体指标
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        all_labels = np.array(all_anomaly_labels)
        all_preds = np.array(all_anomaly_preds)

        overall_p = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        overall_r = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        overall_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        # 按语言分类统计 P / R / F1
        lang_metrics = {}
        unique_langs = sorted(set(all_langs))
        for lang in unique_langs:
            mask = np.array([l == lang for l in all_langs])
            lang_labels = all_labels[mask]
            lang_preds = all_preds[mask]
            if len(lang_labels) > 0:
                lang_p = precision_score(lang_labels, lang_preds, average='binary', zero_division=0)
                lang_r = recall_score(lang_labels, lang_preds, average='binary', zero_division=0)
                lang_f1 = f1_score(lang_labels, lang_preds, average='binary', zero_division=0)
                lang_metrics[lang] = {
                    "precision": round(lang_p, 4),
                    "recall": round(lang_r, 4),
                    "f1": round(lang_f1, 4),
                    "count": int(mask.sum())
                }

        # 打印按语言分类的指标
        print("\n📋 --- 按语言分类指标 ---")
        for lang, lm in lang_metrics.items():
            print(f"  [{lang:>12s}] P={lm['precision']:.4f}  R={lm['recall']:.4f}  F1={lm['f1']:.4f}  (n={lm['count']})")
        print(f"  {'总体 (Overall)':>12s}] P={overall_p:.4f}  R={overall_r:.4f}  F1={overall_f1:.4f}  (n={len(all_labels)})")

        metrics = {
            "val_loss": avg_val_loss,
            "anomaly_acc": accuracy_score(all_labels, all_preds),
            "anomaly_f1": overall_f1,
            "overall_precision": round(overall_p, 4),
            "overall_recall": round(overall_r, 4),
            "overall_f1": round(overall_f1, 4),
            "lang_metrics": lang_metrics,
        }
        return metrics
    

if __name__ == "__main__":
    train_list, val_list, test_list = JSONLDataManager.load_datasets(train_path, val_path, test_path)
    # 将模型加载到显存中
    # 注意：模型初始化应在函数外部，防止显存重复占用
    # 1. 初始化
    print("\n⏳ 开始加载分词器和模型...")
    tokenizer_code = load_with_progress(
        lambda: AutoTokenizer.from_pretrained(graphcodebert_model_id, cache_dir=my_cache_path),
        f"加载分词器: {graphcodebert_model_id}"
    )
    tokenizer_nl = load_with_progress(
        lambda: AutoTokenizer.from_pretrained(roberta_base_model_id, cache_dir=my_cache_path),
        f"加载分词器: {roberta_base_model_id}"
    )
    print("✅ 分词器加载完成")
    # 2. 构建 DataLoader
    collator = AuditCollator(tokenizer_code, tokenizer_nl)
    train_loader = DataLoader(CodeAuditDataset(train_list), batch_size=16, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(CodeAuditDataset(val_list), batch_size=16, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(CodeAuditDataset(test_list), batch_size=16, shuffle=False, collate_fn=collator)
    print("✅ 数据加载器准备就绪")

    #3. 初始化 Pipeline
    model = CodeAuditSystem()
    pipeline = AuditPipeline(model, (tokenizer_code, tokenizer_nl), device=device)
    optimizer = AdamW(model.parameters(), lr=1e-4) # 建议从更小的学习率开始微调
    # 记录过程中的学习率、loss、指标等信息，便于后续分析和调试

    # 4. 动态学习率调度器：当验证损失停滞时自动降低学习率
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # 监控指标越小越好（验证损失）
        factor=0.5,        # 学习率衰减因子（新lr = 旧lr * factor）
        patience=2,        # 容忍验证损失不下降的轮数
        min_lr=1e-7        # 学习率下限
    )

    # 5. 早停法：验证损失连续 patience 轮不改善则提前终止训练
    early_stopping = EarlyStopping(patience=4, min_delta=1e-4, mode='min')

    print("✅ 训练管道初始化完成，准备开始训练")
    print(f"   学习率调度器: ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-7)")
    print(f"   早停策略: patience=4, min_delta=1e-4")

    # 6. 训练主循环（含动态学习率 + 早停）
    max_epochs = 30  # 设置较大的上限，由早停决定实际训练轮数
    best_val_loss = float('inf')
    training_history = []  # 记录每轮的 loss、lr、ACC、F1

    for epoch in range(max_epochs):
        # --- 训练阶段 ---
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{max_epochs} | 当前学习率: {current_lr:.2e}")
        print(f"{'='*50}")
        
        for batch in train_loader:
            loss = pipeline.train_step(batch, optimizer)
            total_loss += loss
        
        avg_train_loss = total_loss / len(train_loader)
        
        # --- 验证阶段 ---
        sample_val = val_list[0]
        res = pipeline(sample_val["code"], sample_val["comment"])
        # 在此评估验证集性能，计算验证损失和指标
        metrics = pipeline.evaluate(val_loader)
        val_loss = metrics['val_loss']

        # --- 动态学习率调整 ---
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  📉 学习率已调整: {current_lr:.2e} → {new_lr:.2e}")

        # --- 早停检查 ---
        early_stopping(val_loss)
        
        # --- 记录训练历史 ---
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "learning_rate": new_lr,
            "accuracy": round(metrics['anomaly_acc'], 6),
            "overall_precision": metrics['overall_precision'],
            "overall_recall": metrics['overall_recall'],
            "overall_f1": metrics['overall_f1'],
            "lang_metrics": metrics['lang_metrics']
        }
        training_history.append(epoch_record)

        # --- 输出报告 ---
        print(f"\n📊 --- Epoch {epoch+1} 验证集性能报告 ---")
        print(f"  训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f}")
        print(f"  1. 异常检测准确率 (ACC): {metrics['anomaly_acc']*100:.2f}%")
        print(f"  2. 总体 P={metrics['overall_precision']:.4f}  R={metrics['overall_recall']:.4f}  F1={metrics['overall_f1']:.4f}")
        print(f"  当前学习率: {new_lr:.2e}")
        print(f"  推理示例: {res}")
        print(f"{'='*50}")

        # 记录最佳模型信息
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  🎯 新的最佳验证损失: {best_val_loss:.4f}")

        # 早停触发则退出训练
        if early_stopping.early_stop:
            print(f"\n🛑 早停触发于 Epoch {epoch+1}，最佳验证损失: {best_val_loss:.4f}")
            break

    print(f"\n✅ 训练完成！共训练 {epoch+1} 轮，最终验证损失: {val_loss:.4f}，最佳验证损失: {best_val_loss:.4f}")

    # 7. 将训练历史记录输出至 output 文件夹

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    print(f"📝 训练历史已保存至: {history_path}")
