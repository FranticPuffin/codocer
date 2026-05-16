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
history_path = os.path.join(output_dir, f"baseline_training_log_{timestamp}.json")

# 自动检测显卡，如果有 NVIDIA 显卡则使用 cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# ===== 关键：设置 HF 离线模式 =====
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 统一设置 cache 路径
my_cache_path = "/root/autodl-tmp/water/hf_cache/"

# Baseline 只使用 GraphCodeBERT
graphcodebert_model_id = "microsoft/graphcodebert-base"


class LoadingSpinner:
    """在慢速操作（如模型/分词器加载）期间显示动态旋转进度指示器"""

    SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # braille 动画

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
    """带进度显示的加载包装器"""
    with LoadingSpinner(description) as spinner:
        result = load_fn()
    return result


class JSONLDataManager:
    @staticmethod
    def load_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        return data

    @staticmethod
    def load_datasets(train_path, val_path, test_path):
        train_data = JSONLDataManager.load_jsonl(train_path)
        val_data = JSONLDataManager.load_jsonl(val_path)
        test_data = JSONLDataManager.load_jsonl(test_path)
        print(f"✅ 数据加载完成: 训练集 {len(train_data)} 条, 验证集 {len(val_data)} 条, 测试集 {len(test_data)} 条")
        return train_data, val_data, test_data


class BaselineCollator:
    """Baseline 数据整理器：只处理 code 输入，不使用 comment/nl"""
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        codes = [item['code'] for item in batch]
        labels = torch.tensor([item.get('is_bug', 0) for item in batch]).long()
        langs = [item.get('lang', 'unknown') for item in batch]

        tk_args = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.max_len,
            "return_tensors": "pt"
        }
        batch_code = self.tokenizer(codes, **tk_args)

        return {
            "code": batch_code,
            "label": labels,
            "lang": langs
        }


class EarlyStopping:
    """早停法"""
    def __init__(self, patience=4, min_delta=0.0, mode='min'):
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
        else:
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
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BaselineModel(nn.Module):
    """Baseline 模型：GraphCodeBERT 编码器 + MLP 分类头
    
    只接受 code 作为输入，输出 is_bug 二分类标签。
    结构：预训练编码器 → CLS pooling → LayerNorm → MLP(768→256→2)
    """
    def __init__(self, model_dim=768):
        super().__init__()
        self.encoder = load_with_progress(
            lambda: AutoModel.from_pretrained(graphcodebert_model_id, cache_dir=my_cache_path),
            f"加载模型: {graphcodebert_model_id}"
        )
        # LayerNorm 稳定 CLS 表示的尺度
        self.layer_norm = nn.LayerNorm(model_dim)
        # MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, code_inputs):
        """
        Args:
            code_inputs: 代码分词结果 (dict of tensors)
        Returns:
            logits: (B, 2) 分类 logits
        """
        # 获取编码器输出
        outputs = self.encoder(**code_inputs)
        # 使用 CLS token 表示
        cls_repr = outputs.last_hidden_state[:, 0, :]  # (B, model_dim)
        # LayerNorm 归一化
        cls_repr = self.layer_norm(cls_repr)
        # MLP 分类
        logits = self.classifier(cls_repr)  # (B, 2)
        return logits

    def compute_loss(self, logits, labels):
        return nn.CrossEntropyLoss()(logits, labels)


class BaselinePipeline:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.scaler = GradScaler()

    def __call__(self, code):
        """推理：输入代码，输出 is_bug 预测"""
        self.model.eval()
        tks = self._prepare_inputs(code)
        with torch.no_grad(), autocast(device_type=self.device.type):
            logits = self.model(**tks)
        status = torch.argmax(logits, dim=-1).item()
        return {"is_bug": bool(status)}

    def train_step(self, batch, optimizer):
        """训练步"""
        self.model.train()
        optimizer.zero_grad()

        c_in = {k: v.to(self.device) for k, v in batch['code'].items()}
        l_a = batch['label'].to(self.device)

        with autocast(device_type=self.device.type):
            logits = self.model(c_in)
            loss = self.model.compute_loss(logits, l_a)

        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        return loss.item()

    def _prepare_inputs(self, code):
        """内部工具：处理输入"""
        args = {"return_tensors": "pt", "padding": "max_length", "max_length": 512, "truncation": True}
        return {
            "code_inputs": {k: v.to(self.device) for k, v in self.tokenizer(code, **args).items()}
        }

    def evaluate(self, dataloader):
        """评估：计算总体和按语言分类的 P/R/F1"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_langs = []
        total_val_loss = 0.0
        num_batches = 0

        print("开始验证评估...")
        with torch.no_grad(), autocast(device_type=self.device.type):
            for batch in dataloader:
                c_in = {k: v.to(self.device) for k, v in batch['code'].items()}
                l_a = batch['label'].to(self.device)

                logits = self.model(c_in)
                loss = self.model.compute_loss(logits, l_a)
                total_val_loss += loss.item()
                num_batches += 1

                all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                all_langs.extend(batch['lang'])

        # 计算总体指标
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        all_labels_arr = np.array(all_labels)
        all_preds_arr = np.array(all_preds)

        overall_p = precision_score(all_labels_arr, all_preds_arr, average='binary', zero_division=0)
        overall_r = recall_score(all_labels_arr, all_preds_arr, average='binary', zero_division=0)
        overall_f1 = f1_score(all_labels_arr, all_preds_arr, average='binary', zero_division=0)

        # 按语言分类统计 P / R / F1
        lang_metrics = {}
        unique_langs = sorted(set(all_langs))
        for lang in unique_langs:
            mask = np.array([l == lang for l in all_langs])
            lang_labels = all_labels_arr[mask]
            lang_preds = all_preds_arr[mask]
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
        print(f"  [{'总体 (Overall)':>12s}] P={overall_p:.4f}  R={overall_r:.4f}  F1={overall_f1:.4f}  (n={len(all_labels_arr)})")

        metrics = {
            "val_loss": avg_val_loss,
            "anomaly_acc": accuracy_score(all_labels_arr, all_preds_arr),
            "overall_precision": round(overall_p, 4),
            "overall_recall": round(overall_r, 4),
            "overall_f1": round(overall_f1, 4),
            "lang_metrics": lang_metrics,
        }
        return metrics


if __name__ == "__main__":
    train_list, val_list, test_list = JSONLDataManager.load_datasets(train_path, val_path, test_path)

    # 1. 加载分词器
    print("\n⏳ 开始加载分词器和模型...")
    tokenizer = load_with_progress(
        lambda: AutoTokenizer.from_pretrained(graphcodebert_model_id, cache_dir=my_cache_path),
        f"加载分词器: {graphcodebert_model_id}"
    )
    print("✅ 分词器加载完成")

    # 2. 构建 DataLoader
    collator = BaselineCollator(tokenizer)
    train_loader = DataLoader(CodeAuditDataset(train_list), batch_size=16, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(CodeAuditDataset(val_list), batch_size=16, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(CodeAuditDataset(test_list), batch_size=16, shuffle=False, collate_fn=collator)
    print("✅ 数据加载器准备就绪")

    # 3. 初始化 Baseline Pipeline
    model = BaselineModel()
    pipeline = BaselinePipeline(model, tokenizer, device=device)
    # 差分学习率：编码器用小学习率（微调），分类头用大学习率（快速收敛）
    optimizer = AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},      # 编码器：缓慢微调
        {"params": model.layer_norm.parameters(), "lr": 1e-4},    # LayerNorm：中等学习率
        {"params": model.classifier.parameters(), "lr": 1e-3},    # 分类头：快速学习
    ])

    # 4. 动态学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    # 5. 早停法
    early_stopping = EarlyStopping(patience=4, min_delta=1e-4, mode='min')

    print("✅ 训练管道初始化完成，准备开始训练")
    print(f"   模型结构: GraphCodeBERT → LayerNorm → MLP(768→256→2)")
    print(f"   输入: code only | 输出: is_bug")
    print(f"   差分学习率: encoder=1e-5, layernorm=1e-4, classifier=1e-3")
    print(f"   学习率调度器: ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-7)")
    print(f"   早停策略: patience=4, min_delta=1e-4")

    # 6. 训练主循环
    max_epochs = 30
    best_val_loss = float('inf')
    training_history = []

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
        res = pipeline(sample_val["code"])
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

    # 7. 保存训练历史
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    print(f" 训练历史已保存至: {history_path}")