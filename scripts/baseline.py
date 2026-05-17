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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，服务器上无需 GUI
import matplotlib.pyplot as plt


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

    @staticmethod
    def _compute_detailed_metrics(labels, preds, probs=None):
        """计算单个分组（总体或某语言）的完整指标
        
        Args:
            labels: 真实标签数组 (0=负样本, 1=正样本)
            preds: 预测标签数组
            probs: 正类概率数组（用于 AUC），可选
            
        Returns:
            dict: 包含 Acc, 正/负样本 P/R/F, AUC 的字典
        """
        acc = accuracy_score(labels, preds)
        
        # 正样本 (label=1) 的 P/R/F
        pos_p = precision_score(labels, preds, pos_label=1, average='binary', zero_division=0)
        pos_r = recall_score(labels, preds, pos_label=1, average='binary', zero_division=0)
        pos_f1 = f1_score(labels, preds, pos_label=1, average='binary', zero_division=0)
        
        # 负样本 (label=0) 的 P/R/F
        neg_p = precision_score(labels, preds, pos_label=0, average='binary', zero_division=0)
        neg_r = recall_score(labels, preds, pos_label=0, average='binary', zero_division=0)
        neg_f1 = f1_score(labels, preds, pos_label=0, average='binary', zero_division=0)
        
        # AUC（需要概率值且两类都存在）
        auc_val = None
        if probs is not None and len(np.unique(labels)) > 1:
            try:
                auc_val = roc_auc_score(labels, probs)
            except ValueError:
                auc_val = None
        
        return {
            "acc": round(acc, 4),
            "pos_precision": round(pos_p, 4),
            "pos_recall": round(pos_r, 4),
            "pos_f1": round(pos_f1, 4),
            "neg_precision": round(neg_p, 4),
            "neg_recall": round(neg_r, 4),
            "neg_f1": round(neg_f1, 4),
            "auc": round(auc_val, 4) if auc_val is not None else None,
            "count": len(labels)
        }

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []  # 正类概率，用于 AUC
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

                # 收集预测结果和概率
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # 正类概率
                all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                all_labels.extend(batch['label'].numpy())
                all_probs.extend(probs)
                all_langs.extend(batch['lang'])

        # 计算总体指标
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        all_labels_arr = np.array(all_labels)
        all_preds_arr = np.array(all_preds)
        all_probs_arr = np.array(all_probs)

        # 总体详细指标
        overall_metrics = self._compute_detailed_metrics(all_labels_arr, all_preds_arr, all_probs_arr)

        # 按语言分类统计详细指标
        lang_metrics = {}
        unique_langs = sorted(set(all_langs))
        for lang in unique_langs:
            mask = np.array([l == lang for l in all_langs])
            lang_labels = all_labels_arr[mask]
            lang_preds = all_preds_arr[mask]
            lang_probs = all_probs_arr[mask]
            if len(lang_labels) > 0:
                lang_metrics[lang] = self._compute_detailed_metrics(lang_labels, lang_preds, lang_probs)

        # 打印按语言分类的详细指标表
        self._print_metrics_table(overall_metrics, lang_metrics)

        metrics = {
            "val_loss": avg_val_loss,
            "anomaly_acc": overall_metrics["acc"],
            "anomaly_f1": overall_metrics["pos_f1"],
            "overall_precision": overall_metrics["pos_precision"],
            "overall_recall": overall_metrics["pos_recall"],
            "overall_f1": overall_metrics["pos_f1"],
            "overall_auc": overall_metrics["auc"],
            "pos_precision": overall_metrics["pos_precision"],
            "pos_recall": overall_metrics["pos_recall"],
            "pos_f1": overall_metrics["pos_f1"],
            "neg_precision": overall_metrics["neg_precision"],
            "neg_recall": overall_metrics["neg_recall"],
            "neg_f1": overall_metrics["neg_f1"],
            "lang_metrics": lang_metrics,
            # 原始数据，用于 AUC 绘图和最终汇总（不写入 JSON 历史）
            "_raw_labels": all_labels_arr,
            "_raw_probs": all_probs_arr,
            "_raw_langs": np.array(all_langs),
        }
        return metrics

    @staticmethod
    def _print_metrics_table(overall_metrics, lang_metrics):
        """以表格形式打印总体和按语言分类的详细指标"""
        # 表头
        header = f"{'类别':<16s} {'Acc':>7s} {'Pos-P':>7s} {'Pos-R':>7s} {'Pos-F':>7s} {'Neg-P':>7s} {'Neg-R':>7s} {'Neg-F':>7s} {'AUC':>7s} {'N':>6s}"
        sep = "-" * len(header)
        
        print(f"\n📋 --- 详细评估指标表 ---")
        print(sep)
        print(header)
        print(sep)
        
        def fmt(val):
            """格式化数值，None 显示为 N/A"""
            return f"{val:.4f}" if val is not None else "  N/A"
        
        # 各语言行
        for lang, lm in lang_metrics.items():
            row = f"{lang:<16s} {fmt(lm['acc']):>7s} {fmt(lm['pos_precision']):>7s} {fmt(lm['pos_recall']):>7s} {fmt(lm['pos_f1']):>7s} {fmt(lm['neg_precision']):>7s} {fmt(lm['neg_recall']):>7s} {fmt(lm['neg_f1']):>7s} {fmt(lm['auc']):>7s} {lm['count']:>6d}"
            print(row)
        
        print(sep)
        
        # 总体行
        om = overall_metrics
        row = f"{'总体 (Overall)':<16s} {fmt(om['acc']):>7s} {fmt(om['pos_precision']):>7s} {fmt(om['pos_recall']):>7s} {fmt(om['pos_f1']):>7s} {fmt(om['neg_precision']):>7s} {fmt(om['neg_recall']):>7s} {fmt(om['neg_f1']):>7s} {fmt(om['auc']):>7s} {om['count']:>6d}"
        print(row)
        print(sep)

    @staticmethod
    def _print_summary_table(val_metrics_dict, test_metrics_dict, best_epoch):
        """打印验证集最佳 + 测试集的最终汇总表格
        
        Args:
            val_metrics_dict: 验证集最佳 epoch 的指标字典 (来自 training_history)
            test_metrics_dict: 测试集 evaluate 返回的 metrics 字典
            best_epoch: 最佳 epoch 编号
        """
        def fmt(val):
            return f"{val:.4f}" if val is not None else "  N/A"

        header = f"{'数据集':<18s} {'Acc':>7s} {'Pos-P':>7s} {'Pos-R':>7s} {'Pos-F':>7s} {'Neg-P':>7s} {'Neg-R':>7s} {'Neg-F':>7s} {'AUC':>7s}"
        sep = "-" * len(header)

        print(f"\n{'='*len(header)}")
        print(f"📊 最终汇总表格（验证集最佳 Epoch + 测试集）")
        print(f"{'='*len(header)}")
        print(sep)
        print(header)
        print(sep)

        # 验证集最佳行
        v = val_metrics_dict
        val_row = f"{'验证集(最佳Ep'+str(best_epoch)+')':<18s} {fmt(v.get('accuracy')):>7s} {fmt(v.get('pos_precision')):>7s} {fmt(v.get('pos_recall')):>7s} {fmt(v.get('pos_f1')):>7s} {fmt(v.get('neg_precision')):>7s} {fmt(v.get('neg_recall')):>7s} {fmt(v.get('neg_f1')):>7s} {fmt(v.get('auc')):>7s}"
        print(val_row)

        # 测试集行
        t = test_metrics_dict
        test_row = f"{'测试集':<18s} {fmt(t.get('anomaly_acc')):>7s} {fmt(t.get('pos_precision')):>7s} {fmt(t.get('pos_recall')):>7s} {fmt(t.get('pos_f1')):>7s} {fmt(t.get('neg_precision')):>7s} {fmt(t.get('neg_recall')):>7s} {fmt(t.get('neg_f1')):>7s} {fmt(t.get('overall_auc')):>7s}"
        print(test_row)

        print(sep)
        print(f"📌 最佳验证 Epoch: {best_epoch}, 验证损失: {v.get('val_loss', 'N/A')}")

    @staticmethod
    def plot_auc(all_labels, all_probs, lang_labels_dict, save_path):
        """绘制 AUC 曲线并保存
        
        Args:
            all_labels: 总体真实标签
            all_probs: 总体正类概率
            lang_labels_dict: {lang: (labels, probs)} 按语言分组的数据
            save_path: 图片保存路径
        """
        # 设置支持中文的字体（服务器环境可能无中文字体，优先尝试常见中文字体）
        import matplotlib.font_manager as fm
        chinese_fonts = [f.name for f in fm.fontManager.ttflist 
                         if any(kw in f.name.lower() for kw in ['simhei', 'simsun', 'noto sans cjk', 'wqy', 'wenquanyi', 'microsoft yahei', 'droid sans fallback', 'arial unicode'])]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams['font.sans-serif']
        else:
            # 无中文字体时使用英文标签，避免显示为方框
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 判断是否可用中文
        use_chinese = bool(chinese_fonts)
        
        plt.figure(figsize=(10, 8))
        
        # 总体 ROC 曲线
        if len(np.unique(all_labels)) > 1:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            overall_label = f'Overall ROC (AUC = {roc_auc:.4f})' if not use_chinese else f'总体 ROC (AUC = {roc_auc:.4f})'
            plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=overall_label)
        
        # 各语言 ROC 曲线
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(lang_labels_dict), 1)))
        for i, (lang, (labels, probs)) in enumerate(sorted(lang_labels_dict.items())):
            if len(np.unique(labels)) > 1:
                try:
                    fpr_l, tpr_l, _ = roc_curve(labels, probs)
                    roc_auc_l = auc(fpr_l, tpr_l)
                    plt.plot(fpr_l, tpr_l, color=colors[i], lw=1.5, linestyle='--',
                             label=f'{lang} (AUC = {roc_auc_l:.4f})')
                except ValueError:
                    pass
        
        # 对角线（随机猜测）
        random_label = 'Random Guess' if not use_chinese else '随机猜测'
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':', alpha=0.6, label=random_label)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        xlabel = 'False Positive Rate' if not use_chinese else '假正率 (False Positive Rate)'
        ylabel = 'True Positive Rate' if not use_chinese else '真正率 (True Positive Rate)'
        title = 'ROC Curve - AUC Evaluation' if not use_chinese else 'ROC 曲线 - AUC 评估'
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 AUC 曲线已保存至: {save_path}")


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
            "pos_precision": metrics['pos_precision'],
            "pos_recall": metrics['pos_recall'],
            "pos_f1": metrics['pos_f1'],
            "neg_precision": metrics['neg_precision'],
            "neg_recall": metrics['neg_recall'],
            "neg_f1": metrics['neg_f1'],
            "auc": metrics['overall_auc'],
            "overall_precision": metrics['overall_precision'],
            "overall_recall": metrics['overall_recall'],
            "overall_f1": metrics['overall_f1'],
            "lang_metrics": metrics['lang_metrics']
        }
        training_history.append(epoch_record)

        # --- 输出报告 ---
        print(f"\n📊 --- Epoch {epoch+1} 验证集性能报告 ---")
        print(f"  训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f}")
        print(f"  1. 准确率 (ACC): {metrics['anomaly_acc']*100:.2f}%")
        print(f"  2. 正样本 P={metrics['pos_precision']:.4f}  R={metrics['pos_recall']:.4f}  F1={metrics['pos_f1']:.4f}")
        print(f"  3. 负样本 P={metrics['neg_precision']:.4f}  R={metrics['neg_recall']:.4f}  F1={metrics['neg_f1']:.4f}")
        auc_str = f"{metrics['overall_auc']:.4f}" if metrics['overall_auc'] is not None else "N/A"
        print(f"  4. AUC: {auc_str}")
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

    # 7. 将训练历史记录输出至 output 文件夹（排除 _raw 内部字段）
    def _sanitize_record(rec):
        """移除不可 JSON 序列化的内部字段"""
        return {k: v for k, v in rec.items() if not k.startswith('_')}

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump([_sanitize_record(r) for r in training_history], f, ensure_ascii=False, indent=2)
    print(f"📝 训练历史已保存至: {history_path}")

    # =====================================================================
    # 8. 最终评估：在测试集上运行完整评估，输出汇总表格 + AUC 曲线
    # =====================================================================
    print("\n" + "=" * 70)
    print("🔬 最终评估：在测试集上运行完整评估")
    print("=" * 70)

    test_metrics = pipeline.evaluate(test_loader)

    # --- 输出最终汇总表格（验证集最佳 Epoch + 测试集）---
    best_epoch_idx = min(range(len(training_history)), key=lambda i: training_history[i]['val_loss'])
    best_ep = training_history[best_epoch_idx]
    BaselinePipeline._print_summary_table(best_ep, test_metrics, best_ep['epoch'])

    # --- 输出测试集按语言分类的详细指标表 ---
    print("\n" + "=" * 70)
    print("📊 测试集详细指标表（含按语言分类）")
    print("=" * 70)
    test_overall = BaselinePipeline._compute_detailed_metrics(
        test_metrics["_raw_labels"],
        np.array([1 if p > 0.5 else 0 for p in test_metrics["_raw_probs"]]),
        test_metrics["_raw_probs"]
    )
    BaselinePipeline._print_metrics_table(test_overall, test_metrics["lang_metrics"])

    # --- 绘制 AUC 曲线 ---
    # 准备按语言分组的数据
    test_lang_dict = {}
    raw_langs = test_metrics["_raw_langs"]
    raw_labels = test_metrics["_raw_labels"]
    raw_probs = test_metrics["_raw_probs"]
    for lang in sorted(set(raw_langs)):
        mask = raw_langs == lang
        test_lang_dict[lang] = (raw_labels[mask], raw_probs[mask])

    # 测试集 AUC 曲线
    auc_plot_path = os.path.join(output_dir, f"baseline_test_auc_{timestamp}.png")
    BaselinePipeline.plot_auc(raw_labels, raw_probs, test_lang_dict, auc_plot_path)

    # 同时为验证集绘制 AUC 曲线（使用最后一个 epoch 的验证数据）
    val_lang_dict = {}
    val_raw_langs = metrics["_raw_langs"]
    val_raw_labels = metrics["_raw_labels"]
    val_raw_probs = metrics["_raw_probs"]
    for lang in sorted(set(val_raw_langs)):
        mask = val_raw_langs == lang
        val_lang_dict[lang] = (val_raw_labels[mask], val_raw_probs[mask])

    val_auc_plot_path = os.path.join(output_dir, f"baseline_val_auc_{timestamp}.png")
    BaselinePipeline.plot_auc(val_raw_labels, val_raw_probs, val_lang_dict, val_auc_plot_path)

    # --- 保存最终测试结果到 JSON ---
    test_result_path = os.path.join(output_dir, f"baseline_test_result_{timestamp}.json")
    test_result = {
        "test_loss": test_metrics["val_loss"],
        "test_acc": test_metrics["anomaly_acc"],
        "test_pos_precision": test_metrics["pos_precision"],
        "test_pos_recall": test_metrics["pos_recall"],
        "test_pos_f1": test_metrics["pos_f1"],
        "test_neg_precision": test_metrics["neg_precision"],
        "test_neg_recall": test_metrics["neg_recall"],
        "test_neg_f1": test_metrics["neg_f1"],
        "test_auc": test_metrics["overall_auc"],
        "test_lang_metrics": test_metrics["lang_metrics"],
        "val_best_epoch": best_ep['epoch'],
        "val_best_loss": best_ep['val_loss'],
        "val_best_acc": best_ep['accuracy'],
        "val_best_pos_precision": best_ep['pos_precision'],
        "val_best_pos_recall": best_ep['pos_recall'],
        "val_best_pos_f1": best_ep['pos_f1'],
        "val_best_neg_precision": best_ep['neg_precision'],
        "val_best_neg_recall": best_ep['neg_recall'],
        "val_best_neg_f1": best_ep['neg_f1'],
        "val_best_auc": best_ep['auc'],
        "training_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
    }
    with open(test_result_path, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    print(f"📝 测试结果已保存至: {test_result_path}")

    print("\n" + "=" * 70)
    print("🎉 全部评估完成！")
    print(f"   📈 测试集 AUC 曲线: {auc_plot_path}")
    print(f"   📈 验证集 AUC 曲线: {val_auc_plot_path}")
    print(f"   📝 训练历史: {history_path}")
    print(f"   📝 测试结果: {test_result_path}")
    print("=" * 70)
