import json
import collections
from sklearn.model_selection import train_test_split


def merge_jsonl(input_files, output_file):
    """
    合并多个 JSONL 文件为一个。
    
    :param input_files: 包含输入文件路径的列表
    :param output_file: 输出文件的路径
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # 简单的去除首尾空格并写入，保持原样合并
                        if line.strip():
                            outfile.write(line.strip() + '\n')
                print(f"Successfully merged: {file_path}")
            except FileNotFoundError:
                print(f"Error: {file_path} not found.")


def split_jsonl_stratified(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42,
                            data_path='../dataset/final_dataset/'):
    """
    按 lang 字段同比例拆分 JSONL 数据集
    """
    # 1. 加载数据并统计元数据
    data = []
    langs = []
    
    print(f"正在读取文件: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append(item)
                # 获取 lang 字段，如果没有则标记为 unknown
                langs.append(item.get("lang", "unknown"))

    # 统计初始分布
    total_count = len(data)
    lang_stats = collections.Counter(langs)
    print("\n--- 原始数据集统计 ---")
    for lang, count in lang_stats.items():
        print(f"语言: {lang:10} | 数量: {count:6} | 占比: {count/total_count:.2%}")

    # 2. 第一次拆分：拆出训练集 (Train) 和 临时集 (Temp)
    # temp_ratio = val_ratio + test_ratio
    train_data, temp_data, train_langs, temp_langs = train_test_split(
        data, langs, 
        test_size=(val_ratio + test_ratio), 
        stratify=langs, 
        random_state=seed
    )

    # 3. 第二次拆分：将临时集对半拆分为 验证集 (Val) 和 测试集 (Test)
    # 计算验证集在临时集中的比例
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=(1 - relative_val_ratio), 
        stratify=temp_langs, 
        random_state=seed
    )

    # 4. 写入文件函数
    def save_jsonl(filename, dataset):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"已保存: {filename} (共 {len(dataset)} 条)")

    print("\n--- 拆分结果 ---")
    save_jsonl(data_path+'train.jsonl', train_data)
    save_jsonl(data_path+'val.jsonl', val_data)
    save_jsonl(data_path+'test.jsonl', test_data)

# --- 调用示例 ---
# 假设你的文件名为 'dataset.jsonl'
# split_jsonl_stratified('dataset.jsonl', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
if __name__ == "__main__":

    # files_to_combine = ['../dataset/Cjs/scored3_output_avg.jsonl', '../dataset/scored/scored3_output_avg.jsonl']
    output = '../dataset/merged_output.jsonl'
    # merge_jsonl(files_to_combine, output)
    split_jsonl_stratified(
        input_file=output,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        data_path='../dataset/final_dataset/'
    )
