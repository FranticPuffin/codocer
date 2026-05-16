import json
import random
import re

def mutate_code(code):
    """
    对代码进行简单的变异操作，以生成对比样本。
    这里采用几种常见的变异策略：
    1. 逻辑运算符取反 (== -> !=)
    2. 数值修改 (0 -> 1)
    3. 增删字符
    """
    mutated = code
    # 策略 1: 逻辑运算符替换
    ops = {' == ': ' != ', ' != ': ' == ', ' > ': ' <= ', ' < ': ' >= ', ' && ': ' || '}
    for op, replacement in ops.items():
        if op in mutated:
            mutated = mutated.replace(op, replacement, 1)
            return mutated
            
    # 策略 2: 常数替换
    if '0' in mutated:
        mutated = mutated.replace('0', '1', 1)
    elif '1' in mutated:
        mutated = mutated.replace('1', '0', 1)
    else:
        # 策略 3: 随机在末尾添加空语句或注释
        mutated = mutated.rstrip('}') + '  /* modified */ \n}'
        
    return mutated

def augment_dataset(input_file, output_file):
    augmented_count = 0
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out:
        
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # 生成随机 bug_id
            bug_id = random.randint(1000, 9999)
            
            # 1. 创建原始样本 (is_bug: true)
            # 根据你的要求，这里将原始代码设为 true
            pos_sample = {
                "lang": data.get("lang", ""),
                "project": data.get("project", ""),
                "bug_ids": bug_id,
                "class": data.get("class", ""),
                "method": "",
                "comment": data.get("comment", ""),
                "code": data.get("code", ""),
                "is_bug": True
            }
            out.write(json.dumps(pos_sample, ensure_ascii=False) + "\n")
            
            # 2. 创建变异样本 (is_bug: false)
            neg_code = mutate_code(data["code"])
            neg_sample = pos_sample.copy()
            neg_sample["code"] = neg_code
            neg_sample["is_bug"] = False
            out.write(json.dumps(neg_sample, ensure_ascii=False) + "\n")
            
            augmented_count += 2

    print(f"处理完成！已生成 {augmented_count} 条增强样本 (1:1 比例)。")

if __name__ == "__main__":
    # 请确保 input.jsonl 是你之前清洗过的完整函数数据集
    augment_dataset("clean.jsonl", "final_augmented_train.jsonl")