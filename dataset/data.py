import json

def batch_merge_jsonl(file1_path, file2_path, file3_path, output_path, batch_size=10000):
    """
    高效流式处理：逐行对齐读取，分批次或逐行直接写入，极低内存占用。
    """
    print("开始流式合并文件...")
    
    # 同时打开所有相关文件
    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(file2_path, 'r', encoding='utf-8') as f2, \
         open(file3_path, 'r', encoding='utf-8') as f3, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        buffer = []
        count = 0
        
        # 使用 zip 逐行打包读取，保证行对齐
        for line1, line2, line3 in zip(f1, f2, f3):
            line1, line2, line3 = line1.strip(), line2.strip(), line3.strip()
            if not line1 or not line2 or not line3:
                continue
            
            try:
                # 解析 JSON
                data1 = json.loads(line1)
                data2 = json.loads(line2)
                data3 = json.loads(line3)
                
                # 构造合并后的单行字典
                merged_item = {
                    # 共有基础字段
                    "lang": data1.get("lang"),
                    "project": data1.get("project"),
                    "class": data1.get("class"),
                    "code": data1.get("code"),
                    # 文件 3 独有
                    "comment": data3.get("optimised_comment"),
                    # 文件 1 独有
                    "score_style": data1.get("score_style"),
                    "score_robustness": data1.get("score_robustness"),
                    "score_correctness": data1.get("score_correctness"),
                    # 文件 2 独有并重命名
                    "score_attack_surface": data2.get("score"),
                    "is_bug": data1.get("is_bug", False)
                }
                
                # 将序列化后的字符串存入缓存区
                buffer.append(json.dumps(merged_item, ensure_ascii=False) + '\n')
                count += 1
                
                # 达到设定的批次大小后，集中写入磁盘并清空缓存
                if len(buffer) >= batch_size:
                    f_out.writelines(buffer)
                    buffer.clear()
                    print(f"已处理并写入 {count} 条数据...")
                    
            except json.JSONDecodeError as e:
                print(f"在第 {count + 1} 行附近发生 JSON 解析错误，已跳过该行。错误信息: {e}")
                continue

        # 写入最后一批剩余的数据
        if buffer:
            f_out.writelines(buffer)
            buffer.clear()

    print(f"合并完成！总计成功处理并写入 {count} 条数据。输出文件：{output_path}")

# --- 运行入口 ---
if __name__ == "__main__":
    # 请替换为您本地的实际文件路径
    FILE1 = "./final_merged_data.jsonl"
    FILE2 = "./scored4_output_1.jsonl"
    FILE3 = "./comment_optimised_output.jsonl"
    OUTPUT = "./merged_output.jsonl"
    
    # 批次大小可根据磁盘I/O性能调节，10000条刷盘一次是一个较为平衡的选择
    batch_merge_jsonl(FILE1, FILE2, FILE3, OUTPUT, batch_size=300)