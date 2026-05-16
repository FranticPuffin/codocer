import json
import torch
import re
from transformers import BitsAndBytesConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
# 将 'D:/ModelCache' 替换为你希望存储模型的真实路径
os.environ['MODELSCOPE_CACHE'] = 'D:/ModelCache'

# --- 配置区 ---
MODEL_ID = "qwen/Qwen2.5-Coder-7B-Instruct"
INPUT_FILE = "./final_merged_data.jsonl"  # 评估输入文件，包含代码、注释和 is_bug 字段
OUTPUT_FILE = "./scoredcjs_output.jsonl"
NUM_RUNS = 4  # 评估维度数

# --- 四次评估对应四个不同维度 ---
DIMENSIONS = [
     {
        "key": "attack_surface",
        "name": "攻击面暴露度与数据流敏感性",
        "prompt": """You are an expert code auditor. Rate the 'Anomaly Score' of the following code on a scale of 0.0 to 1.0, focusing ONLY on Attack Surface Exposure & Data Flow Sensitivity.
Criteria:
1. External input handling — Does the code directly process data from untrusted or network sources (e.g., HTTP requests, socket input, file I/O from external paths, environment variables, command-line arguments)?
2. Data flow reachability — Can externally supplied data flow through this function to security-sensitive operations (e.g., SQL queries, OS commands, file writes, authentication logic, cryptographic operations)?
3. Exposure position — Is this function located at a system boundary (API endpoint, network handler, middleware) or deep within internal utility logic?
4. Sensitivity of processed data — Does the code handle sensitive data such as credentials, tokens, personal information, or financial data?

Score 1.0 means the code is deep inside the system with NO exposure to external inputs or sensitive data — minimal attack surface. Score 0.0 means the code is at a critical system boundary, directly processing untrusted external input that flows into security-sensitive operations — maximum attack surface and risk.

Language: {lang}
Context: {comment}
Code:
{code}

Respond with only the numerical score (e.g., 0.50)."""
    },
    {
        "key": "style",
        "name": "代码风格与规范",
        "prompt": """You are an expert code auditor. Rate the 'Anomaly Score' of the following code on a scale of 0.0 to 1.0, focusing ONLY on Code Style & Conventions.
Criteria:
1. Naming conventions (variable, function, class names follow language standards)
2. Code formatting and indentation consistency
3. Adherence to language-specific idioms and best practices
4. Readability and clarity of code structure

Score 1.0 means excellent style and conventions. Score 0.0 means extremely poor style or complete disregard for conventions.

Language: {lang}
Context: {comment}
Code: 
{code}

Respond with only the numerical score (e.g., 0.50)."""
    },
    {
        "key": "robustness",
        "name": "健壮性",
        "prompt": """You are an expert code auditor. Rate the 'Anomaly Score' of the following code on a scale of 0.0 to 1.0, focusing ONLY on Robustness.
Criteria:
1. Error handling (proper try-catch, exception management)
2. Edge case coverage (null checks, boundary conditions)
3. Defensive programming (input validation, safe defaults)
4. Resource management (proper cleanup, no leaks)

Score 1.0 means highly robust code. Score 0.0 means no error handling and extremely fragile code.

Language: {lang}
Context: {comment}
Code:
{code}

Respond with only the numerical score (e.g., 0.50)."""
    },
    {
        "key": "correctness",
        "name": "逻辑正确性",
        "prompt": """You are an expert code auditor. Rate the 'Anomaly Score' of the following code on a scale of 0.0 to 1.0, focusing ONLY on Logical Correctness.
Criteria:
1. Alignment with the stated intent in the comment/context
2. Absence of logical flaws or incorrect algorithms
3. Correct control flow (loops, conditionals, recursion)
4. Correct data flow (variable usage, data transformations)

Score 1.0 means logically correct and fully aligned with intent. Score 0.0 means critical logical bugs or completely wrong implementation.

Language: {lang}
Context: {comment}
Code:
{code}

Respond with only the numerical score (e.g., 0.50)."""
    }
   
]

# --- 检测GPU并自动配置 ---
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"检测到GPU: {gpu_name}, 显存: {gpu_mem:.1f}GB")
else:
    print("未检测到GPU，将使用CPU模式运行（速度较慢）")

# --- 模型加载 ---
# 16GB显存建议使用4-bit量化，既能保证速度又能处理长代码

if HAS_CUDA:
    # 针对 4070Ti S (16GB) 的精细化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, # 开启双量化，极致节省显存
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # 强制将所有层放在 GPU 上，不让它往 CPU 卸载
        device_map="auto", 
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
else:
    # CPU模式：不使用量化，使用float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

def get_anomaly_score(lang, code, comment, dimension_index=0):
    """
    调用模型对代码进行评分，按指定维度给出评分
    dimension_index: 0=代码风格, 1=健壮性, 2=逻辑正确性, 3=攻击面暴露度与数据流敏感性
    """
    dim = DIMENSIONS[dimension_index]
    prompt = dim["prompt"].format(lang=lang, code=code, comment=comment)

    messages = [
        {"role": "system", "content": "You are a precise analyzer. Respond only with a number."},
        {"role": "user", "content": prompt}
    ]
    
    # 格式化输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    # 生成推理
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16,
        temperature=0.2,  # 低温度增加确定性
        do_sample=False
    )
    
    # 解析结果
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_text = response.split("assistant")[-1].strip()
    
    # 使用正则表达式提取数字
    match = re.search(r"(\d+\.\d+|\d+)", output_text)
    if match:
        try:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5
    return 0.5

def process_single_run(run_index):
    """
    执行单次维度评估，将结果保存到带编号的输出文件中
    run_index: 1=代码风格, 2=健壮性, 3=逻辑正确性, 4=攻击面暴露度与数据流敏感性
    返回输出文件路径
    """
    dimension_index = run_index - 1  # 转为0-based索引
    dim = DIMENSIONS[dimension_index]
    output_file = f"./scored4_output_{run_index}.jsonl"
    print(f"\n{'='*60}")
    print(f"开始第 {run_index} 次评估（维度: {dim['name']}）: {INPUT_FILE} ...")
    print(f"{'='*60}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        count = 0
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # 1. 获取模型原始评分（按维度）
            raw_score = get_anomaly_score(
                data.get("lang", ""), 
                data.get("code", ""), 
                data.get("comment", ""),
                dimension_index=dimension_index
            )
            
            # 2. 根据 is_bug 逻辑调整 (若为True，系数乘以0.5)
            is_bug = data.get("is_bug", False)
            final_score = raw_score * 0.5 if is_bug else raw_score
            
            # 3. 构造符合要求的输出格式
            output_obj = {
                "lang": data.get("lang"),
                "project": data.get("project"),
                "class": data.get("class"),
                "comment": data.get("comment"),
                "code": data.get("code"),
                "score": round(final_score, 4),
                "dimension": dim["key"],
                "dimension_name": dim["name"],
                "is_bug": is_bug
            }
            
            f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            count += 1
            if count % 10 == 0:
                print(f"[第{run_index}次-{dim['name']}] 已处理 {count} 条数据...")

    print(f"第 {run_index} 次评估（维度: {dim['name']}）完成！结果已保存至: {output_file}")
    return output_file

def compute_average():
    """
    读取各维度评估的输出文件，计算每条数据得分的平均值，
    生成最终的平均得分文件
    """
    avg_output_file = "./scored4_output_avg.jsonl"
    
    # 读取各维度评估的所有数据
    all_runs = []
    for i in range(1, NUM_RUNS + 1):
        run_file = f"./scored4_output_{i}.jsonl"
        run_data = []
        with open(run_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                run_data.append(json.loads(line))
        all_runs.append(run_data)
        print(f"已加载 {run_file}（维度: {DIMENSIONS[i-1]['name']}），共 {len(run_data)} 条数据")
    
    # 验证各次评估的数据条数一致
    num_entries = len(all_runs[0])
    for i, run_data in enumerate(all_runs):
        if len(run_data) != num_entries:
            print(f"警告: 第{i+1}次评估的数据条数({len(run_data)})与第1次({num_entries})不一致！")
    
    # 计算平均得分并输出
    with open(avg_output_file, 'w', encoding='utf-8') as f_out:
        for idx in range(num_entries):
            # 以第一次评估的数据为基础结构
            avg_obj = {
                "lang": all_runs[0][idx].get("lang"),
                "project": all_runs[0][idx].get("project"),
                "class": all_runs[0][idx].get("class"),
                "comment": all_runs[0][idx].get("comment"),
                "code": all_runs[0][idx].get("code"),
                "is_bug": all_runs[0][idx].get("is_bug"),
            }
            
            # 收集各维度的得分
            scores = []
            score_details = {}
            for run_idx in range(NUM_RUNS):
                score = all_runs[run_idx][idx].get("score", 0.0)
                scores.append(score)
                dim_key = DIMENSIONS[run_idx]["key"]
                score_details[f"score_{dim_key}"] = score
            
            # 计算平均得分
            avg_score = sum(scores) / len(scores)
            
            # 写入各维度得分和平均得分
            avg_obj.update(score_details)
            avg_obj["score_avg"] = round(avg_score, 4)
            
            f_out.write(json.dumps(avg_obj, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print(f"四维度平均得分计算完成！结果已保存至: {avg_output_file}")
    print(f"共处理 {num_entries} 条数据")
    print(f"维度: {', '.join(d['name'] for d in DIMENSIONS)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 执行第四次维度评估
    for run_idx in range(1, NUM_RUNS+1):
        process_single_run(run_idx)
    
    # 计算四次维度评估的平均得分
    compute_average()