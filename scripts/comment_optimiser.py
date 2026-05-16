import json
import torch
import re
from transformers import BitsAndBytesConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
# 将 'D:/ModelCache' 替换为你希望存储模型的真实路径
os.environ['MODELSCOPE_CACHE'] = 'D:/ModelCache'

# --- 配置区 ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
INPUT_FILE = "./last_data.jsonl"  # 输入文件，包含代码、注释等字段
OUTPUT_FILE = "./last_optimised_output.jsonl"

# --- 注释优化提示词 ---
OPTIMISE_PROMPT = """You are an expert code comment optimizer. Your task is to optimize the following code comment.

Requirements:
1. Remove meaningless symbols such as /*, */, //, *, ---, ===, and other decorative/punctuation-only markers that carry no semantic information.
2. Preserve the original semantic meaning exactly — do NOT change, add, or remove any technical meaning.
3. Appropriately expand the comment to make it clearer and more descriptive, while staying faithful to the original intent. For example, if the original comment is brief or cryptic, elaborate on what it likely means in context.
4. Output ONLY the optimized comment text, with no extra explanation, no quotation marks, no prefixes like "Optimized comment:".

Language: {lang}
Original comment:
{comment}

Optimized comment:"""

# --- 检测GPU并自动配置 ---
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"检测到GPU: {gpu_name}, 显存: {gpu_mem:.1f}GB")
else:
    print("未检测到GPU，将使用CPU模式运行（速度较慢）")

# --- 模型加载 ---
if HAS_CUDA:
    # 针对 4070Ti S (16GB) 的精细化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # 开启双量化，极致节省显存
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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


def optimise_comment(lang, comment):
    """
    调用模型对注释进行优化，返回优化后的注释文本
    """
    prompt = OPTIMISE_PROMPT.format(lang=lang, comment=comment)

    messages = [
        {"role": "system", "content": "You are a precise code comment optimizer. Output only the optimized comment text, nothing else."},
        {"role": "user", "content": prompt}
    ]

    # 格式化输入
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    # 生成推理
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,  # 注释优化需要更多token
        temperature=0.3,     # 低温度增加确定性，但允许适度创造性扩充
        do_sample=True,
        top_p=0.9
    )

    # 解析结果
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output_text = response.split("assistant")[-1].strip()

    # 清理输出：去除可能的多余引号和前后缀
    output_text = output_text.strip()
    # 如果模型用引号包裹了结果，去掉引号
    if (output_text.startswith('"') and output_text.endswith('"')) or \
       (output_text.startswith("'") and output_text.endswith("'")):
        output_text = output_text[1:-1].strip()
    # 如果模型添加了 "Optimized comment:" 之类的前缀，去掉
    prefix_patterns = [
        r'^[Oo]ptimized\s*comment\s*[:：]\s*',
        r'^[Oo]ptimised\s*comment\s*[:：]\s*',
        r'^[Rr]esult\s*[:：]\s*',
        r'^[Aa]nswer\s*[:：]\s*',
    ]
    for pat in prefix_patterns:
        output_text = re.sub(pat, '', output_text).strip()

    # 如果模型返回为空，则回退到原始注释（仅去除符号）
    if not output_text:
        output_text = re.sub(r'[/*\n\r]', ' ', comment).strip()

    return output_text


def process_all():
    """
    读取输入文件，对每条数据的注释进行优化，保存结果
    """
    print(f"\n{'='*60}")
    print(f"开始注释优化处理: {INPUT_FILE} ...")
    print(f"{'='*60}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        count = 0
        for line in f_in:
            if not line.strip():
                continue

            data = json.loads(line)

            # 获取原始注释
            original_comment = data.get("comment", "")
            lang = data.get("lang", "")

            # 如果注释为空，跳过优化，直接写入原始数据
            if not original_comment.strip():
                output_obj = {
                    "lang": data.get("lang"),
                    "project": data.get("project"),
                    "bug_ids": data.get("bug_ids"),
                    "class": data.get("class"),
                    "method": data.get("method"),
                    "original_comment": original_comment,
                    "optimised_comment": original_comment,
                    "code": data.get("code"),
                    "is_bug": data.get("is_bug"),
                }
                f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
                count += 1
                if count % 10 == 0:
                    print(f"已处理 {count} 条数据...")
                continue

            # 调用模型优化注释
            optimised = optimise_comment(lang, original_comment)

            # 构造输出格式
            output_obj = {
                "lang": data.get("lang"),
                "project": data.get("project"),
                "bug_ids": data.get("bug_ids"),
                "class": data.get("class"),
                "method": data.get("method"),
                "original_comment": original_comment,
                "optimised_comment": optimised,
                "code": data.get("code"),
                "is_bug": data.get("is_bug"),
            }

            f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            count += 1
            if count % 10 == 0:
                print(f"已处理 {count} 条数据...")

    print(f"\n{'='*60}")
    print(f"注释优化处理完成！结果已保存至: {OUTPUT_FILE}")
    print(f"共处理 {count} 条数据")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all()