import os
import json
import re
import git
from tree_sitter_languages import get_language, get_parser

# 配置
INPUT_JSON = "github_top_repos.json"
OUTPUT_JSONL = "extracted_functions_clean.jsonl"
TMP_DIR = "./tmp_repos"
MAX_FUNCTIONS_PER_REPO = 400

# 正则表达式：匹配 C/JS 风格的注释
# 捕获组 1: 多行注释; 捕获组 2: 单行注释
COMMENT_RE = re.compile(r'(/\*[\s\S]*?\*/)|(//.*)')

def clean_code_and_get_comments(func_body):
    """
    提取代码中的所有注释，并返回清洗后的代码
    """
    internal_comments = []
    
    def replace_func(match):
        comment = match.group(0)
        if comment:
            internal_comments.append(comment.strip())
        return "" # 将注释替换为空字符串

    # 移除注释并提取
    cleaned_code = COMMENT_RE.sub(replace_func, func_body)
    
    # 移除多余的空行和首尾空格
    cleaned_code = "\n".join([line for line in cleaned_code.splitlines() if line.strip()])
    
    return cleaned_code.strip(), internal_comments

import tree_sitter_languages
from tree_sitter import Parser

# --- 修复后的初始化逻辑 ---
# 获取语言对象
language_c = tree_sitter_languages.get_language("c")
language_js = tree_sitter_languages.get_language("javascript")

# 初始化解析器并手动分配语言
parser_c = Parser()
parser_c.set_language(language_c)

parser_js = Parser()
parser_js.set_language(language_js)

parsers = {
    "C": parser_c,
    "JavaScript": parser_js
}

# 对应的 Query 逻辑也需要微调
QUERIES = {
    "C": language_c.query("(function_definition) @func"),
    "JavaScript": language_js.query("(function_declaration) @func (method_definition) @func (arrow_function) @func")
}

import os
import json
import re

# 预编译正则，用于精准匹配和替换
COMMENT_RE = re.compile(r'(/\*[\s\S]*?\*/)|(//.*)')

def process_repo_functions(file_path, lang, repo_name):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    parser = parsers.get(lang)
    query = QUERIES.get(lang)
    if not parser or not query: return []

    tree = parser.parse(bytes(content, "utf8"))
    captures = query.captures(tree.root_node)

    extracted_data = []
    for node, _ in captures:
        if node.has_error: continue

        # --- 步骤 1：寻找函数的总边界（包含函数体和上方紧邻的注释） ---
        start_byte = node.start_byte
        end_byte = node.end_byte
        
        # 向上寻找所有关联的注释节点，扩展起始边界
        curr = node.prev_sibling
        while curr and curr.type in ['comment', 'block_comment']:
            start_byte = curr.start_byte
            curr = curr.prev_sibling
            
        # 提取整个“函数区块”的原始文本
        full_block_text = content[start_byte:end_byte]

        # --- 步骤 2：分离注释与代码 ---
        all_comments = []
        
        def extract_and_remove(match):
            c = match.group(0)
            if c:
                all_comments.append(c.strip())
            return "" # 替换为空，实现代码纯净化

        # 使用正则进行分流
        pure_code = COMMENT_RE.sub(extract_and_remove, full_block_text)
        
        # 清洗代码格式：去除空行并首尾去空格
        pure_code = "\n".join([line for line in pure_code.splitlines() if line.strip()])
        combined_comment = "\n".join(all_comments).strip()

        # --- 步骤 3：过滤并封装 ---
        # 只有当 comment 不为空时才保留
        if not combined_comment:
            continue
            
        # 针对 JS 的特殊修复：如果代码剥离注释后剩下残缺字符（如箭头函数的残留），进行检查
        if lang == "javascript" and len(pure_code) < 5:
            continue

        extracted_data.append({
            "lang": lang.lower(),
            "project": repo_name,
            "class": os.path.basename(file_path),
            "comment": combined_comment,
            "code": pure_code
        })
        
    return extracted_data
def main():
    if not os.path.exists(TMP_DIR): os.makedirs(TMP_DIR)
    
    # 假设你已经有了前一步生成的 github_top_repos.json[cite: 1]
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
        for lang, repos in data.items():
            ext = ".c" if lang == "C" else ".js"
            for repo in repos:
                repo_name = repo['name']
                local_path = os.path.join(TMP_DIR, repo_name.replace("/", "_"))
                
                if not os.path.exists(local_path):
                    print(f"Cloning {repo_name}...")
                    try:
                        git.Repo.clone_from(repo['url'], local_path, depth=1)
                    except: continue

                count = 0
                for root, _, files in os.walk(local_path):
                    if count >= MAX_FUNCTIONS_PER_REPO: break
                    for file in files:
                        if file.endswith(ext):
                            funcs = process_repo_functions(os.path.join(root, file), lang, repo_name)
                            for item in funcs:
                                if count >= MAX_FUNCTIONS_PER_REPO: break
                                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                                count += 1
                print(f"Finished {repo_name}: {count} functions.")

if __name__ == "__main__":
    main()