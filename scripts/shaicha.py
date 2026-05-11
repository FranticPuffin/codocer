import json
import tree_sitter_languages
from tree_sitter import Parser

# 初始化解析器
lang_parsers = {
    "c": Parser(),
    "javascript": Parser()
}
lang_parsers["c"].set_language(tree_sitter_languages.get_language("c"))
lang_parsers["javascript"].set_language(tree_sitter_languages.get_language("javascript"))

def is_complete_function(code, lang):
    """
    通过语法树检查代码是否为完整函数
    """
    if not code or len(code.strip()) < 70:
        return False
    
    # 1. 基础大括号匹配检查
    if code.count('{') != code.count('}') or code.count('{') == 0:
        return False

    # 2. 语法树深度检查
    parser = lang_parsers.get(lang)
    if not parser:
        return True # 无法解析的语言默认跳过

    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    # 检查根节点下是否存在 ERROR
    if root.has_error:
        return False

    # 检查根节点的第一个子节点是否代表一个完整的函数结构
    # C 语言通常是 function_definition
    # JS 可能是 function_declaration, lexical_declaration 或 expression_statement (箭头函数)
    valid_types = {
        "c": ["function_definition"],
        "javascript": ["function_declaration", "method_definition", "variable_declaration", "expression_statement"]
    }
    
    # 遍历第一层子节点，寻找函数特征
    found_func = False
    for child in root.children:
        if child.type in valid_types.get(lang, []):
            found_func = True
            break
            
    return found_func

def clean_dataset(input_file, output_file):
    valid_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_file, 'w', encoding='utf-8') as out:
        
        for line in f:
            total_count += 1
            data = json.loads(line)
            
            # 执行筛查
            if is_complete_function(data["code"], data["lang"]):
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
                valid_count += 1
                
    print(f"处理完成！原始数据: {total_count}, 保留完整函数: {valid_count}, 过滤率: {(1 - valid_count/total_count)*100:.2f}%")

if __name__ == "__main__":
    clean_dataset("cleaned_dataset.jsonl", "clean.jsonl")