import json
import ast
import javalang
import textwrap

def python_ast_to_dict(node):
    if isinstance(node, ast.AST):
        result = {"node_type": node.__class__.__name__}
        for field, value in ast.iter_fields(node):
            result[field] = python_ast_to_dict(value)
        return result
    elif isinstance(node, list):
        return [python_ast_to_dict(item) for item in node]
    elif node is Ellipsis:
        return "..."
    return node

def java_ast_to_dict(node):
    if isinstance(node, (javalang.tree.Node, list, set)):
        if isinstance(node, (list, set)):
            return [java_ast_to_dict(item) for item in node]
        
        result = {"node_type": node.__class__.__name__}
        # 修复属性访问：javalang 节点使用 attrs 属性
        for attr in getattr(node, 'attrs', []):
            value = getattr(node, attr)
            result[attr] = java_ast_to_dict(value)
        return result
    return node

def process_jsonl_final(input_path, output_path):
    success_count = 0
    dropped_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                lang = data.get("lang", "").lower()
                code = data.get("code", "")
                ast_obj = None

                if lang == "python":
                    # 修复缩进并解析
                    clean_code = textwrap.dedent(code)
                    tree = ast.parse(clean_code)
                    ast_obj = python_ast_to_dict(tree)
                
                elif lang == "java":
                    # --- 正确的 javalang 解析流程 ---
                    try:
                        # 尝试全文件解析
                        tree = javalang.parse.parse(code)
                    except:
                        # 尝试成员（方法）解析
                        tokens = javalang.tokenizer.tokenize(code)
                        parser = javalang.parser.Parser(tokens)
                        tree = parser.parse_member_declaration()
                    
                    if tree:
                        ast_obj = java_ast_to_dict(tree)

                # 只有成功提取 AST 且能被 JSON 序列化的才写入
                if ast_obj:
                    data["code"] = ast_obj
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    success_count += 1
                else:
                    dropped_count += 1

            except Exception as e:
                dropped_count += 1
                # 打印错误方便调试，但该行已被“物理删除”
                print(f"Line {line_num} dropped: {type(e).__name__} - {e}")

    print(f"\n处理完毕！")
    print(f"成功保存: {success_count} 行")
    print(f"解析失败已剔除: {dropped_count} 行")
# 执行处理
process_jsonl_final('../dataset/scored/scored3_output_avg.jsonl', '../dataset/AST/AST_output.jsonl')
