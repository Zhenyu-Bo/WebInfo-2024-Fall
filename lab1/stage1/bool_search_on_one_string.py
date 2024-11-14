# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-14

"""
使用dictionary_as_a_string.py压缩的倒排索引表实现布尔查询。
"""

import pandas as pd
import ast
import re
from dictionary_as_a_string import load_term_string, load_term_table, query_posting_list

def read_all_ids(file_path):
    """
    读取所有ID，返回ID的集合。
    """
    all_ids = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            id = int(line.strip())
            all_ids.add(id)
    return all_ids

def tokenize(expression):
    """
    将布尔表达式分词。
    """
    tokens = re.findall(r'AND|OR|NOT|\w+|\(|\)', expression)
    return tokens

def infix_to_postfix(tokens):
    """
    将中缀表达式转换为后缀表达式（逆波兰表达式）。
    """
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    operator_stack = []
    operators = {'AND', 'OR', 'NOT'}
    for token in tokens:
        if token not in operators and token not in {'(', ')'}:
            output.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  # 弹出 '('
        else:  # 操作符
            while operator_stack and operator_stack[-1] != '(' and precedence.get(operator_stack[-1], 0) >= precedence.get(token, 0):
                output.append(operator_stack.pop())
            operator_stack.append(token)
    while operator_stack:
        output.append(operator_stack.pop())
    return output

def evaluate_postfix(postfix_tokens, term_string, term_table, posting_list_file_path, all_ids):
    """
    评估后缀表达式，返回符合条件的ID集合。
    """
    stack = []
    operators = {'AND', 'OR', 'NOT'}
    for token in postfix_tokens:
        if token not in operators:
            doc_ids = query_posting_list(token, term_string, term_table, posting_list_file_path)
            stack.append(set(doc_ids))
        elif token == 'NOT':
            operand = stack.pop()
            stack.append(all_ids - operand)
        else:
            right = stack.pop()
            left = stack.pop()
            if token == 'AND':
                stack.append(left & right)
            elif token == 'OR':
                stack.append(left | right)
    return stack.pop() if stack else set()

def display_results(result_ids, words_df):
    """
    根据查询结果的ID，从 DataFrame 中查找并打印ID和标签。
    """
    df = words_df[words_df['id'].isin(result_ids)]
    if not df.empty:
        for _, row in df.iterrows():
            id = row['id']
            words = ast.literal_eval(row['words'])
            print(f"ID: {id}")
            print(f"标签: {', '.join(words)}\n")
    else:
        print("没有符合条件的结果。")

def main():
    # 让用户选择查询类型
    choice = input("请选择查询类型（1 - 书籍，2 - 电影）：\n")
    if choice == '1':
        # 定义书籍文件路径
        term_string_file_path = 'data/book_term_string.txt'
        term_table_file_path = 'data/book_term_table.csv'
        posting_list_file_path = 'data/book_posting_list.txt'
        all_ids_file_path = 'data/Book_id.txt'
        words_df_file_path = 'data/book_words.csv'
    elif choice == '2':
        # 定义电影文件路径
        term_string_file_path = 'data/movie_term_string.txt'
        term_table_file_path = 'data/movie_term_table.csv'
        posting_list_file_path = 'data/movie_posting_list.txt'
        all_ids_file_path = 'data/Movie_id.txt'
        words_df_file_path = 'data/movie_words.csv'
    else:
        print("输入错误，请输入 1 或 2。")
        return

    print("正在加载数据，请稍候...")

    # 加载词项字符串和词项表
    term_string = load_term_string(term_string_file_path)
    term_table = load_term_table(term_table_file_path)
    if not term_string or not term_table:
        print("词项字符串或词项表加载失败或为空。")
        return

    # 加载所有ID
    all_ids = read_all_ids(all_ids_file_path)

    # 加载词表，用于显示结果
    words_df = pd.read_csv(words_df_file_path, dtype={'id': int, 'words': str})

    print("数据加载完成！")

    while True:
        expression = input("请输入布尔查询表达式：\n")
        tokens = tokenize(expression)
        postfix_tokens = infix_to_postfix(tokens)
        result_ids = evaluate_postfix(postfix_tokens, term_string, term_table, posting_list_file_path, all_ids)
        if result_ids:
            print("查询结果：\n")
            display_results(result_ids, words_df)
        else:
            print("没有符合条件的结果。")

        cont = input("是否继续查询？(Y/N): ")
        if cont.strip().lower() != 'y':
            print("感谢您的使用！")
            break

if __name__ == "__main__":
    main()
