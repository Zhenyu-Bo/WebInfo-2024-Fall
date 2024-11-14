# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-14

"""
在经过可变长度编码压缩后的倒排索引表上执行布尔查询。
"""

import pandas as pd
import ast
import re
from var_len_decompress import get_inverted_index_list

# 倒排索引表，movie_inverted_index_table.csv 
# 全id表，Movie_id.txt
# 词表，movie_words.csv 用于打印结果 目前格式同助教提供的selected_book_top_1200.csv

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

def evaluate_postfix(postfix_tokens, compressed_file_path, vocabulary_file_path, all_ids):
    """
    评估后缀表达式，返回符合条件的ID集合。
    """
    stack = []
    operators = {'AND', 'OR', 'NOT'}
    for token in postfix_tokens:
        if token not in operators:
            inverted_index = get_inverted_index_list(token, compressed_file_path, vocabulary_file_path)
            stack.append(set(inverted_index))
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
    print("正在加载数据，请稍候...")

    # 加载书籍数据
    book_all_ids = read_all_ids('data/Book_id.txt')
    book_words_df = pd.read_csv('data/book_words.csv', dtype={'id': int, 'words': str})

    # 加载电影数据
    movie_all_ids = read_all_ids('data/Movie_id.txt')
    movie_words_df = pd.read_csv('data/movie_words.csv', dtype={'id': int, 'words': str})

    print("数据加载完成！")

    while True:
        choice = input("请选择查询类型（1 - 书籍，2 - 电影）：\n")
        if choice == '1':
            compressed_file_path = "data/book_inverted_index_compressed.bin"
            vocabulary_file_path = "data/book_vocabulary.csv"
            all_ids = book_all_ids
            words_df = book_words_df
        elif choice == '2':
            compressed_file_path = "data/movie_inverted_index_compressed.bin"
            vocabulary_file_path = "data/movie_vocabulary.csv"
            all_ids = movie_all_ids
            words_df = movie_words_df
        else:
            print("输入错误，请输入 1 或 2。")
            continue  # 重新开始循环

        expression = input("请输入布尔查询表达式：\n")
        tokens = tokenize(expression)
        postfix_tokens = infix_to_postfix(tokens)
        result_ids = evaluate_postfix(postfix_tokens, compressed_file_path, vocabulary_file_path, all_ids)
        if result_ids:
            print("查询结果：\n")
            display_results(result_ids, words_df)
        else:
            print("没有符合条件的结果。")

        cont = input("是否继续查询？(Y/N): ")
        if cont.strip().lower() != 'y':
            print("感谢您的使用！")
            break  # 退出循环，结束程序

if __name__ == "__main__":
    main()
