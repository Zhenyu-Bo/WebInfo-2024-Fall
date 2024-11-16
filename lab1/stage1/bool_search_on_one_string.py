# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-14

"""
使用dictionary_as_a_string.py压缩的倒排索引表实现布尔查询。
"""

import pandas as pd
import ast
from dictionary_as_a_string import load_term_string, load_term_table, query_posting_list
from bool_search import read_all_ids, tokenize, infix_to_postfix, display_results


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


def main():
    print("正在加载数据，请稍候...")

    # 加载书籍数据
    book_term_string = load_term_string('data/book_term_string.txt')
    book_term_table = load_term_table('data/book_term_table.csv')
    book_all_ids = read_all_ids('data/Book_id.txt')
    book_words_df = pd.read_csv('data/book_words.csv', dtype={'id': int, 'words': str})

    # 加载电影数据
    movie_term_string = load_term_string('data/movie_term_string.txt')
    movie_term_table = load_term_table('data/movie_term_table.csv')
    movie_all_ids = read_all_ids('data/Movie_id.txt')
    movie_words_df = pd.read_csv('data/movie_words.csv', dtype={'id': int, 'words': str})

    print("数据加载完成！")

    while True:
        choice = input("请选择查询类型（1 - 书籍，2 - 电影）：\n")
        if choice == '1':
            term_string = book_term_string
            term_table = book_term_table
            all_ids = book_all_ids
            words_df = book_words_df
            posting_list_file_path = 'data/book_posting_list.txt'
        elif choice == '2':
            term_string = movie_term_string
            term_table = movie_term_table
            all_ids = movie_all_ids
            words_df = movie_words_df
            posting_list_file_path = 'data/movie_posting_list.txt'
        else:
            print("输入错误，请输入 1 或 2。")
            continue  # 重新开始循环

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
            break  # 退出循环，结束程序

if __name__ == "__main__":
    main()
