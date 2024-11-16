# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-14

"""
在经过可变长度编码压缩后的倒排索引表上执行布尔查询。
"""

import pandas as pd
import time
from var_len_decompress import get_inverted_index_list
from bool_search import read_all_ids, tokenize, infix_to_postfix, display_results


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
        elif token == 'NOT' or token == 'not':
            operand = stack.pop()
            stack.append(all_ids - operand)
        else:
            right = stack.pop()
            left = stack.pop()
            if token == 'AND' or token == 'and':
                stack.append(left & right)
            elif token == 'OR' or token == 'or':
                stack.append(left | right)
    return stack.pop() if stack else set()


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
        # 记录查询开始时间
        start_time = time.time()
        tokens = tokenize(expression)
        postfix_tokens = infix_to_postfix(tokens)
        result_ids = evaluate_postfix(postfix_tokens, compressed_file_path, vocabulary_file_path, all_ids)
        # 计算查询耗时
        elapsed_time = time.time() - start_time

        if result_ids:
            print("查询结果：\n")
            display_results(result_ids, words_df)
        else:
            print("没有符合条件的结果。")

        print(f"查询耗时：{elapsed_time:.6f} 秒")

        cont = input("是否继续查询？(Y/N): ")
        if cont.strip().lower() != 'y':
            print("感谢您的使用！")
            break  # 退出循环，结束程序

if __name__ == "__main__":
    main()
