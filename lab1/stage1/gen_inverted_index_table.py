# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-06

import ast
import pandas as pd
from math import sqrt


def read_words_from_csv(file_path):
    """
    读取 CSV 文件，提取所有单词和文档内容。

    参数：
    - file_path: CSV 文件路径。

    返回：
    - all_words: 所有词项的集合。
    - documents: 字典，键为文档 ID，值为该文档包含的单词集合。
    """
    data = pd.read_csv(file_path, dtype={'id': int, 'words': str})
    all_words = set()
    documents = {}
    for idx in range(len(data)):
        words = ast.literal_eval(data.at[idx, 'words'])
        doc_id = data.at[idx, 'id']
        documents[doc_id] = words
        all_words.update(words)
    return all_words, documents


def generate_inverted_index_table(all_words, documents, output_file):
    """
    生成倒排索引表并保存为 CSV 文件。

    参数：
    - all_words: 集合，包含所有词项。
    - documents: 字典，键为文档 ID，值为该文档的标签集合。
    - output_file: 输出的 CSV 文件路径。
    """
    inverted_index_table = []
    for word in all_words:
        doc_ids = [doc_id for doc_id, words in documents.items() if word in words]
        doc_ids_sorted = sorted(doc_ids)
        num_docs = len(doc_ids_sorted)
        l = int(sqrt(num_docs))  # 设置跳表的间隔
        skip_table = []
        if num_docs > l:
            for i in range(num_docs):
                if i % l == 0:
                    if i < num_docs - l:
                        skip_info = {'index': i + l, 'value': doc_ids_sorted[i + l]}
                    else:
                        # 最后一个跳表指针指向末尾
                        skip_info = {'index': num_docs - 1, 'value': doc_ids_sorted[num_docs - 1]}
                    skip_table.append(skip_info)
        else:
            skip_info = {'index': None, 'value': None}
            skip_table.append(skip_info)
        inverted_index_table.append({'word': word, 'id_list': doc_ids_sorted, 'skip_table': skip_table})
    pd.DataFrame(inverted_index_table).to_csv(output_file, index=False)


def main():
    # 处理书籍数据
    all_book_words, book_documents = read_words_from_csv("data/book_output.csv")
    generate_inverted_index_table(all_book_words, book_documents, "data/book_inverted_index_table.csv")
    print("书籍倒排索引表已成功生成")

    # 处理电影数据
    all_movie_words, movie_documents = read_words_from_csv("data/movie_output.csv")
    generate_inverted_index_table(all_movie_words, movie_documents, "data/movie_inverted_index_table.csv")
    print("电影倒排索引表已成功生成")


if __name__ == "__main__":
    main()
