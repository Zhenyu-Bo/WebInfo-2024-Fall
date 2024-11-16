# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-07
"""
将词典视作单一字符串，从倒排索引表中提取词典和倒排表项，节省词项存储空间。
"""

import csv
import json
import os

def process_inverted_index(input_csv, term_string_file, term_table_file, posting_list_file):
    """
    处理倒排索引表，生成词项字符串、词项表和倒排表文件。
    参数：
    - input_csv: 倒排索引表的 CSV 文件路径。
    - term_string_file: 保存词项字符串的文件路径。
    - term_table_file: 保存词项表的文件路径。
    - posting_list_file: 保存倒排表的文件路径。
    """
    # 读取倒排索引表
    terms = []
    posting_lists = {}
    with open(input_csv, 'r', encoding='utf-8') as f_csv:
        reader = csv.DictReader(f_csv)
        for row in reader:
            word = row['word']
            id_list = json.loads(row['id_list'])
            terms.append(word)
            posting_lists[word] = id_list

    # 将词项按照字典序排序
    terms.sort()

    # 创建词项字符串并记录每个词项的起始位置
    term_string = ''
    current_position = 0
    term_pointers = {}
    for term in terms:
        term_pointers[term] = current_position
        term_string += term
        current_position += len(term)

    # 保存词项字符串
    with open(term_string_file, 'w', encoding='utf-8') as f_term_string:
        f_term_string.write(term_string)

    # 保存倒排表并记录每个倒排列表的起始偏移量
    posting_pointers = {}
    current_offset = 0
    with open(posting_list_file, 'w', encoding='utf-8', newline='\n') as f_postings:  # 指定行结束符
        for term in terms:
            id_list = posting_lists[term]
            posting_line = ' '.join(map(str, id_list)) + '\n'  # 以空格分隔的文档ID列表
            f_postings.write(posting_line)
            posting_pointers[term] = current_offset
            current_offset += len(posting_line.encode('utf-8'))  # 计算字节长度

    # 创建词项表并保存（不包含 'word'）
    with open(term_table_file, 'w', encoding='utf-8', newline='') as f_term_table:
        writer = csv.writer(f_term_table)
        writer.writerow(['doc_freq', 'posting_ptr', 'term_ptr', 'term_length'])
        for term in terms:
            doc_freq = len(posting_lists[term])
            posting_ptr = posting_pointers[term]
            term_ptr = term_pointers[term]
            term_length = len(term)
            writer.writerow([doc_freq, posting_ptr, term_ptr, term_length])

def load_term_table(term_table_file_path):
    """
    加载词项表，将其内容存储在一个列表中。
    参数：
    - term_table_file_path: 词项表文件路径，包含文档频率、倒排指针、词项指针和词项长度。
    返回：
    - term_table: 词项表列表，按字典序排序。
    """
    term_table = []
    try:
        with open(term_table_file_path, "r", encoding="UTF-8") as f_term_table:
            csv_reader = csv.DictReader(f_term_table)
            for row in csv_reader:
                term_table.append({
                    'doc_freq': int(row['doc_freq']),
                    'posting_ptr': int(row['posting_ptr']),
                    'term_ptr': int(row['term_ptr']),
                    'term_length': int(row['term_length'])
                })
    except IOError as e:
        print(f"无法打开或读取词项表文件: {e}")
    except ValueError as e:
        print(f"解析词项表数据失败: {e}")
    return term_table

def load_term_string(term_string_file_path):
    """
    加载词项字符串。
    参数：
    - term_string_file_path: 词项字符串文件路径。
    返回：
    - term_string: 词项字符串。
    """
    try:
        with open(term_string_file_path, "r", encoding="UTF-8") as f_term_string:
            term_string = f_term_string.read()
        return term_string
    except IOError as e:
        print(f"无法打开或读取词项字符串文件: {e}")
        return ""

def binary_search_term(term_string, term_table, word):
    """
    在词项字符串中使用二分查找定位词项。
    参数：
    - term_string: 词项字符串。
    - term_table: 词项表列表。
    - word: 目标词项。
    返回：
    - index: 词项在 term_table 中的索引，如果未找到则返回 -1。
    """
    left = 0
    right = len(term_table) - 1
    while left <= right:
        mid = (left + right) // 2
        term_ptr = term_table[mid]['term_ptr']
        term_length = term_table[mid]['term_length']
        mid_word = term_string[term_ptr:term_ptr + term_length]
        if mid_word == word:
            return mid
        elif mid_word < word:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def query_posting_list(word, term_string, term_table, posting_list_file_path):
    """
    查询特定词项的倒排索引列表。
    参数：
    - word: 目标词项。
    - term_string: 词项字符串。
    - term_table: 词项表列表，按字典序排序。
    - posting_list_file_path: 倒排表文件路径，包含所有倒排列表项。
    返回：
    - doc_ids: 文档ID列表。如果词项不存在或查询失败，返回空列表。
    """
    index = binary_search_term(term_string, term_table, word)
    if index == -1:
        print(f"词项 '{word}' 不存在于词项表中。")
        return []

    posting_ptr = term_table[index]['posting_ptr']

    try:
        with open(posting_list_file_path, "rb") as f_postings:
            f_postings.seek(posting_ptr)
            # 读取到下一个换行符为止，获取完整的倒排列表行
            posting_bytes = bytearray()
            while True:
                byte = f_postings.read(1)
                if not byte or byte == b'\n':
                    break
                posting_bytes += byte
            posting_line = posting_bytes.decode('utf-8')
            # 将倒排列表行拆分为整数
            doc_ids = list(map(int, posting_line.strip().split()))
            return doc_ids
    except IOError as e:
        print(f"无法打开或读取倒排表文件: {e}")
        return []
    except Exception as e:
        print(f"解析倒排表时发生错误: {e}")
        return []

def main():
    # 处理书籍数据
    # 定义文件路径
    book_input_csv = 'data/book_inverted_index_table.csv'
    book_term_string_file = 'data/book_term_string.txt'
    book_term_table_file = 'data/book_term_table.csv'
    book_posting_list_file = 'data/book_posting_list.txt'  # 使用文本文件

    # 处理倒排索引表
    process_inverted_index(book_input_csv, book_term_string_file, book_term_table_file, book_posting_list_file)
    print('书籍词典和倒排表已成功生成。')

    # 处理电影数据
    movie_input_csv = 'data/movie_inverted_index_table.csv'
    movie_term_string_file = 'data/movie_term_string.txt'
    movie_term_table_file = 'data/movie_term_table.csv'
    movie_posting_list_file = 'data/movie_posting_list.txt'  # 使用文本文件

    process_inverted_index(movie_input_csv, movie_term_string_file, movie_term_table_file, movie_posting_list_file)
    print('电影词典和倒排表已成功生成。')

    # 加载词项表和词项字符串
    book_term_table = load_term_table(book_term_table_file)
    book_term_string = load_term_string(book_term_string_file)
    book_posting_list_file_path = "data/book_posting_list.txt"

    while True:
        word = input("请输入要查询的词项（输入 'exit' 退出）：").strip()
        if word.lower() == 'exit':
            print("退出查询。")
            break

        # 查询词项的倒排列表
        book_doc_ids = query_posting_list(word, book_term_string, book_term_table, book_posting_list_file_path)
        if book_doc_ids:
            print(f"词项 '{word}' 的文档ID列表：{book_doc_ids}")
        else:
            print(f"词项 '{word}' 没有对应的文档或查询失败。")

if __name__ == '__main__':
    main()