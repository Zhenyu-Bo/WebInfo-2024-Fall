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
    # term_positions = {}
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
    with open(posting_list_file, 'w', encoding='utf-8') as f_postings:
        for term in terms:
            id_list = posting_lists[term]
            posting_line = ' '.join(map(str, id_list)) + '\n'
            f_postings.write(posting_line)
            posting_pointers[term] = current_offset
            current_offset += len(posting_line.encode('utf-8'))

    # 创建词项表并保存
    with open(term_table_file, 'w', encoding='utf-8', newline='') as f_term_table:
        writer = csv.writer(f_term_table)
        writer.writerow(['doc_freq', 'posting_ptr', 'term_ptr'])
        for term in terms:
            doc_freq = len(posting_lists[term])
            posting_ptr = posting_pointers[term]
            term_ptr = term_pointers[term]
            writer.writerow([doc_freq, posting_ptr, term_ptr])


def load_term_table(term_table_file_path):
    """
    加载词项表，将其内容存储在一个字典中。

    参数：
    - term_table_file_path: 词项表文件路径，包含文档频率、倒排指针和词项指针。

    返回：
    - term_table: 词典字典，键为词项，值为包含文档频率、倒排指针和词项指针的字典。
    """
    term_table = {}
    try:
        with open(term_table_file_path, "r", encoding="UTF-8") as f_term_table:
            csv_reader = csv.DictReader(f_term_table)
            for row in csv_reader:
                doc_freq = int(row['doc_freq'])
                posting_ptr = int(row['posting_ptr'])
                term_ptr = int(row['term_ptr'])
                term_table[term_ptr] = {
                    'doc_freq': doc_freq,
                    'posting_ptr': posting_ptr,
                    'term_ptr': term_ptr
                }
    except IOError as e:
        print(f"无法打开或读取词项表文件: {e}")
    except ValueError as e:
        print(f"解析词项表数据失败: {e}")
    return term_table


def load_term_string(term_string_file_path):
    """
    加载词项字符串。
    """
    try:
        with open(term_string_file_path, "r", encoding="UTF-8") as f_term_string:
            term_string = f_term_string.read()
        return term_string
    except IOError as e:
        print(f"无法打开或读取词项字符串文件: {e}")
        return ""


def query_posting_list(word, term_string, term_table, posting_list_file_path):
    """
    查询特定词项的倒排索引列表。

    参数：
    - word: 目标词项。
    - term_string: 词项字符串。
    - term_table: 词典字典，包含词项及其相关指针。
    - posting_list_file_path: 倒排表文件路径，包含所有倒排列表项。

    返回：
    - doc_ids: 解压缩后的文档ID列表。如果词项不存在或查询失败，返回空列表。
    """
    # 查找词项指针
    term_ptr = None
    # next_term_ptr = None
    for ptr in sorted(term_table.keys()):
        if term_string[ptr:ptr+len(word)] == word:
            term_ptr = ptr
            # next_term_ptr = min([p for p in term_table.keys() if p > ptr], default=len(term_string))
            break

    if term_ptr is None:
        print(f"词项 '{word}' 不存在于词项表中。")
        return []

    posting_ptr = term_table[term_ptr]['posting_ptr']

    # 打开倒排表文件，找到指定位置
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
            posting_str = posting_bytes.decode('utf-8')
            # 将倒排列表字符串转换为整数列表
            doc_ids = list(map(int, posting_str.strip().split()))
            return doc_ids
    except IOError as e:
        print(f"无法打开或读取倒排表文件: {e}")
        return []
    except Exception as e:
        print(f"解析倒排表时发生错误: {e}")
        return []


def main():
    # 定义文件路径
    input_csv = 'data/book_inverted_index_table.csv'
    term_string_file = 'data/term_string.txt'
    term_table_file = 'data/term_table.csv'
    posting_list_file = 'data/posting_list.txt'

    # 处理倒排索引表
    process_inverted_index(input_csv, term_string_file, term_table_file, posting_list_file)
    print('词典和倒排表已成功生成。')

    # 定义文件路径
    term_string_file_path = 'data/term_string.txt'
    term_table_file_path = 'data/term_table.csv'
    posting_list_file_path = 'data/posting_list.txt'

    # 检查文件是否存在
    if not os.path.exists(term_string_file_path):
        print(f"词项字符串文件 '{term_string_file_path}' 不存在。")
        return
    if not os.path.exists(term_table_file_path):
        print(f"词项表文件 '{term_table_file_path}' 不存在。")
        return
    if not os.path.exists(posting_list_file_path):
        print(f"倒排表文件 '{posting_list_file_path}' 不存在。")
        return

    # 加载词项字符串和词项表
    term_string = load_term_string(term_string_file_path)
    term_table = load_term_table(term_table_file_path)
    if not term_string or not term_table:
        print("词项字符串或词项表加载失败或为空。")
        return

    while True:
        # 用户输入词项
        word = input("请输入要查询的词项（输入 'exit' 退出）：").strip()
        if word.lower() == 'exit':
            print("退出查询。")
            break

        # 查询倒排列表
        doc_ids = query_posting_list(word, term_string, term_table, posting_list_file_path)

        if doc_ids:
            print(f"词项 '{word}' 对应的文档 ID 列表（共 {len(doc_ids)} 个文档）:")
            print(doc_ids)
        else:
            print(f"词项 '{word}' 没有对应的文档或查询失败。")


if __name__ == '__main__':
    main()
