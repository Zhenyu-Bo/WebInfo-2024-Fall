# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-08

"""
根据词典文件查找并解压缩特定词项的倒排索引。
"""

import csv
import os


def variable_byte_decode(byte_data: bytes) -> list[int]:
    """
    使用可变字节解码（Variable Byte Decoding）将字节序列解码为整数列表。

    参数：
    - byte_data: 压缩的字节序列。

    返回：
    - 解码后的整数列表。
    """
    numbers = []
    current_number = 0
    for byte in byte_data:
        if byte >= 128:
            # 当最高位为1时，这是一个数字的最后一个字节
            current_number = (current_number << 7) | (byte - 128)
            numbers.append(current_number)
            current_number = 0
        else:
            # 当最高位为0时，这是一个数字的中间字节
            current_number = (current_number << 7) | byte
    return numbers


def reconstruct_doc_ids(gaps: list[int]) -> list[int]:
    """
    根据文档间距列表重建原始的文档 ID 列表。

    参数：
    - gaps: 文档间距列表。

    返回：
    - 原始的文档 ID 列表。
    """
    doc_ids = []
    current_id = 0
    for gap in gaps:
        current_id += gap
        doc_ids.append(current_id)
    return doc_ids


def load_vocabulary(vocabulary_file_path: str) -> dict:
    """
    加载词典文件，将其内容存储在一个字典中。

    参数：
    - vocabulary_file_path: 词汇表文件路径，包含词项及其在倒排表中的字节偏移量和长度。

    返回：
    - 词典字典，键为词项，值为包含偏移量和长度的字典。
    """
    vocabulary = {}
    try:
        with open(vocabulary_file_path, "r", encoding="UTF-8") as f_vocab:
            csv_reader = csv.DictReader(f_vocab)
            for row in csv_reader:
                word = row['word']
                offset = int(row['offset'])
                length = int(row['length'])
                vocabulary[word] = {'offset': offset, 'length': length}
    except IOError as e:
        print(f"文件操作失败: {e}")
    except ValueError as e:
        print(f"数据解析失败: {e}")
    return vocabulary


def get_inverted_index_list(word: str, compressed_file_path="data/book_inverted_index_compressed.bin",
                            vocabulary_file_path="data/book_vocabulary.csv") -> list[int]:
    """
    根据给定的词项，从压缩的倒排表文件中提取并解压缩对应的倒排索引。

    参数：
    - word: 目标词项。
    - compressed_file_path: 压缩后的倒排表二进制文件路径。
    - vocabulary: 词典字典，包含词项及其在倒排表中的字节偏移量和长度。

    返回：
    - 解压缩后的文档 ID 列表。如果词项不存在，则返回空列表。
    """
    # 检查文件是否存在
    if not os.path.exists(compressed_file_path):
        print(f"压缩文件 '{compressed_file_path}' 不存在。")
        return
    if not os.path.exists(vocabulary_file_path):
        print(f"词典文件 '{vocabulary_file_path}' 不存在。")
        return

    # 加载词典
    vocabulary = load_vocabulary(vocabulary_file_path)

    if word not in vocabulary:
        print(f"词项 '{word}' 不存在于词典中。")
        return []

    offset = vocabulary[word]['offset']
    length = vocabulary[word]['length']

    try:
        with open(compressed_file_path, "rb") as f_bin:
            f_bin.seek(offset)
            compressed_bytes = f_bin.read(length)
            gaps = variable_byte_decode(compressed_bytes)
            doc_ids = reconstruct_doc_ids(gaps)
            return doc_ids
    except IOError as e:
        print(f"文件操作失败: {e}")
        return []
    except (Exception) as e:
        print(f"解压缩过程中发生错误: {e}")
        return []


def main():
    """
    主函数，根据用户输入的词项，解压缩并显示其倒排索引。
    """
    # 定义文件路径
    compressed_file_path = "data/book_inverted_index_compressed.bin"
    vocabulary_file_path = "data/book_vocabulary.csv"

    # 用户输入词项
    word = input("请输入要查询的词项: ").strip()

    # 获取并解压缩倒排索引
    doc_ids = get_inverted_index_list(word, compressed_file_path, vocabulary_file_path)

    if doc_ids:
        print(f"词项 '{word}' 对应的文档 ID 列表: {doc_ids}")
    else:
        print(f"词项 '{word}' 没有对应的文档或解压缩失败。")


if __name__ == "__main__":
    main()
