# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-07

"""
压缩倒排索引表并拆分为词典和倒排表文件。
"""

import json
import csv


def variable_byte_encode(gaps: list[int]) -> bytes:
    """
    使用可变字节编码（Variable Byte Encoding）对文档间距进行压缩。

    参数：
    - gaps: 文档间距列表。

    返回：
    - 压缩后的字节序列。
    """
    encoded_bytes = []
    for gap in gaps:
        byte_chunks = []
        while gap >= 128:
            byte_chunks.append(gap % 128)
            gap = gap // 128
        byte_chunks.append(gap + 128)  # 设置最后一个字节的最高位为1
        # 由于要按大端顺序存储，反转字节顺序
        byte_chunks = byte_chunks[::-1]
        encoded_bytes.extend(byte_chunks)
    return bytes(encoded_bytes)

def calculate_gaps(index_list: list[int]) -> list[int]:
    """
    计算文档间距列表。

    假设输入的 `index_list` 是一个递增的文档 ID 列表，返回每个文档 ID 与前一个文档 ID 之间的差值列表。

    参数：
    - index_list: 递增的文档 ID 列表。

    返回：
    - 文档间距列表。
    """
    gaps = []
    previous_id = 0  # 假设第一个文档的前一个 ID 为 0

    for doc_id in index_list:
        gap = doc_id - previous_id
        gaps.append(gap)
        previous_id = doc_id

    return gaps


def compress(index_list: list[int]) -> bytes:
    """
    压缩文档 ID 列表为字节序列。

    使用文档间距替代 ID 并应用可变字节编码进行压缩。

    参数：
    - index_list: 递增的文档 ID 列表。

    返回：
    - 压缩后的字节序列。
    """
    # 计算文档间距
    gaps = calculate_gaps(index_list)

    # 使用可变字节编码压缩间距列表
    compressed_data = variable_byte_encode(gaps)

    return compressed_data


def save_compressed_data(csv_file_path: str, compressed_file_path: str, vocabulary_file_path: str) -> None:
    """
    保存压缩后的倒排索引和词汇表。

    参数：
    - csv_file_path: 生成的倒排索引 CSV 文件路径（例如 "book_inverted_index_table.csv"）。
    - compressed_file_path: 压缩后的倒排表二进制文件路径。
    - vocabulary_file_path: 词汇表文件路径，包含词项及其在倒排表中的字节偏移量和长度。
    """
    # vocabulary = []
    current_offset = 0

    try:
        with open(compressed_file_path, "wb") as f_bin, \
             open(vocabulary_file_path, "w", encoding="UTF-8", newline='') as f_vocab, \
             open(csv_file_path, "r", encoding="UTF-8") as f_csv:

            csv_reader = csv.DictReader(f_csv)
            vocab_writer = csv.writer(f_vocab)
            # 写入词汇表的表头
            vocab_writer.writerow(['word', 'offset', 'length'])

            for row in csv_reader:
                word = row['word']
                # 解析 id_list（假设以列表字符串形式存储）
                id_list_str = row['id_list']
                id_list = json.loads(id_list_str)

                # 压缩文档 ID 列表
                compressed_bytes = compress(id_list)

                # 写入倒排表
                f_bin.write(compressed_bytes)

                # 记录词典信息
                vocab_writer.writerow([word, current_offset, len(compressed_bytes)])
                current_offset += len(compressed_bytes)

    except IOError as e:
        print(f"文件操作失败: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON 解码失败: {e}")


if __name__ == "__main__":
    # 处理书籍倒排索引表
    csv_file_path = "data/book_inverted_index_table.csv"
    compressed_file_path = "data/book_inverted_index_compressed.bin"
    vocabulary_file_path = "data/book_vocabulary.csv"

    # 保存压缩后的倒排索引和词汇表
    save_compressed_data(
        csv_file_path,
        compressed_file_path,
        vocabulary_file_path
    )

    # 处理电影倒排索引表
    csv_file_path = "data/movie_inverted_index_table.csv"
    compressed_file_path = "data/movie_inverted_index_compressed.bin"
    vocabulary_file_path = "data/movie_vocabulary.csv"

    save_compressed_data(
        csv_file_path,
        compressed_file_path,
        vocabulary_file_path
    )
