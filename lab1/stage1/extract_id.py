# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-14

"""
从数据集中提取 ID 列，保存到文本文件中。
"""

import pandas as pd

def extract_ids(input_csv, output_txt):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv)

    # 提取 ID 列（假设 ID 在第一列）
    ids = df.iloc[:, 0]

    # 将 ID 写入文本文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        for id in ids:
            f.write(f"{id}\n")


def main():
    # 处理书籍 ID
    extract_ids('../data/selected_book_top_1200_data_tag.csv', 'data/Book_id.txt')
    print('Book_id.txt 文件已生成。')

    # 处理电影 ID
    extract_ids('../data/selected_movie_top_1200_data_tag.csv', 'data/Movie_id.txt')
    print('Movie_id.txt 文件已生成。')

if __name__ == "__main__":
    main()
