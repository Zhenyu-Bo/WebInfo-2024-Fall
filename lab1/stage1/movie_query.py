# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-06

import ast
import csv
import pandas as pd
import synonyms
from utils import utils


def not_merge_index_table(word_ids: list[str], all_ids: list[str]) -> list[str]:
    """执行索引表的 NOT 操作，返回所有 ID 中不包含指定词的 ID 列表。"""
    return list(set(all_ids) - set(word_ids))


def and_merge_index_table(word1_ids: list[str], word2_ids: list[str]) -> list[str]:
    """执行索引表的 AND 操作，返回两个词共同出现的 ID 列表。"""
    return list(set(word1_ids) & set(word2_ids))


def or_merge_index_table(word1_ids: list[str], word2_ids: list[str]) -> list[str]:
    """执行索引表的 OR 操作，返回两个词任一出现的 ID 列表。"""
    return list(set(word1_ids) | set(word2_ids))


def get_movie_synonym_words() -> list[tuple[str, str]]:
    """获取用户输入的查询词及其对应的近义词。"""
    sentence = input("请输入查询的语句：\n")
    stopwords = {'导演', '编剧', '主演', '类型', '制片国家/地区', '又名', 'IMDb', '语言'}
    util = utils()
    words = util.split(sentence)
    words = words - stopwords  # 去除停用词
    synonym_pairs = []
    for word in words:
        nearby_words = synonyms.nearby(word)
        if nearby_words[0]:
            synonym_word = nearby_words[0][1]
            synonym_pairs.append((word, synonym_word))
        else:
            synonym_pairs.append((word, word))
    return synonym_pairs


def generate_word_id_list(synonym_pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """生成每个词及其近义词对应的 ID 列表。"""
    file_name = "../data/movie_inverted_list.csv"
    index_table = {}
    query_word_ids = {}
    with open(file_name, encoding="utf8", mode='r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # 跳过表头
        for row in csv_reader:
            index_table[row[0]] = ast.literal_eval(row[1])

    for word, synonym in synonym_pairs:
        word_ids = index_table.get(word, [])
        synonym_ids = index_table.get(synonym, [])
        combined_ids = or_merge_index_table(word_ids, synonym_ids)
        query_word_ids[word] = combined_ids

    return query_word_ids


def natural_language_process() -> None:
    """处理自然语言查询，输出查询结果。"""
    synonym_pairs = get_movie_synonym_words()
    query_word_ids = generate_word_id_list(synonym_pairs)
    id_count = {}

    for ids in query_word_ids.values():
        for movie_id in ids:
            id_count[movie_id] = id_count.get(movie_id, 0) + 1

    # 按出现次数降序排序
    sorted_results = sorted(id_count.items(), key=lambda x: x[1], reverse=True)
    movie_info = pd.read_csv("../data/selected_movie_top_1200_data_tag.csv")

    for i, (movie_id, _) in enumerate(sorted_results):
        if i >= 5:
            break
        print("-" * 20 + f" 查询结果 {i + 1} " + "-" * 20)
        print(f"ID:\n\t{movie_id}")
        movie_data = movie_info.loc[movie_info['Movie'] == int(movie_id)]
        if not movie_data.empty:
            movie_tags = movie_data.iloc[0]['Tags']
            print(f"标签: \n\t{movie_tags}")
        else:
            print("未找到电影信息。")


if __name__ == '__main__':
    natural_language_process()
