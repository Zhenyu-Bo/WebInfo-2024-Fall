import pandas as pd
import ast
import jieba
import thulac
from collections import defaultdict
import difflib
from tqdm import tqdm  # 导入 tqdm 库
import os

# 加载中文停用词表
with open('data/baidu_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())

# 加载同义词词典，词典格式为 'Aa01A01= 人 士 人物 人士 人氏 人选'
synonym_dict = {}
with open('data/dict_synonym.txt', 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="加载同义词词典"):
        line = line.strip()
        if '=' in line:
            _, synonyms_str = line.split('=', 1)
            words = synonyms_str.strip().split()
            main_word = words[0]
            for word in words:
                synonym_dict[word] = main_word

def correct_word(word, vocabulary):
    # 使用编辑距离进行纠错
    closest_matches = difflib.get_close_matches(word, vocabulary, n=1, cutoff=0.8)
    if closest_matches:
        return closest_matches[0]
    else:
        return word

def process_tags(tags_str, mode='jieba'):
    # 将字符串形式的集合转换为实际的集合对象
    tags = ast.literal_eval(tags_str)
    words_set = set()

    if mode == 'jieba':
        # 使用 jieba 分词
        for tag in tags:
            words = jieba.lcut(tag)
            words_set.update(words)
    elif mode == 'thulac':
        # 使用 THULAC 分词
        thu1 = thulac.thulac(seg_only=True)
        for tag in tags:
            words = thu1.cut(tag, text=True).split()
            words_set.update(words)
    elif mode == 'both':
        # 同时使用 jieba 和 THULAC 分词
        # jieba 分词
        for tag in tags:
            words = jieba.lcut(tag)
            words_set.update(words)
        # THULAC 分词
        thu1 = thulac.thulac(seg_only=True)
        for tag in tags:
            words = thu1.cut(tag, text=True).split()
            words_set.update(words)
    else:
        raise ValueError("Invalid mode. Choose from 'jieba', 'thulac', or 'both'.")

    # 去除停用词
    words_set = words_set - stopwords

    # 合并同义词
    merged_words = set()
    for word in words_set:
        if word in synonym_dict:
            merged_words.add(synonym_dict[word])
        else:
            merged_words.add(word)

    # 构建词汇表用于纠错
    vocabulary = merged_words.copy()

    # 纠错
    final_words = set()
    for word in merged_words:
        corrected_word = correct_word(word, vocabulary)
        final_words.add(corrected_word)

    return list(final_words)

def process_file(input_file, output_file, id_name, mode='jieba'):
    data = pd.read_csv(input_file)
    ids = data[id_name].tolist()
    tags_list = data['Tags'].tolist()

    processed_data = []
    for doc_id, tags_str in tqdm(zip(ids, tags_list), desc=f"处理文件 {input_file}", total=len(ids)):
        words = process_tags(tags_str, mode)
        processed_data.append({'id': doc_id, 'words': words})

    df = pd.DataFrame(processed_data)

    # 获取输出文件的目录路径
    output_dir = os.path.dirname(output_file)
    # 如果目录不存在，创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # 设置分词模式，可选 'jieba'，'thulac'，'both'
    mode = 'jieba'

    # 处理书籍数据
    process_file('../data/selected_book_top_1200_data_tag.csv', 'split_output/pro/book_words.csv', 'Book', mode)
    # 处理电影数据
    process_file('../data/selected_movie_top_1200_data_tag.csv', 'split_output/pro/movie_words.csv', 'Movie', mode)