import pandas as pd
import ast
import jieba
import thulac
from tqdm import tqdm  # 导入 tqdm 库
import os

def process_tags(tags_str: str, mode='jieba') -> list[str]:
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
        thu1 = thulac.thulac(seg_only=True)
        for tag in tags:
            # jieba 分词
            words = jieba.lcut(tag)
            words_set.update(words)
            # THULAC 分词
            words = thu1.cut(tag, text=True).split()
            words_set.update(words)
    else:
        raise ValueError("Invalid mode. Choose from 'jieba', 'thulac', or 'both'.")

    return list(words_set)

def process_file(input_file: str, output_file: str, id_name: str, mode='jieba') -> None:
    data = pd.read_csv(input_file)
    ids = data[id_name].tolist()
    tags_list = data['Tags'].tolist()
    
    processed_data = []
    # 添加进度条
    for doc_id, tags_str in tqdm(zip(ids, tags_list), total=len(ids), desc=f"处理文件 {input_file}"):
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
    process_file('../data/selected_book_top_1200_data_tag.csv', 'split_output/book_words.csv', 'Book', mode)
    # 处理电影数据
    process_file('../data/selected_movie_top_1200_data_tag.csv', 'split_output/movie_words.csv', 'Movie', mode)