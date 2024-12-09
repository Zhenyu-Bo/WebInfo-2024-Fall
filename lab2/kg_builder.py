# kg_builder.py
# Author: Zhenyu Bo

import gzip
from tqdm import tqdm

def build_kg(freebase_path, kg_output_path, entities_set):
    """
    构建知识图谱，提取与实体集合相关的三元组。

    参数：
    - freebase_path: Freebase 数据路径。
    - kg_output_path: 知识图谱输出路径。
    - entities_set: 实体集合。
    """
    triple_list = []
    with gzip.open(freebase_path, 'rb') as f_in, \
            gzip.open(kg_output_path, 'wb') as f_out:
        # 使用 tqdm 包装文件对象，以显示进度条
        for line in tqdm(f_in, desc="Processing", unit=" lines"):
            line = line.strip()
            h, r, t = line.decode().split('\t')[:3]
            if h in entities_set or t in entities_set:
                f_out.write((h + '\t' + r + '\t' + t + '\n').encode())
                triple_list.append((h, r, t))
    print("知识图谱构建完成，共包含三元组数量：", len(triple_list))
    return triple_list