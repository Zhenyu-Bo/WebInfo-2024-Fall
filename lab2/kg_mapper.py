# kg_mapper.py

from extract import load_movie_id_map


def create_entity_relation_mappings(triple_list, movie_id_map):
    """
    为实体和关系创建索引映射。

    参数：
    - triple_list: 原始三元组列表。
    - movie_id_map: 原始电影ID到索引的映射字典。

    返回：
    - entity_id_map: 实体到索引的映射字典。
    - relation_id_map: 关系到索引的映射字典。
    """
    entity_id_map = {}
    relation_id_map = {}
    current_entity_id = max(movie_id_map.values()) + 1  # 其他实体的起始索引
    current_relation_id = 0

    for h, r, t in triple_list:
        # 映射头实体
        if h not in movie_id_map and h not in entity_id_map:
            entity_id_map[h] = current_entity_id
            current_entity_id += 1
        # 映射尾实体
        if t not in movie_id_map and t not in entity_id_map:
            entity_id_map[t] = current_entity_id
            current_entity_id += 1
        # 映射关系
        if r not in relation_id_map:
            relation_id_map[r] = current_relation_id
            current_relation_id += 1

    return entity_id_map, relation_id_map

def map_triples_to_ids(triple_list, movie_id_map, entity_id_map, relation_id_map):
    """
    将三元组映射为索引表示。

    参数：
    - triple_list: 原始三元组列表。
    - movie_id_map: 原始电影ID到索引的映射字典。
    - entity_id_map: 其他实体到索引的映射字典。
    - relation_id_map: 关系到索引的映射字典。

    返回：
    - mapped_triples: 由索引值组成的三元组列表。
    """
    mapped_triples = []
    for h, r, t in triple_list:
        # 映射头实体
        if h in movie_id_map:
            head_id = movie_id_map[h]
        else:
            head_id = entity_id_map[h]

        # 映射尾实体
        if t in movie_id_map:
            tail_id = movie_id_map[t]
        else:
            tail_id = entity_id_map[t]

        # 映射关系
        relation_id = relation_id_map[r]

        mapped_triples.append((head_id, relation_id, tail_id))

    return mapped_triples

def save_mapped_triples(mapped_triples, output_path):
    """
    保存映射后的三元组到文件。

    参数：
    - mapped_triples: 由索引值组成的三元组列表。
    - output_path: 输出文件路径。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for head_id, relation_id, tail_id in mapped_triples:
            f.write(f"{head_id}\t{relation_id}\t{tail_id}\n")
    print(f"映射后的知识图谱已保存到 {output_path}")

def main():
    # 文件路径
    MOVIE_ID_MAP_PATH = 'data/movie_id_map.txt'
    DOUBAN_TO_FB_PATH = 'data/douban2fb.txt'
    KG_INPUT_PATH = 'data/kg_filtered.txt.gz'  # 之前生成的知识图谱文件路径
    KG_OUTPUT_PATH = 'baseline/data/Douban/kg_final.txt'

    # 加载电影ID映射
    movie_id_map = load_movie_id_map(MOVIE_ID_MAP_PATH, DOUBAN_TO_FB_PATH)
    num_of_movies = max(movie_id_map.values()) + 1

    # 加载知识图谱三元组
    triple_list = []
    with open(KG_INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triple_list.append((h, r, t))

    # 创建实体和关系的索引映射
    entity_id_map, relation_id_map = create_entity_relation_mappings(triple_list, movie_id_map)
    num_of_entities = num_of_movies + len(entity_id_map)
    num_of_relations = len(relation_id_map)
    print(f"电影总数：{num_of_movies}，实体总数：{num_of_entities}，关系总数：{num_of_relations}")

    # 将三元组映射为索引表示
    mapped_triples = map_triples_to_ids(triple_list, movie_id_map, entity_id_map, relation_id_map)

    # 保存映射后的三元组
    save_mapped_triples(mapped_triples, KG_OUTPUT_PATH)

if __name__ == '__main__':
    main()