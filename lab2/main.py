# main.py
# Author: Zhenyu Bo

from extract import load_base_entities, load_entities_from_triples
from kg_builder import build_kg
from kg_filter import filter_kg

def main():
    # 定义文件路径
    ENTITY_FILE_PATH = 'data/douban2fb.txt'
    FREEBASE_PATH = 'data/freebase_douban.gz'
    KG_OUTPUT_PATH = 'data/kg_initial.txt.gz'
    KG_FILTERED_OUTPUT_PATH = 'data/kg_filtered.txt'

    # 加载基础实体
    base_entities = load_base_entities(ENTITY_FILE_PATH)

    # 构建初始知识图谱
    triple_list = build_kg(FREEBASE_PATH, KG_OUTPUT_PATH, base_entities)

    # 统计并验证实体和关系
    entities_set, relations_set = load_entities_from_triples(triple_list)
    for entity in base_entities:
        assert entity in entities_set
    print("知识图谱验证完成。")
    print("知识图谱包含三元组数量：", len(triple_list))
    print("初始知识图谱包含实体数量：", len(entities_set))
    print("初始知识图谱包含关系数量：", len(relations_set))

    # 过滤知识图谱
    min_entity_freq = 10
    min_relation_freq = 50
    filtered_triples = filter_kg(triple_list, min_entity_freq, min_relation_freq)

    # 保存过滤后的知识图谱
    with open(KG_FILTERED_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        for triple in filtered_triples:
            f_out.write('\t'.join(triple) + '\n')


def main_2():
    """基于第一跳子图过滤后的实体集合，构建第二跳子图。"""
    TRIPLES_PATH = 'data/kg_filtered.txt'
    FREEBASE_PATH = 'data/freebase_douban.gz'
    KG_OUTPUT_PATH = 'data/kg_2.txt.gz'
    KG_FILTERED_OUTPUT_PATH = 'data/kg_2_filtered.txt'
    
    # 从文件中加载三元组
    triple_list = []
    with open(TRIPLES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triple_list.append((h, r, t))
    # 从三元组中提取实体集合
    entities_set_1, _ = load_entities_from_triples(triple_list)
    print("实体集合加载完成。")
    # 构建第二跳子图
    triple_list = build_kg(FREEBASE_PATH, KG_OUTPUT_PATH, entities_set_1)
    # 统计并验证实体和关系
    entities_set_2, relations_set = load_entities_from_triples(triple_list)
    for entity in entities_set_1:
        assert entity in entities_set_2
    print("知识图谱验证完成。")
    print("知识图谱包含三元组数量：", len(triple_list))
    print("知识图谱包含实体数量：", len(entities_set_2))
    print("知识图谱包含关系数量：", len(relations_set))
    # 过滤知识图谱
    min_entity_freq = 10
    min_relation_freq = 50
    filtered_triples = filter_kg(triple_list, min_entity_freq, min_relation_freq)
    # 保存过滤后的知识图谱
    with open(KG_FILTERED_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        for triple in filtered_triples:
            f_out.write('\t'.join(triple) + '\n')

if __name__ == '__main__':
    main()
    # main_2()
