# kg_filter.py
# Author: Zhenyu Bo

def filter_kg(triple_list, min_entity_freq, min_relation_freq):
    """
    过滤知识图谱，根据频率和关系数量限制。

    参数：
    - triple_list: 三元组列表。
    - min_freq: 最小出现频率。
    - max_relations: 最大关系数量。

    返回：
    - filtered_triples: 过滤后的三元组列表。
    """
    relation_count = {}
    for _, r, _ in triple_list:
        relation_count[r] = relation_count.get(r, 0) + 1
        
    # 过滤实体
    entity_count = {}
    for h, _, t in triple_list:
        entity_count[h] = entity_count.get(h, 0) + 1
        entity_count[t] = entity_count.get(t, 0) + 1
    selected_entities = {e for e, count in entity_count.items() if count >= min_entity_freq}

    # 过滤关系
    selected_relations = {r for r, count in relation_count.items() if count >= min_relation_freq}
    # if len(selected_relations) > max_relations:
    #     selected_relations = set(list(selected_relations)[:max_relations])

    # 过滤三元组
    filtered_triples = [triple for triple in triple_list if triple[1] in selected_relations and triple[0] in selected_entities and triple[2] in selected_entities]
    print("知识图谱过滤完成，剩余三元组数量：", len(filtered_triples))
    return filtered_triples