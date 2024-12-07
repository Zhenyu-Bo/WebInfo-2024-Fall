# extract.py

def load_base_entities(entity_file_path):
    """
    加载基础实体集合。

    参数：
    - entity_file_path: 基础实体文件路径。

    返回：
    - base_entities: 基础实体的集合。
    """
    base_entities = set()
    with open(entity_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            _, entity = line.strip().split()
            entity_uri = f"<http://rdf.freebase.com/ns/{entity}>"
            base_entities.add(entity_uri)
    print("基础实体加载完成。")
    return base_entities

def load_entities_from_triples(triple_list):
    """
    从三元组列表中提取实体和关系集合。

    参数：
    - triple_list: 三元组列表。

    返回：
    - entities_set: 实体集合。
    - relations_set: 关系集合。
    """
    entities_set = set()
    relations_set = set()
    for h, r, t in triple_list:
        entities_set.update([h, t])
        relations_set.add(r)
    return entities_set, relations_set