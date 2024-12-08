# extract.py
# Author: Zhenyu Bo

def load_base_entities(douban2fb_path):
    """
    获得电影 ID 在 Freebase 中对应的实体集合。
    """
    base_entities = set()
    with open(douban2fb_path, 'r', encoding='utf-8') as f:
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

def load_movie_id_map(movie_id_map_path, douban2fb_path):
    """
    加载电影ID映射关系。
    """
    douban_id_to_entity_map = load_id_to_entity_map(douban2fb_path)
    movie_id_map = {}
    with open(movie_id_map_path, 'r', encoding='utf-8') as f:
        for line in f:
            douban_id, mapped_id = line.strip().split('\t')
            movie_uri = douban_id_to_entity_map[douban_id]
            movie_id_map[movie_uri] = int(mapped_id)
    return movie_id_map

def load_id_to_entity_map(douban2fb_path):
    """
    加载豆瓣电影ID到实体的映射关系。
    """
    douban_id_to_entity_map = {}
    with open(douban2fb_path, 'r', encoding='utf-8') as f:
        for line in f:
            douban_id, entity = line.strip().split()
            entity_uri = f"<http://rdf.freebase.com/ns/{entity}>"
            douban_id_to_entity_map[douban_id] = entity_uri
    return douban_id_to_entity_map