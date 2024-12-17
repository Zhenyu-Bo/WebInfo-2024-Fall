# Web信息处理与应用 Lab2

PB22081571 薄震宇	PB22111613 王翔辉	PB22020514 郭东昊

[TOC]

## 1. 文件组织





## 2. 实验任务

必做：

1. 根据实验一中提供的电影 ID 列表，匹配获得 Freebase 中对应的实体。
2. 从公开图谱中匹配从 Freebase 中抽取的电影实体有关系的实体和关系，生成知识图谱。为保障图谱质量，可以按一定的规则过滤图谱中的实体和关系。
3. 对图谱中的实体和关系做映射并根据映射关系将前面得到的知识图谱映射为由索引值组成的三元组。
4. 基于 baseline 框架代码，完成基于图谱嵌入的模型。需要完成 KG 的构建，实现 TransE 算法，采用多任务方式（KG 损失与 CF 损失相加）对模型进行更新。
5. 分析不同的设计的图谱嵌入方法对知识感知推荐性能的影响，对比分析基础推荐方法和知识感知推荐的实验结果差异。

选做：

1. 将多任务方式更改为迭代优化方式，即 KG 损失与 CF 损失迭代地对模型进行优化。
2. 调研相关综述4，思考如何改进自己的模型，并动手尝试。

## 3. 实验过程

### 3.1 提取实体

这里我们需要根据`douban2fb.txt`文件，匹配获得 Freebase 中对应的实体，一共有 578 个可匹配实体。

我们只需要读取`douban2fb.txt`文件，获取每一行的第二列（表示实体），为其加上前缀`<http://rdf.freebase.com/ns/`即表示电影在 Freebase 中对应的实体，再将这些实体放入一个集合中即可。代码如下：

```python3
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
```

### 3.2 构建知识图谱

这里我们需要以 578 个可匹配实体为起点，通过三元组关联，提取一跳可达的全部实体以形成新的起点集合。将重复两次该步骤所获得的全部实体及对应三元组合并用于后面实验的知识图谱子图。

对于每一跳知识图谱，我们首先需要从 Freebase 中提取所有包含实体的三元组，然后按照一定的规则过滤三元组。下面分别说明这两个步骤。

#### 3.2.1 提取初始图谱

这一步里我们需要从 Freebase 中提取所有包含实体集合中的实体的三元组并返回，考虑到存储空间有限，保存三元组文件时用`gzip`进行压缩，保存的文件以`.txt.gz`为后缀。代码如下：

```python3
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
```

对于第一跳，我们需要从第一步中得到的 578 个实体出发，提取将其作为头或尾与 Freebase 中的三元组，而对于第二跳，我们则需要从第一跳中得到的所有实体出发。

第一跳

#### 3.2.2 过滤知识图谱

为了保障图谱的质量，我们需要对前面得到的图谱进行过滤。我们的过滤条件有：

1. 实体需以`<http://rdf.freebase.com/ns/`为前缀。
2. 实体涉及到三元组至少有10个。
3. 关系至少在50个三元组中出现。

我们可以先遍历三元组，统计实体及关系出现的次数，然后根据条件过滤。

过滤实体的代码如下：

```python3
entity_count = {}
    for h, _, t in triple_list:
        entity_count[h] = entity_count.get(h, 0) + 1
        entity_count[t] = entity_count.get(t, 0) + 1
    selected_entities = {e for e, count in entity_count.items() if e.startswith('<http://rdf.freebase.com/ns/') and count >= min_entity_freq}
```

过滤关系的代码如下：
```python3
relation_count = {}
    for _, r, _ in triple_list:
        relation_count[r] = relation_count.get(r, 0) + 1
    selected_relations = {r for r, count in relation_count.items() if count >= min_relation_freq}
```

最后过滤得到实体在`selected_entities`中并且关系在`selected_relations`中的三元组：

```
filtered_triples = [triple for triple in triple_list if triple[1] in selected_relations and triple[0] in selected_entities and triple[2] in selected_entities]
```

第一跳得到的初始子图及过滤后的结果如下：

![1](figs/build_kg.png)

由于内存原因，在构建第二跳子图时程序总是中途终止，所以我们只执行了一跳。

### 3.3 图谱映射

这里我们需要根据`douban2fb.txt`中提供的电影 ID 与电影实体的映射关系以及`movie_id_map.txt`中提供的电影 ID 与索引值的映射关系，将电影实体映射到索引值上，并继续对其余实体和关系做映射。

#### 3.3.1 电影实体映射

首先我们需要以电影 ID 为媒介，对电影实体做映射。代码如下：

```python3
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
```

我们首先读取`douban2fb.txt`，将电影 ID 到实体的映射关系存在了字典`douban_id_to_entity_map`中，然后对于每一个电影 ID ，我们可以根据`movie_id_map.txt`获得其对应的索引值，将其对应的电影实体和索引值的映射存在字典`movie_index_map`中。

#### 3.1.2 其余实体和关系映射

对于除去电影实体的其余实体，可以接着电影实体的索引值继续做映射，即从`max(movie_index_map.values()) + 1`开始。对于关系，可以从索引 0 开始映射。

主要是通过下面的函数来得到字典：

```python3
def create_entity_relation_mappings(triple_list, movie_index_map):
    """
    为实体和关系创建索引映射。

    参数：
    - triple_list: 原始三元组列表。
    - movie_id_map: 电影实体到索引的映射字典。

    返回：
    - entity_id_map: 其他实体到索引的映射字典。
    - relation_id_map: 关系到索引的映射字典。
    """
    entity_id_map = {}
    relation_id_map = {}
    current_entity_id = max(movie_index_map.values()) + 1  # 其他实体的起始索引
    current_relation_id = 0

    for h, r, t in triple_list:
        # 映射头实体
        if h not in movie_index_map and h not in entity_id_map:
            entity_id_map[h] = current_entity_id
            current_entity_id += 1
        # 映射尾实体
        if t not in movie_index_map and t not in entity_id_map:
            entity_id_map[t] = current_entity_id
            current_entity_id += 1
        # 映射关系
        if r not in relation_id_map:
            relation_id_map[r] = current_relation_id
            current_relation_id += 1

    return entity_id_map, relation_id_map
```

然后根据得到的实体映射字典和关系映射字典，我们就可以将前面得到的电影图谱映射为由索引值组成的三元组：

```python3
def map_triples_to_ids(triple_list, movie_index_map, entity_id_map, relation_id_map):
    """
    将三元组映射为索引表示。

    参数：
    - triple_list: 原始三元组列表。
    - movie_index_map: 电影实体到索引的映射字典。
    - entity_id_map: 其他实体到索引的映射字典。
    - relation_id_map: 关系到索引的映射字典。

    返回：
    - mapped_triples: 由索引值组成的三元组列表。
    """
    mapped_triples = []
    for h, r, t in triple_list:
        # 映射头实体
        if h in movie_index_map:
            head_id = movie_index_map[h]
        else:
            head_id = entity_id_map[h]

        # 映射尾实体
        if t in movie_index_map:
            tail_id = movie_index_map[t]
        else:
            tail_id = entity_id_map[t]

        # 映射关系
        relation_id = relation_id_map[r]

        mapped_triples.append((head_id, relation_id, tail_id))

    return mapped_triples
```

最后将映射后的三元组保存到`baseline\data\Douban\kg_final.txt`文件中即可。

映射前的知识图谱大小为 6.21 MB，而映射后变为了 693 KB，可见**映射处理大幅压缩了存储空间**。

### 3.4 模型构建
#### 3.4.1  处理以得到kg_data、kg_dict、relation_dict
- 将三元组数据读入，构建逆向三元组，将原三元组和逆向三元组拼接为新的DataFrame，保存在 self.kg_data 中。
- 计算关系数，实体数和三元组的数量
- 根据 self.kg_data 构建字典 self.kg_dict ，其中key为h, value为tuple(t, r)，和字典 self.relation_dict，其中key为r, value为tuple(h, t)。
- 代码如下：

```python

    def construct_data(self, kg_data):
        '''
            kg_data 为 DataFrame 类型
        '''
        # 1. 为KG添加逆向三元组，即对于KG中任意三元组(h, r, t)，添加逆向三元组 (t, r+n_relations, h)，
        #    并将原三元组和逆向三元组拼接为新的DataFrame，保存在 self.kg_data 中。
        
        # 获取原始关系数量
        n_relations = len(set(kg_data['r']))
        
        # 创建逆向三元组
        kg_data_inv = kg_data.copy()
        kg_data_inv = kg_data_inv.rename(columns={'h': 't', 't': 'h'})
        kg_data_inv['r'] += n_relations  # 逆向关系的ID加上原始关系数，以区分正向和逆向关系
        
        # 合并正向和逆向三元组
        self.kg_data = pd.concat([kg_data, kg_data_inv], ignore_index=True)
        
        # 2. 计算关系数，实体数和三元组的数量
        self.n_relations = len(set(self.kg_data['r']))
        self.n_entities = len(set(self.kg_data['h']).union(set(self.kg_data['t'])))
        self.n_kg_data = len(self.kg_data)
        
        # 3. 根据 self.kg_data 构建字典 self.kg_dict ，其中key为h, value为tuple(t, r)，
        #    和字典 self.relation_dict，其中key为r, value为tuple(h, t)。
        self.kg_dict = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)
        
        for idx, row in self.kg_data.iterrows():
            h, r, t = row['h'], row['r'], row['t']
            self.kg_dict[h].append((t, r))
            self.relation_dict[r].append((h, t))
```

#### 3.4.2  知识图谱嵌入
- 采用两种方式对知识图谱进行嵌入，分别为逐个元素相加和逐个元素相乘。

1. 通过将embedding直接相加得到最终的embedding

   ```python
   # calc_cf_loss()函数:
   item_pos_cf_embed = item_pos_embed + item_pos_kg_embed
   item_neg_cf_embed = item_neg_embed + item_neg_kg_embed
   
   # calc_loss()函数:
   item_cf_embed = item_embed + item_kg_embed
   ```

2. 通过将embedding直接相乘得到最终的embedding

   ```python
   # calc_cf_loss()函数:
   item_pos_cf_embed = item_pos_embed * item_pos_kg_embed
   item_neg_cf_embed = item_neg_embed * item_neg_kg_embed
   
   # calc_loss()函数:
   item_cf_embed = item_embed * item_kg_embed
   ```

### 3.5 【选做】将多任务方式更改为迭代优化方式

要将目前的多任务联合训练方式更改为迭代优化方式，也就是说，交替地优化知识图谱（KG）损失和协同过滤（CF）损失，需要对训练过程进行修改。具体来说，可以在每个训练周期中分开计算和优化 KG 损失和 CF 损失，而不是将它们加在一起同时优化。

将`main_Embedding_based.py`改写，改写后的代码保存在`main_Embedding_based_iter.py`中。

以下是具体的修改步骤：

1. **修改 `train`函数中的训练循环：**

 在当前的` train`函数中，模型是在同一个循环中同时计算 KG 损失和 CF 损失，并将它们相加后进行优化。我们需要将这个过程拆分成两个独立的步骤，先优化 CF 损失，再优化 KG 损失，或者反过来。可以在每个 epoch 中交替进行。

   ```python
   for epoch in range(1, args.n_epoch + 1):
       model.train()

       # 首先优化 CF 损失
       total_cf_loss = 0
       n_cf_batch = data.n_cf_train // data.cf_batch_size + 1
       for iter in range(1, n_cf_batch + 1):
           cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.cf_batch_size)
           cf_batch_user = cf_batch_user.to(device)
           cf_batch_pos_item = cf_batch_pos_item.to(device)
           cf_batch_neg_item = cf_batch_neg_item.to(device)

           optimizer.zero_grad()
           cf_loss = model.calc_cf_loss(cf_batch_user, cf_batch_pos_item, cf_batch_neg_item)
           cf_loss.backward()
           optimizer.step()
           total_cf_loss += cf_loss.item()

       # 然后优化 KG 损失
       total_kg_loss = 0
       n_kg_batch = data.n_kg_train // data.kg_batch_size + 1
       for iter in range(1, n_kg_batch + 1):
           kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(data.kg_dict, data.kg_batch_size, data.n_entities)
           kg_batch_head = kg_batch_head.to(device)
           kg_batch_relation = kg_batch_relation.to(device)
           kg_batch_pos_tail = kg_batch_pos_tail.to(device)
           kg_batch_neg_tail = kg_batch_neg_tail.to(device)

           optimizer.zero_grad()
           kg_loss = model.calc_kg_loss(kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)
           kg_loss.backward()
           optimizer.step()
           total_kg_loss += kg_loss.item()

       # 打印损失
       logging.info('Epoch {:04d} | CF Loss {:.4f} | KG Loss {:.4f}'.format(epoch, total_cf_loss / n_cf_batch, total_kg_loss / n_kg_batch))
   ```

3. **修改 `forward`函数以适应需求：**

修改原来的`is_train`函数为`mode`，可以根据传入的参数决定调用哪个损失函数。

   ```python
   def forward(self, *input, mode):
       if mode == 'cf':
           return self.calc_cf_loss(*input)
       elif mode == 'kg':
           return self.calc_kg_loss(*input)
       else:
           raise ValueError('Mode can only be "cf" or "kg".')
   ```

除此以外，我们还调用了`matplotlib`实现损失函数的可视化

##  4. 实验结果

### 基础推荐方法

直接运行
```python3
python3 main_KG_free.py
```

<!--
2024-12-11 20:51:22,882 - root - INFO - Best CF Evaluation: Epoch 0040 | Precision [0.2975, 0.2579], Recall [0.0669, 0.1130], NDCG [0.3054, 0.2821] -->


模型及结果保存在`baseline/trained_model/Douban/KG_free/dim32_lr0.001_l20.0001`下

| Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| :------: | :----: | :-------: | :-----: |
| 0.0669   | 0.3054 | 0.1130    | 0.2821  |



### 逐个元素相加嵌入
模型及结果保存在`baseline/trained_model/Douban/Embedding_based/dim32_lr0.001_l20.0001_TransE`下

<!-- 2024-12-11 21:05:36,349 - root - INFO - Best CF Evaluation: Epoch 0030 | Precision [0.3136, 0.2651], Recall [0.0704, 0.1151], NDCG [0.3285, 0.2969]

 -->

| Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| :------: | :----: | :-------: | :-----: |
| 0.0704   | 0.3285 | 0.1151    | 0.2969  |

### 逐个元素相乘嵌入

<!-- 2024-12-11 21:14:05,535 - root - INFO - Best CF Evaluation: Epoch 0050 | Precision [0.3011, 0.2604], Recall [0.0698, 0.1159], NDCG [0.3138, 0.2886] -->


| Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| :------: | :----: | :-------: | :-----: |
| 0.0698   | 0.3138 | 0.1159    | 0.2886  |

##  5. 实验结果分析

### 一、基础推荐方法与知识感知推荐的比较

从实验结果中可以看出，**知识感知推荐方法**（即引入知识图谱嵌入的方法）相比于**基础推荐方法**，在各项指标上都有一定程度的提升。

- **基础推荐方法**：

  |    指标    | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
  | :--------: | :------: | :----: | :-------: | :-----: |
  | **数值**   |  0.0669  | 0.3054 |  0.1130   | 0.2821  |

- **逐元素相加嵌入**：

  |    指标    | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
  | :--------: | :------: | :----: | :-------: | :-----: |
  | **数值**   |  0.0704  | 0.3285 |  0.1151   | 0.2969  |
  | **提升**   | +0.0035  |+0.0231 |  +0.0021  | +0.0148 |

- **逐元素相乘嵌入**：

  |    指标    | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
  | :--------: | :------: | :----: | :-------: | :-----: |
  | **数值**   |  0.0698  | 0.3138 |  0.1159   | 0.2886  |
  | **提升**   | +0.0029  |+0.0084 |  +0.0029  | +0.0065 |

可以看出，引入知识图谱嵌入的推荐方法在 **Recall** 和 **NDCG** 指标上均有提升。这表明利用知识图谱提供的额外信息，能够增强物品表示的丰富性，从而提高推荐性能。

### **二、 多任务学习与迭代优化的比较**

为进一步探讨不同优化策略的影响，我们对**多任务学习**与**迭代优化**两种方法进行了对比分析。

#### **指标对比**

| 方法         | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| ------------ | -------- | ------ | --------- | ------- |
| **多任务**   | 0.0704   | 0.3285 | 0.1151    | 0.2969  |
| **迭代优化** | 0.0692   | 0.3136 | 0.1130    | 0.2839  |

**迭代优化的趋势变化**：

| Epoch | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| ----- | -------- | ------ | --------- | ------- |
| 10    | 0.0443   | 0.2144 | 0.0841    | 0.2085  |
| 20    | 0.0606   | 0.2841 | 0.1021    | 0.2593  |
| 30    | 0.0663   | 0.3063 | 0.1131    | 0.2811  |
| 40    | 0.0692   | 0.3136 | 0.1130    | 0.2839  |
| 50    | 0.0650   | 0.3054 | 0.1123    | 0.2836  |
| 100   | 0.0611   | 0.2799 | 0.1046    | 0.2587  |

损失函数图如下

![image](baseline\trained_model\Douban\Embedding_based\dim32_lr0.001_l20.0001_TransE_20241217180203\training_loss.png)

可以看到，KG Loss早早收敛，由于本实验是采用的知识图谱(KG)进行的优化训练，所以尽管在100轮后CF Loss还能继续下降，仍然终止训练。

#### **分析与对比**

1. **性能对比**
   - **多任务学习**：性能较为稳定，最佳 Recall@5 为 **0.0704**，NDCG@5 为 **0.3285**。
   - **迭代优化**：在前期（Epoch 40）达到最佳性能，Recall@5 为 **0.0692**，NDCG@5 为 **0.3136**，但后期存在波动。
2. **训练过程**
   - **多任务学习**：同时优化 CF 损失和 KG 损失，参数更新较为平滑，收敛更稳定。
   - **迭代优化**：交替优化 CF 损失和 KG 损失，训练过程较不稳定，容易在后期出现参数震荡。
3. **结果结论**
   - 多任务学习的表现整体优于迭代优化，尤其在 NDCG 指标上优势明显。
   - 迭代优化的训练前期表现较好，但收敛性不如多任务学习。

### 三、不同知识图谱嵌入方法的比较

对比**逐元素相加**和**逐元素相乘**两种嵌入方法，可以发现：

- **逐元素相加嵌入**在 **Recall@5** 和 **NDCG@5/10** 指标上表现更好。
- **逐元素相乘嵌入**在 **Recall@10** 指标上略有优势。

具体比较：

|    指标    | 相加嵌入 | 相乘嵌入 | 差异（相加-相乘） |
| :--------: | :------: | :------: | :---------------: |
| **Recall@5**  |  0.0704  |  0.0698  |      +0.0006      |
| **NDCG@5**    |  0.3285  |  0.3138  |      +0.0147      |
| **Recall@10** |  0.1151  |  0.1159  |      -0.0008      |
| **NDCG@10**   |  0.2969  |  0.2886  |      +0.0083      |

#### 可能的原因分析

1. **信息融合方式的影响**：
   - **逐元素相加**：
   
     - **优势**：相加操作等效于对不同来源的特征进行线性融合，可能更有利于保留并综合物品嵌入和知识图谱嵌入的原始信息。
     - **影响**：在相加的过程中，嵌入向量的每个维度的信息能够被充分利用，提升模型的表达能力。
   
   - **逐元素相乘**：
   
     - **优势**：相乘操作能够捕获物品嵌入和知识图谱嵌入之间的交互关系。
     - **影响**：相乘可能会导致嵌入向量的值变小，甚至接近于零，可能会导致信息的损失。在模型训练不足或数据不充分的情况下，效果可能不如相加明显。
   
2. **模型复杂度和泛化能力**：

   - **相加嵌入**的模型复杂度相对较低，更容易训练和收敛，泛化能力可能更好。
   - **相乘嵌入**引入了非线性的交互，模型复杂度增加，可能需要更多的数据和更长的训练时间来充分发挥效果。

3. **数据集特点**：

   - 数据集中物品和知识图谱实体的特征可能更适合线性组合，即相加方式更能体现其关联性。


### 四、基础推荐方法与知识感知推荐的差异分析

1. **知识感知推荐的优势**：

   - **丰富的语义信息**：知识图谱提供了物品之间的语义关联，弥补了协同过滤方法仅依赖用户行为数据的不足。
   - **改进的物品表示**：通过融合物品嵌入和知识图谱嵌入，物品的表示更加全面，有助于提升推荐的准确性。

2. **性能提升有限的原因**：

   - **知识图谱质量**：如果知识图谱的覆盖度和准确性不高，对模型的提升作用可能有限。
   - **模型简单**：当前模型仅仅进行了简单的嵌入融合，未充分利用知识图谱的结构信息。
   - **数据稀疏性**：在冷启动或长尾物品的情况下，知识图谱的作用可能更明显，而在数据相对丰富时，提升较为有限。

### 五、简要总结

1. **基础推荐与知识感知推荐**：
   引入知识图谱嵌入的推荐方法显著提升了推荐性能，表明知识图谱在丰富物品表示方面的有效性。
2. **多任务与迭代优化对比**：
   - **多任务学习** 表现更稳定，收敛效果更优，适合大多数场景。
   - **迭代优化** 前期表现尚可，但收敛性较差，后期性能出现波动。
3. **知识图谱嵌入方法对比**：
   - **逐元素相加** 的表现优于逐元素相乘，尤其在推荐结果排序质量上更胜一筹。