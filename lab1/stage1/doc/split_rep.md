### 分词部分
#### 1. 导入必要的模块

```python
import csv
import jieba
import pkuseg
import ast
```

- `csv`：用于读取和写入CSV文件。
- `jieba`、`pkuseg`：中文分词工具，支持用户选择。
- `ast`：用于将字符串形式的集合安全地解析为实际的集合对象。

#### 2. 选择分词工具

```python
# 选择分词工具
tool_choice = input("请选择分词工具（输入 'jieba' 或 'pkuseg'）：")
if tool_choice.lower() == 'jieba':
    use_jieba = True
    print("使用 Jieba 分词")
elif tool_choice.lower() == 'pkuseg':
    use_jieba = False
    print("使用 PKUSeg 分词")
else:
    print("输入有误，默认使用 Jieba 分词")
    use_jieba = True
```

- 通过命令行输入，用户可选择使用 `jieba` 或 `pkuseg` 进行分词。
- 默认情况下，若输入有误，则使用 `jieba` 分词。

#### 3. 加载停用词表

```python
# 加载停用词表
stopwords_file = 'cn_stopwords.txt'
stopwords = set()
with open(stopwords_file, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip()
        if word:
            stopwords.add(word)
```

- 从文件 `cn_stopwords.txt` 中读取停用词。
- 使用 `set` 存储停用词，便于快速查找。

**停用词过滤的实现**

- 在分词后的结果中，遍历每个词，检查其是否在 `stopwords` 集合中。
- 如果在停用词列表中，则过滤掉，不纳入后续处理。

#### 4. 加载同义词词典

```python
# 加载同义词词典
synonyms_file = 'syno_from_baidu_hanyu.txt'
synonym_dict = {}
with open(synonyms_file, 'r', encoding='utf-8') as f:
    for line in f:
        words = line.strip().split()
        if words:
            representative = words[0]
            for word in words:
                synonym_dict[word] = representative
```

- 读取 `syno_from_baidu_hanyu.txt` 文件，每行包含一组同义词，使用空格分隔。
- 将同义词组中的所有词映射到代表词（该组的第一个词），构建 `synonym_dict` 字典。

**同义词替换的实现**

- 在词语经过停用词过滤后，检查是否在 `synonym_dict` 中。
- 如果存在同义词映射，则将词替换为对应的代表词。

#### 5. 定义文件列表并初始化分词器

```python
# 定义文件列表
file_list = [
    {'input': 'selected_book_top_1200_data_tag.csv', 'output': 'book_output.csv'},
    {'input': 'selected_movie_top_1200_data_tag.csv', 'output': 'movie_output.csv'}
]

# 初始化 pkuseg 分词器
if not use_jieba:
    seg = pkuseg.pkuseg()
```

- 指定需要处理的输入文件和对应的输出文件。
- 如果用户选择了 `pkuseg`，则初始化相应的分词器实例。

#### 6. 处理文件并进行分词、停用词过滤和同义词替换

```python
for file_pair in file_list:
    input_file = file_pair['input']
    output_file = file_pair['output']
    print(f"正在处理文件：{input_file}")

    with open(input_file, 'r', encoding='utf-8') as csvfile_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as csvfile_out:
        reader = csv.reader(csvfile_in)
        writer = csv.writer(csvfile_out)

        # 读取并写入表头
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            item_id = row[0]
            tags_str = row[1]

            # 解析 Tags 字段
            try:
                tags_set = ast.literal_eval(tags_str)
            except:
                print(f"解析文件 {input_file} 中的 ID {item_id} 的标签时出错")
                continue

            # 分词、停用词过滤、同义词替换
            new_tags = set()
            for tag in tags_set:
                # 分词
                if use_jieba:
                    words = jieba.lcut(tag)
                else:
                    words = seg.cut(tag)

                for word in words:
                    word = word.strip()
                    # 停用词过滤
                    if word and word not in stopwords:
                        # 同义词替换
                        representative = synonym_dict.get(word, word)
                        new_tags.add(representative)

            # 重建 Tags 字符串
            new_tags_str = "{" + ", ".join(f"'{tag}'" for tag in new_tags) + "}"
            writer.writerow([item_id, new_tags_str])

    print(f"文件 {input_file} 处理完成，结果已保存到 {output_file}")

print("所有文件处理完成！")
```

- **读取文件并初始化读写器**：打开输入和输出文件，创建 CSV 读写对象。
- **处理每一行数据**：
  - 解析标签字符串 `tags_str`，将其转换为集合 `tags_set`。
  - **分词**：根据用户选择的分词工具，对每个标签 `tag` 进行分词。
  - **停用词过滤**：对于分词结果 `words`，逐一检查是否在 `stopwords` 中。仅保留不在停用词列表中的词。
  - **同义词替换**：检查 `synonym_dict`，将词替换为对应的代表词。
  - **收集新标签**：将处理后的词语加入 `new_tags` 集合，自动去重。
- **写入处理结果**：将新的标签集合 `new_tags` 重新格式化为字符串形式，写入输出文件。

#### 7.解释说明

**实验结果：**
**我们在代码中插入了一些统计时间和分词结果的代码(代码文件split_test.py)，得到了一些统计数据，以下分析都基于这部分实验结果**

```shell
root@LAPTOP-Q9CFOCTC ~/w/m/n/W/l/stage1 (main) [1]# python3 split.py                                                       (base) 
请选择分词工具（输入 'jieba' 或 'pkuseg'）：jieba
使用 Jieba 分词
正在处理文件：../data/selected_book_top_1200_data_tag.csv
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.607 seconds.
Prefix dict has been built successfully.
文件 ../data/selected_book_top_1200_data_tag.csv 处理完成，结果已保存到 data/book_output.csv
分词过程运行时间：3.12 秒
总词数：319327
去除的停用词和非法词语数量：71344
替换的同义词数量：34034

正在处理文件：../data/selected_movie_top_1200_data_tag.csv
文件 ../data/selected_movie_top_1200_data_tag.csv 处理完成，结果已保存到 data/movie_output.csv
分词过程运行时间：10.46 秒
总词数：1318761
去除的停用词和非法词语数量：322347
替换的同义词数量：135582

所有文件处理完成！

```


```shell
root@LAPTOP-Q9CFOCTC ~/w/m/n/W/l/stage1 (main)# python3 split.py                                                           (base) 
请选择分词工具（输入 'jieba' 或 'pkuseg'）：pkuseg
使用 PKUSeg 分词
正在处理文件：../data/selected_book_top_1200_data_tag.csv
文件 ../data/selected_book_top_1200_data_tag.csv 处理完成，结果已保存到 data/book_output.csv
分词过程运行时间：11.97 秒
总词数：319431
去除的停用词和非法词语数量：74024
替换的同义词数量：37748

正在处理文件：../data/selected_movie_top_1200_data_tag.csv
文件 ../data/selected_movie_top_1200_data_tag.csv 处理完成，结果已保存到 data/movie_output.csv
分词过程运行时间：54.53 秒
总词数：1229071
去除的停用词和非法词语数量：320200
替换的同义词数量：143182

所有文件处理完成！

``` 


1. **分词工具选择**：在处理中文文本时，分词是一个重要的预处理步骤。本程序提供了两种中文分词工具的选择：`jieba` 和 `pkuseg`。用户可以根据实际需求选择合适的分词工具。

最终方案是：使用 `jieba` 分词工具，选择精确模式进行分词，开启 HMM 模型。 原因如下：

- 经过测试，`jieba` 分词工具在速度和效果上表现优异，适用于大多数中文文本的分词需求。所以推荐使用 `jieba` 分词工具。

- jieba支持三种分词模式：精确模式、全模式和搜索引擎模式。其中，精确模式试图将句子最精确地切开，适合文本分析；全模式把句子中所有可以成词的词语都扫描出来，速度非常快，但是不能解决歧义，并产生很多相同词项（这些词项占用大量存储空间并有很多在同义词去除中被删去，故不采用）；搜索引擎模式在精确模式的基础上，对长词再次切分，提高召回率，但速度较慢。

- jieba支持**HMM模型**，默认开启。


2. **停用词过滤**：

使用已有的中文停用词表 `cn_stopwords.txt`，存储于 stage1/data 目录下，对分词结果进行停用词过滤。


3.**同义词替换** ：

使用已有的中文同义词表 `syno_from_baidu_hanyu.txt`，存储于 stage1/data 目录下，对分词结果进行同义词替换。