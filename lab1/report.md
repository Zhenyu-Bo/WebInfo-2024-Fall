# Web信息处理与应用 Lab1

PB22081571薄震宇	PB22111613王翔辉	PB22020514郭东昊

[TOC]

## 实验要求

**stage1**：根据给定的豆瓣 Movie&Book 的 tag 信息，实现电影和书籍的检索。对于给定的查询，通过分词、构建索引、索引优化等流程，实现面向给定查询条件的精确索引。

**stage2**：基于豆瓣 Movie&Book 的 tag 信息、以及提供的豆瓣电影与书籍的评分记录以及用户间的社交关系，判断用户的偏好，对用户交互过的 item（电影、书籍）进行（基于得分预测的）排序。

## 文件组织

目录结构及文件含义如下

```
├── data/			// 本次实验提供的数据文件
├── figs/			// 图片文件
├── stage1/			// 第一阶段
│   ├── data/		// 第一阶段生成的数据文件
│   ├── doc			// 第一阶段的相关文档
│   ├── notebook
│   ├── bool_search.py	// 实现布尔查询
│   ├── bool_search_on_one_string.py  	// 实现将词项看作单一字符串压缩对应的布尔查询
│   ├── bool_search_with_skip_list.py	// 实现使用跳表指针的布尔查询
│   ├── dictionary_as_a_string.py		// 实现将词项看作单一字符串的压缩及查询一个词项的倒排列表
│   ├── extract_id.py					// 提取所有文档ID
│   ├── gen_inverted_index_table.py		// 生成倒排索引表
│   ├── split.py						// 实现分词等预处理操作
│   ├── var_len_bool_search.py			// 实现间距代替ID+可变长度编码压缩对应的布尔查询
│   ├── var_len_compress.py				// 实现间距代替ID+可变长度编码压缩方式的索引压缩
│   └── var_len_search_one_word.py		// 实现间距代替ID+可变长度编码压缩的查询一个词项的倒排列表
├── stage2/			// 第二阶段
│   ├── data/
│   ├── doc/
│   ├── graph_rec_model.py
│   ├── graphrec.ipynb
│   ├── test.py
│   ├── text_embedding.ipynb
│   └── utils.py
└── report.md		// 实验报告
```



## 实验过程

### stage1

#### 数据预处理

##### 1. 导入必要的模块

```python
import csv
import jieba
import pkuseg
import ast
```

- `csv`：用于读取和写入CSV文件。
- `jieba`、`pkuseg`：中文分词工具，支持用户选择。
- `ast`：用于将字符串形式的集合安全地解析为实际的集合对象。

##### 2. 选择分词工具

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

##### 3. 加载停用词表

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

##### 4. 加载同义词词典

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

##### 5. 定义文件列表并初始化分词器

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

##### 6. 处理文件并进行分词、停用词过滤和同义词替换

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

#### 建立倒排索引表与优化

这一部分的实现在文件`gen_inverted_index_table.py`中。

##### 建立

总体可以分为两大步骤：

第一步：读取预处理后的文件，提取出所有词项的集合`all_words`，建立文档 ID - 词项字典`documents`，键为文档ID，值为相应的文档包含的所有词项。这一步骤代码如下：

第二步：首先初始化一个空列表`inverted_index_table`作为倒排表。然后对于`all_words`中的每一个词项`word`，获取包含`word`的文档ID列表。可以通过`doc_ids = [doc_id for doc_id, words in documents.items() if word in words]`获得。为了方便后续的操作，再对`doc_id`进行排序得到`doc_ids_sorted`。然后将`word`和`doc_ids_sorted`写入`inverted_index_table`。

最终将`inverted_index_table`转换为CSV文件存储。

##### 优化

我们通过设置跳表指针对倒排索引表进行优化，这里也体现了上面对`doc_ids`进行排序的必要性。

简单起见，我们取跳表指针的间隔为$\sqrt{L}$（设倒排表长度为$L$）。

跳表指针的设置在遍历词项时进行。

对于每一个词项，设其文档`id`列表的长度为`len`。首先初始化其跳表指针为一个空列表`skip_table`。然后遍历排序后的文档 ID 列表doc_ids_sorted中的每个索引 `i`：

如果当前索引`i`是跳表间隔`l`的倍数，则设置跳表指针。如果`i`小于`len - l`，则跳表指针指向`i+l`位置的文档ID，否则跳表指针执行文档列表的末尾。

如果文档数量小于等于跳表间隔也即文档数量为1，这时显然没有必要也无法设置跳表指针，故其跳表信息设为空。

最后再将`skip_table`与`word`，`doc_ids_sorted`一并写入`inverted_index_table`中。

以上两个步骤对应的函数如下：

```python
def read_words_from_csv(file_path):
    """
    读取 CSV 文件，提取所有单词和文档内容。

    参数：
    - file_path: CSV 文件路径。

    返回：
    - all_words: 所有词项的集合。
    - documents: 字典，键为文档 ID，值为该文档包含的单词集合。
    """
    data = pd.read_csv(file_path, dtype={'id': int, 'words': str})
    all_words = set()
    documents = {}
    for idx in range(len(data)):
        words = ast.literal_eval(data.at[idx, 'words'])
        doc_id = data.at[idx, 'id']
        documents[doc_id] = words
        all_words.update(words)
    return all_words, documents


def generate_inverted_index_table(all_words, documents, output_file):
    """
    生成倒排索引表并保存为 CSV 文件。

    参数：
    - all_words: 集合，包含所有词项。
    - documents: 字典，键为文档 ID，值为该文档的标签集合。
    - output_file: 输出的 CSV 文件路径。
    """
    inverted_index_table = []
    for word in all_words:
        doc_ids = [doc_id for doc_id, words in documents.items() if word in words]
        doc_ids_sorted = sorted(doc_ids)
        num_docs = len(doc_ids_sorted)
        l = int(sqrt(num_docs))  # 设置跳表的间隔
        skip_table = []
        if num_docs > l:
            for i in range(num_docs):
                if i % l == 0:
                    if i < num_docs - l:
                        skip_info = {'index': i + l, 'value': doc_ids_sorted[i + l]}
                    else:
                        # 最后一个跳表指针指向末尾
                        skip_info = {'index': num_docs - 1, 'value': doc_ids_sorted[num_docs - 1]}
                    skip_table.append(skip_info)
        else:
            skip_info = {'index': None, 'value': None}
            skip_table.append(skip_info)
        inverted_index_table.append({'word': word, 'id_list': doc_ids_sorted, 'skip_table': skip_table})
    pd.DataFrame(inverted_index_table).to_csv(output_file, index=False)
```

#### 布尔查询

这一部分的实现在文件`bool_search.py`中。

##### 输入文件及格式

程序需要以下输入文件：

1. **倒排索引表**

   - **书籍倒排索引表**：`book_inverted_index_table.csv`
   - **电影倒排索引表**：`movie_inverted_index_table.csv`

   **文件格式**：

   - CSV 文件，包含以下字段：
     - `word`：单词或词语。
     - `id_list`：包含该单词的文档 ID 列表，存储为字符串形式的列表，例如：`"[1, 2, 3]"`。
     - `skip_table`：跳表信息，存储为字符串形式的列表，包含跳跃节点的索引和对应的值，例如：`"[{'index': 2, 'value': 3}, {'index': 5, 'value': 6}]"`。

2. **全 ID 表**

   - **书籍 ID 表**：`Book_id.txt`
   - **电影 ID 表**：`Movie_id.txt`

   **文件格式**：

   - 文本文件，每行一个文档 ID，表示所有文档的集合。

3. **词表**

   - **书籍词表**：`book_words.csv`
   - **电影词表**：`movie_words.csv`

   **文件格式**：

   - CSV 文件，包含以下字段：
     - `id`：文档 ID。
     - `words`：文档的标签或关键字列表，存储为字符串形式的列表，例如：`"['爱情', '小说', '文学']"`。

##### 处理布尔表达式

程序支持对文档进行布尔查询，支持的操作符包括：

- `AND`：逻辑与，表示取集合的交集。
- `OR`：逻辑或，表示取集合的并集。
- `NOT`：逻辑非，表示对集合取补集。
- 括号 `(` 和 `)`：用于改变运算的优先级。

**处理步骤如下**：

1. **分词**

   使用正则表达式将输入的布尔表达式拆分为标记（Token），包括操作数（单词）、操作符和括号。

   例如，输入表达式：

   ```
   (爱情 AND 小说) OR (NOT 科幻)
   ```

   分词结果为：

   ```python
   ['(', '爱情', 'AND', '小说', ')', 'OR', '(', 'NOT', '科幻', ')']
   ```

2. **中缀表达式转后缀表达式**

   为了方便计算机处理，将中缀表达式转换为后缀表达式（逆波兰表达式）。

   - 定义操作符的优先级：`NOT` > `AND` > `OR`。
   - 使用栈（Stack）来保存操作符，按照逆波兰表达式转换算法处理。

   转换后的后缀表达式为：

   ```python
   ['爱情', '小说', 'AND', '科幻', 'NOT', 'OR']
   ```

3. **后缀表达式求值**

   遍历后缀表达式，使用栈来计算结果集。

   - **遇到操作数**（单词）：
     - 从倒排索引表中获取对应的 `id_list` 和 `skip_table`，将二者作为一个元组压入栈中。
   - **遇到操作符**：
     - **`AND` 操作符**：
       - 从栈中弹出右操作数和左操作数（各包含 `id_list` 和 `skip_table`）。
       - 使用跳表优化的交集操作 `intersect_with_skips`，计算结果并压入栈中。
     - **`OR` 操作符**：
       - 从栈中弹出右操作数和左操作数。
       - 对两个 `id_list` 取并集，结果压入栈中。
     - **`NOT` 操作符**：
       - 从栈中弹出一个操作数。
       - 计算全 ID 集合与该操作数的差集，结果压入栈中。
   - 最终，栈顶元素即为查询结果的文档 ID 列表。

4. **结果展示**

   根据查询结果的文档 ID，从词表中查找对应的标签或关键字，打印输出。

##### 利用跳表优化查询

在处理 `AND` 操作时，特别是当文档 ID 列表较长时，利用跳表可以加速查询过程，减少比较次数。

###### 跳表（Skip List）原理

跳表是一种在有序链表上增加多级索引的结构，允许在查找时跳过部分元素，加速定位。

在倒排索引中，跳表由一系列跳跃节点组成，每个跳跃节点包含：

- `index`：在 `id_list` 中的索引位置。
- `value`：对应的文档 ID。

###### 交集操作的优化

**传统交集操作**：

- 顺序遍历两个有序的文档 ID 列表，逐一比较，找到共同的文档 ID。
- 当列表很长时，比较次数多，效率较低。

**利用跳表的交集操作**：

- 在比较 `p1[i]` 和 `p2[j]` 时，如果 `p1[i] < p2[j]`，尝试利用 `p1` 的跳表跳过一些元素。
  - 查找 `p1` 中下一个跳跃节点，如果其值 `p1[next_skip_idx]` 小于等于 `p2[j]`，则将 `i` 直接跳到 `next_skip_idx`。
  - 否则，`i` 增加 1。
- 同理，如果 `p2[j] < p1[i]`，利用 `p2` 的跳表尝试跳过一些元素。
- 当 `p1[i] == p2[j]` 时，将该文档 ID 加入结果列表，`i` 和 `j` 均增加 1。

**算法步骤**：

```python
def intersect_with_skips(p1, p2, skip_p1, skip_p2):
    answer = []
    i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            next_skip_idx = get_next_skip_idx(skip_p1, i)
            if next_skip_idx != -1 and p1[next_skip_idx] <= p2[j]:
                i = next_skip_idx
            else:
                i += 1
        else:
            next_skip_idx = get_next_skip_idx(skip_p2, j)
            if next_skip_idx != -1 and p2[next_skip_idx] <= p1[i]:
                j = next_skip_idx
            else:
                j += 1
    return answer
```

**获取下一个跳跃索引**：

```python
def get_next_skip_idx(skip_table, current_idx):
    for skip in skip_table:
        if skip['index'] > current_idx:
            return skip['index']
    return -1
```

##### 注意事项

- **跳表的构建**：
  - 跳表需要在构建倒排索引时生成，并存储在 `skip_table` 字段中。
  - 跳表的选择应根据列表长度和优化需求进行调整。

- **数据格式**：
  - 在读取倒排索引表时，确保正确解析 `id_list` 和 `skip_table`，并按照需要的数据结构存储。

- **性能提升**：
  - 利用跳表可以在处理长列表时显著减少比较次数，提高查询效率。
  - 对于较短的列表，跳表的效果可能不明显，但不会降低性能。

通过以上方法，程序能够高效地处理布尔查询，并利用跳表优化查询性能。

#### 索引压缩与查询

我们使用了**将词典视作单一字符串**和**间距代替文档ID+可变长度编码**两种压缩策略，并且为相应的压缩策略实现了在压缩后的倒排索引表上查询**一个词项的倒排表**的接口和进行布尔查询的算法。

##### 将词典视作单一字符串

**压缩：**

这一部分对应的文件为`dictionary_as_a_string.py`。

步骤如下：

1. **读取倒排索引表**：从输入的 CSV 文件中读取词项和对应的文档 ID 列表。
2. **排序词项**：将词项按照字典序排序，使得后续可以使用二分查找。
3. **创建词项字符串**：将所有词项连接成一个单一的字符串，并记录每个词项的起始位置。
4. **保存词项字符串**：将词项字符串保存到文件中。
5. **保存倒排表**：将倒排列表保存到文件中，并记录每个倒排列表的起始偏移量。
6. **创建词项表**：创建词项表并保存到文件中，记录每个词项的文档频率、倒排指针、词项指针和词项长度。

最终得到了词项字符串文件、词项表文件和倒排表文件。

与老师课上讲的不同的是，这里我们将词项的长度保存到了词项表中而不是字符串中每一个词项前，这样在查词项表时就可以根据词项指针和词项长度来获得词项，而不用先获得词项长度，再获得词项。也就是说，我们的词项表的表头是`doc_frequency,posting_ptr,term_ptr,term_length`，分别表示文档频率，倒排表指针，词项指针和词项长度。

这里的指针在实现时用偏移量来表示，也就是说通过`term_ptr`表示相应的词项在字符串文件中的偏移量，`posting_ptr`表示倒排列表在倒排表文件中的偏移量。

**查询一个词项**

这一部分对应的代码也在`dictionary_as_a_string.py`里。

这里的查询并不是根据用户的输入进行布尔查询，而是查询给定词项对应的倒排列表，可以在布尔查询中使用这一个接口用来查询。

基本步骤如下：

1. **加载词项表和词项字符串**：从文件中加载词项表和词项字符串。
2. **在词项字符串中查找词项**：使用二分查找定位词项，这也体现了压缩时对词项进行排序的必要性。
3. **获取倒排列表的起始偏移量**：从词项表中获取倒排列表的起始偏移量。
4. **读取倒排列表**：从倒排表文件中读取特定词项的倒排列表。
5. **返回查询结果**：返回查询到的文档 ID 列表，如果没有则返回空列表。

总的来说，首先通过二分查找定位词项，然后通过词项表文件获得词项对应的倒排表的偏移量，然后可以在倒排表文件中得到倒排表。

**在压缩后的倒排索引表上进行布尔查询：**

这一部分对应的文件为`bool_search_on_one_string.py`

这里与前面实现的布尔查询基本相同，需要修改`bool_search.py/evaluate_postfix`中的下面这段代码：

```python
if token not in operators:
    if token in inverted_index:
        stack.append(set(inverted_index[token]))
    else:
        stack.append(set())
```

因为这里是直接从倒排索引表读取词项的倒排列表，修改为使用前面编写的查询一个词项的倒排列表的接口即可：

```python
if token not in operators:
    doc_ids = query_posting_list(token, term_string, term_table, posting_list_file_path)
    stack.append(set(doc_ids))
```

**压缩前后存储空间与检索效率的比较：**

**存储空间：**

压缩前后的倒排索引表的相关文件大小如下：

![compress1](C:\Users\bzy\Downloads\figs\compress1_book.png)

![compress1](C:\Users\bzy\Downloads\figs\compress1_movie.png)

第一张图中的`book_inverted_index_table.csv`是书籍数据的原始倒排索引表，`book_posting_list`，`book_term_string`，`book_term_table`是对书籍倒排索引表进行压缩后得到的三个文件。

也就是倒排表索引从$3.20MB = 3276.8 KB$被压缩到了$1.33MB+124KB+340KB = 1825.92KB$

压缩了约$\frac{3276.8-1825.92}{3276.8} \times 100\% = 44.277\%$

类似的，第二张图的几个文件是电影数据压缩前后的倒排索引表相关文件。可以看到倒排索引表从$10.1MB = 10342.4KB$被压缩到了$4.70MB+365KB+0.98MB = 6181.32KB$

压缩了约$\frac{10342.4 - 6181.32}{10342.4} \times 100\% = 40.233\%$

两个倒排索引表都压缩了$40\%$以上，压缩效果明显。

**检索效率：**

从理论上分析，检索效率应下降，因为压缩后需要在词项字符串上进行二分查找，然后才能获得倒排列表指针进而获得倒排列表，而压缩前可以直接将倒排索引表转化为字典，然后直接根据词项获得其倒排列表。

但是在实际执行时，压缩前后执行一次查找的时间都是`0.000000s`。这可能是因为数据较少，倒排索引表较小，不论是否压缩，都可以在极短的时间内完成查找，故难以比较。

##### 间距代替文档ID+可变长度编码

**压缩：**

这一部分对应的文件为`var_len_compress.py`

压缩主要分为两个步骤：

第一步，计算一个词项的倒排列表中的文档ID间距，使用间距代替文档ID

第二步，对计算出来的间距进行可变长度编码

计算间距的代码较为简单，下面说明一下编码的代码：

```python
def variable_byte_encode(gaps: list[int]) -> bytes:
    """
    使用可变字节编码（Variable Byte Encoding）对文档间距进行压缩。

    参数：
    - gaps: 文档间距列表。

    返回：
    - 压缩后的字节序列。
    """
    encoded_bytes = []
    for gap in gaps:
        byte_chunks = []
        while gap >= 128:
            byte_chunks.append(gap % 128)
            gap = gap // 128
        byte_chunks.append(gap + 128)  # 设置最后一个字节的最高位为1
        # 由于要按大端顺序存储，反转字节顺序
        byte_chunks = byte_chunks[::-1]
        encoded_bytes.extend(byte_chunks)
    return bytes(encoded_bytes)
```

参数为文档间距列表，对于列表中的每一个元素`gap`，对其进行循环处理：

* 当`gap`大于等于 128 时，将`gap`对 128 取模的结果添加到`byte_chunks`列表中，并将`gap`整除 128。
* 当`gap`小于 128 时，将`gap`加上 128（设置最高位为1），并将结果添加到`byte_chunks`列表中。

循环结束后，由于要按大端顺序存储，所以要反转`byte_chunks`列表。然后再将`byte_chunks`列表中的字节块添加到`encode_bytes`列表中。

压缩后得到两个文件：

1. **压缩后的倒排表二进制文件**：包含压缩后的倒排索引数据。
2. **词汇表文件**：包含词项及其在倒排表中的字节偏移量和长度。

其中词汇表文件的格式为`word, offset, length`，`offset`和`length`分别表示`word`对应的倒排列表在压缩后的倒排表中偏移量和长度，后面可以使用`offset`和`length`获取`word`对应的压缩后的倒排列表再进行解压缩。

**查找一个词项的倒排列表：**

这一步对应的文件为`var_len_search_one_word.py`

上面已经提到，对于给定的词项，可以使用`offset`和`length`获取`word`对应的压缩后的倒排列表再进行解压缩，解压缩后的是间距列表，还需要根据间距列表构建出原始的倒排列表。

根据间距列表构建出原始的倒排列表较为容易，这里主要说明一下解压缩间距列表的代码：

```python
def variable_byte_decode(byte_data: bytes) -> list[int]:
    """
    使用可变字节解码（Variable Byte Decoding）将字节序列解码为整数列表。

    参数：
    - byte_data: 压缩的字节序列。

    返回：
    - 解码后的整数列表。
    """
    numbers = []
    current_number = 0
    for byte in byte_data:
        if byte >= 128:
            # 当最高位为1时，这是一个数字的第一个字节
            # 先将之前的数字加入列表
            if current_number != 0:
                numbers.append(current_number)
                current_number = 0
            # 去掉最高位的1，得到数字的高位
            current_number = (current_number << 7) | (byte - 128)
        else:
            # 当最高位为0时，这是一个数字的中间字节
            current_number = (current_number << 7) | byte
    # 将最后一个数字加入列表
    numbers.append(current_number)
    return numbers
```

输入参数是压缩后的字节序列。对于其中的每一个字节，根据字节的最高位判断是数字的第一个字节还是中间字节，构建当前数字。当字节最高位为1时，说明当前字节是下一个数字的第一个字节，需要将当前数字`current_number`添加到`numbers`中然后再赋值为0以继续处理。

在循环结束后，还需要将最后一个数字加入列表即将`current_number`添加到`numbers`中。

得到的`numbers`即为文档ID间距列表。

**在压缩后的倒排索引表上布尔查询：**

这一部分对应的文件为`var_len_bool_search.py`

这里与前面实现的布尔查询基本相同，需要修改`bool_search.py/evaluate_postfix`中的下面这段代码：

```python
if token not in operators:
    if token in inverted_index:
        stack.append(set(inverted_index[token]))
    else:
        stack.append(set())
```

因为这里是直接从倒排索引表读取词项的倒排列表，修改为使用前面编写的查询一个词项的倒排列表的接口即可：

```python
if token not in operators:
    inverted_index = get_inverted_index_list(token, compressed_file_path, vocabulary_file_path)
    stack.append(set(inverted_index))
```

**压缩前后的存储空间与检索效率比较：**

**存储空间：**

压缩前后的倒排索引表的相关文件大小如下：

![compress2](C:\Users\bzy\Downloads\figs\compress2_book.png)

![compress2](C:\Users\bzy\Downloads\figs\compress2_movie.png)

第一张图中的`book_inverted_index_table.csv`是书籍数据的原始倒排索引表，`book_inverted_index_compressed.bin`和`book_vocabulary.csv`是对书籍倒排索引表进行压缩后得到的两个个文件。

也就是说倒排表索引从$3.20MB = 3276.8 KB$被压缩到了$409KB + 327 KB = 736KB$

压缩了约$\frac{3276.8-736}{3276.8} \times 100\% = 77.539\%$

类似的，第二张图的几个文件是电影数据压缩前后的倒排索引表相关文件。可以看到倒排索引表从$10.1MB = 10342.4KB$被压缩到了$1.20MB + 954KB = 2182.8KB$

压缩了约$\frac{10342.4 - 2182.8}{10342.4} \times 100\% = 78.895\%$

两个倒排索引表都压缩了将近$80\%$，压缩效果十分明显，比上面将词典视作单一字符串的效果更好。

**检索效率：**

从理论上分析，检索效率应下降，因为压缩后需要对获得的字节序列进行解码得到间距列表，再由间距列表重新构建出倒排列表，而压缩前可以直接将倒排索引表转化为字典，然后直接根据词项获得其倒排列表。

但是类似于前面将词典视作单一字符串的压缩方式，在实际执行时，压缩前后执行一次查找的时间都是`0.000000s`。这可能是因为数据较少，倒排列表较短，解码和重构建可以在极短时间内完成，所以在检索时间上只有极小的变化。

### stage2







