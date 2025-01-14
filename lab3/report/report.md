# Web 信息处理与应用 Lab3

PB22081571 薄震宇	PB22111613 王翔辉	PB22020514 郭东昊

[TOC]



## 1. 实验背景

随着人工智能技术的迅猛发展，检索增强生成（Retrieval-Augmented Generation, RAG）系统在知识密集型任务中展现出强大的能力。RAG 系统通过结合信息检索技术与大型语言模型（LLM），能够从外部知识库中提取相关信息，并将其作为上下文输入生成模型，以提供更准确和上下文相关的答案。本实验旨在利用 LangChain 框架，基于公开的法律知识数据库，开发一个简单的 RAG 问答应用示例，以比较大模型的生成式检索与普通检索的区别，并评估引入 RAG 后大模型在专业搜索上的表现。

## 2. 实验介绍

### 2.1 RAG 介绍

RAG 模型由 Facebook AI Research（FAIR）团队于 2020 年首次提出，主要分为三个阶段：

1. **索引（Indexing）**：将外部数据源中的内容转换为向量表示，并存储在向量数据库中。
2. **检索（Retrieval）**：使用用户的查询向量在向量数据库中搜索最相关的信息。
3. **生成（Generation）**：将检索到的信息与用户的查询结合，通过语言模型生成最终的回答。

### 2.2 基于 LangChain 实现 RAG 系统

**LangChain** 是一个用于开发由大语言模型（LLM）支持的应用程序的框架，提供了各种工具和接口，帮助开发者集成和管理语言模型的功能。

本实验通过以下步骤实现 RAG 系统：

1. **数据准备**：加载和清洗法律条文及问答数据，进行文本分割和向量化，并存储在向量数据库中。
2. **数据检索**：根据用户查询，从向量数据库中检索相关文档。
3. **LLM 生成**：将检索到的文档与用户查询结合，生成法律问答。

## 3. 实验内容

### 3.1 数据集说明

我们手动将数据集分为了两个文件，便于处理。

- **law_data.csv**：包含中华人民共和国法律手册最核心的约600条法律条文。
- **qa_data.csv**：包含百度知道约2400条法律问答数据。

### 3.2 任务说明

1. **数据准备阶段**：
    - 数据提取：使用 `CSVLoader` 加载 CSV 文件，处理包含英文逗号和跨行双引号的情况。
    - 数据清洗：替换英文逗号为中文逗号，移除空值和重复项。
    - 文本分割：法律条文每行对应一个法条，问答数据每组问答对应一个整体。
    - 向量化：使用 HuggingFace 的嵌入模型将文本转换为向量。
    - 数据入库：将向量存储到 FAISS 向量数据库中。

2. **数据检索阶段**：
    - TODO
  
3. **LLM 生成阶段**：
    - TODO

## 4. 实验过程

### 4.1 数据准备阶段

**数据分析**：

- law_data.csv
  - 法律条文的文字表述都很专业准确，每行对应一个法条。
  - 有的一个法条就被一对英文双引号包裹，有的多个法条被一对跨行的英文双引号包裹
- qa_data.csv
  - 一组问答用一对跨行的英文双引号"包裹。一组问答一般占两行，有时占三行。

针对以上特征，文本分割和向量化的思路如下： 

- 法律条文每行对应一个向量
- 问答部分每一组问答对应一个向量
- 参数quotechar='"' 指定了双引号作为引用字符，这样 Pandas 在读取 CSV 文件时会将双引号内的内容视为一个整体，即使内容中包含换行符。这样可以确保数据在读取时不会被错误地分割，可以用这个来识别一组问答。 需要注意的是，如果用此方法读取法律条文，法律条文需要每行对应一个向量，故引号包裹的多行法条被读取后，还需要以换行符为分割。

#### 4.1.1 函数 `data_pre`

##### 功能
该函数负责数据提取、清洗、文本分割、向量化以及将文档存储到 FAISS 向量数据库中。

##### 输入参数
- `law_csv_path` (str): `law_data.csv` 文件的路径。
- `qa_csv_path` (str): `qa_data.csv` 文件的路径。
- `faiss_index_path` (str): FAISS 索引文件的保存路径。

##### 输出
- 返回分割并打标签后的文档列表，用于在 `main` 函数中展示示例文档。
- 将 FAISS 索引文件存储在指定的目录下。

##### 具体步骤
1. **加载CSV文件**：
   
    - 使用 `CSVLoader` 分别加载 `law_data.csv` 和 `qa_data.csv`，并指定双引号 `"` 作为引用字符，确保跨行的问答组被正确读取为一个整体。
    - 替换英文逗号 `,` 为中文逗号 `，`，统一标点符号，避免分割时出错。
    
    ```python
    law_loader = CSVLoader(
        file_path=law_csv_path, 
        encoding="utf-8", 
        source_column="data",
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'escapechar': '\\'
        }
    )
    
    law_docs = law_loader.load()
    ```
    
3. **文本分割**：
    - 使用 `CharacterTextSplitter` 按换行符 `\n` 分割法律条文，每行对应一个法条。
    - 对问答数据不进行进一步分割，每组问答作为一个整体。
    
    ```python
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    
    law_splits = text_splitter.split_documents(law_docs)
    ```

4. **添加标签**：
    - 为法律条文和问答数据分别添加标签 `law` 和 `qa`，便于后续区分和检索。
    
    ```python
    for doc in law_splits:
        doc.metadata["label"] = "law"
    for doc in qa_docs:
        doc.metadata["label"] = "qa"
        
    documents = law_splits + qa_docs
    ```

5. **向量化和数据入库**：
   
    - 检查 FAISS 索引目录是否存在，若不存在则创建并进行向量化和存储。
      - 若存在则不进行向量化和存储，**因为实在太费时间了，一次要十几分钟**。仅作为被main函数调用输出示例。
    
    - 使用 HuggingFace 的 `BAAI/bge-base-en-v1.5` 嵌入模型将文档转换为向量。
    - 使用 FAISS 向量数据库存储向量，并保存索引文件。
    
    ```python
    if not os.path.exists(faiss_index_path):
        os.makedirs(faiss_index_path)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(faiss_index_path)
    ```

#### 4.1.2 函数 `main`

##### 功能
该函数调用 `data_pre` 函数进行数据预处理，并展示部分分割后的文档示例。

##### 具体步骤
1. **定义文件路径**：
   
    ```python
    law_csv_path = "law_data.csv"
    qa_csv_path = "qa_data.csv"
    faiss_index_path = "faiss_index"
    ```
    
2. **调用数据预处理函数**：
    ```python
    documents = data_pre(law_csv_path, qa_csv_path, faiss_index_path)
    ```

3. **分组展示文档**：
   
    - 提取前10条法律条文和前10组问答数据进行展示。
    
    ```python
    law_examples = [doc for doc in documents if doc.metadata['label'] == 'law'][:10]
    qa_examples = [doc for doc in documents if doc.metadata['label'] == 'qa'][:10]
    ```
    
4. **打印示例文档**：
   
    - 分别打印法律条文和问答示例，以验证数据分割和标签添加的效果。
    
    ```python
    print("法律条文示例：\n")
    for i, doc in enumerate(law_examples, 1):
        print(f"文档 {i}:")
        print(f"标签: {doc.metadata['label']}")
        print(f"内容: {doc.page_content}\n")
    
    print("问答示例：\n")
    for i, doc in enumerate(qa_examples, 1):
        print(f"文档 {i}:")
        print(f"标签: {doc.metadata['label']}")
        print(f"内容: {doc.page_content}\n")
    ```

### 4.2 数据检索阶段

这一阶段我们需要根据用户的提问，通过高效的检索方法，召回与提问最相关的知识。

首先我们需要加载前面保存的向量化模型与 FAISS 索引，然后使用 similarity_search 方法即可实现相似性检索。

代码如下：

```python
# 加载向量化模型
embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")

# 加载 FAISS 索引
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# 执行相似性检索
similar_docs = vectorstore.similarity_search(query, k=top_k)
```

其中 `k` 参数用于指定相似性搜索中返回的最相似文档的数量。

### 4.3 LLM 生成阶段

#### 使用 RAG

这一阶段我们需要将检索得到的相关知识注入 prompt，大模型参考当前提问和相关知识，生成相应的答案。这里的关键在于 prompt 的构造。

我们使用的**通义千问-plus** 模型。根据官方文档的介绍，模型的基本调用方法如下：

```python
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '你是谁？'}
        ]
)
print(completion.choices[0].message.content)
```

所以我们可以将检索出的与问题相关的文档做一些处理后提供给大模型，并告诉大模型只能使用检索到的上下文来回答问题。

代码如下：

```python
try:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    context_str = "\n\n".join([f"{i+1}. {doc.page_content.replace('data: ', '')}" for i, doc in enumerate(retrieved_docs)])
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
        {'role': 'system', 'content': '你是专业的法律知识问答助手。你需要使用以下检索到的上下文片段来回答问题，检索到的上下文如下：'},
        {'role': 'system', 'content': context_str},
        {'role': 'system', 'content': '你只能使用上述上下文来回答下面的问题，禁止根据常识和已知信息回答问题。如果上下文中没有足够依据，直接回答“未找到相关答案”。'},
        {'role': 'user', 'content': query}
        ]
    )
    print("=================New QA===================")
    print("Context:\n", context_str)
    print("Question:", query)
    res = completion.choices[0].message.content
    print("Answer:", res)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
    res = "ERROR"
```

其中使用了以下代码来处理检索后得到的文档内容以去除文档内容中开头的 `data:` 并分隔文档：

```python
context_str = "\n\n".join([f"{i+1}. {doc.page_content.replace('data: ', '')}" for i, doc in enumerate(retrieved_docs)])
```

#### 不使用 RAG

为了对比 RAG 与普通检索的区别以及引入 RAG 前后大模型在专业搜索上的区别，我们还实现了不使用 RAG 技术时即直接询问大模型时大模型的回答结果。

在代码实现上，只需要在调用大模型时不向它提供额外的信息即可，也不做额外的要求，也就是通过下面的代码调用大模型进行回答：

```python
response = client.chat.completions.create(
    model="qwen-turbo",
    messages=[
        {'role': 'system', 'content': '你是一名专业的法律顾问，请直接回答下面的问题：'},
        {'role': 'user', 'content': question}
    ]
)
```

## 5. 实验结果

### 使用 RAG

使用 RAG 技术时大模型对各个问题的输出如下：

![1-4](figs/1-4.png)

![5-6](figs/5-6.png)

这里因为有些问题的相关上下文太长不方便截图，所以我们最终没有输出相关的上下文，只输出了问题和答案。

根据输出可以看出，大模型成功回答了第1，3，5，6个问题，而对于第2个和第4个问题大模型回答未找到相关答案，这可能是因为文档中确实没有相关内容，也可能是我们在数据处理或检索方便存在问题，但是不管怎样，说明大模型确实只根据我们提供的文档来生成回答，如果提供的文档中没有相关内容，大模型就会回答”未找到相关答案“。

### 不使用 RAG

不使用 RAG 技术，直接询问大模型时，大模型的输出如下：

![1](figs/1.png)

![2](figs/2.png)

![3](figs/3.png)

## 6. 实验结果分析

通过对比引入 RAG 前后大模型的输出和理论分析，我们可以得到以下结论：

引入 RAG 前，大模型依赖其自身的训练知识进行回答。可能存在的问题是其知识库过于庞大而导致回答不够专业，并且由于模型是基于历史数据训练的，所以在遇到新的信息或事件时会显得滞后。此外，在大模型的知识库中缺乏相关知识时，大模型可能会出现幻觉现象，生成的回答不可靠。

而引入 RAG 后，大模型可以依赖我们注入的高度相关且专业的知识来回答，回答的专业性自然更高。此外，我们可以实时检索外部知识库，这样就模型能够引用最新的法规、案例等信息，提升回答的时效性和准确性。并且本次实验要求大模型未能基于提供的上下文中生成答案时回答”未找到答案“，有效避免了幻觉现象。



