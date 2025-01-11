# 实验报告：基于 LangChain 的法律检索增强生成（RAG）系统

## 一、实验背景

随着人工智能技术的迅猛发展，检索增强生成（Retrieval-Augmented Generation, RAG）系统在知识密集型任务中展现出强大的能力。RAG 系统通过结合信息检索技术与大型语言模型（LLM），能够从外部知识库中提取相关信息，并将其作为上下文输入生成模型，以提供更准确和上下文相关的答案。本实验旨在利用 LangChain 框架，基于公开的法律知识数据库，开发一个简单的 RAG 问答应用示例，以比较大模型的生成式检索与普通检索的区别，并评估引入 RAG 后大模型在专业搜索上的表现。

## 二、实验介绍

### （一）RAG 介绍

RAG 模型由 Facebook AI Research（FAIR）团队于 2020 年首次提出，主要分为三个阶段：

1. **索引（Indexing）**：将外部数据源中的内容转换为向量表示，并存储在向量数据库中。
2. **检索（Retrieval）**：使用用户的查询向量在向量数据库中搜索最相关的信息。
3. **生成（Generation）**：将检索到的信息与用户的查询结合，通过语言模型生成最终的回答。

### （二）基于 LangChain 实现 RAG 系统

**LangChain** 是一个用于开发由大语言模型（LLM）支持的应用程序的框架，提供了各种工具和接口，帮助开发者集成和管理语言模型的功能。

本实验通过以下步骤实现 RAG 系统：

1. **数据准备**：加载和清洗法律条文及问答数据，进行文本分割和向量化，并存储在向量数据库中。
2. **数据检索**：根据用户查询，从向量数据库中检索相关文档。
3. **LLM 生成**：将检索到的文档与用户查询结合，生成法律问答。

## 三、实验内容

### （一）数据集说明

我们手动将数据集分为了两个文件，便于处理。

- **law_data.csv**：包含中华人民共和国法律手册最核心的约600条法律条文。
- **qa_data.csv**：包含百度知道约2400条法律问答数据。

### （二）任务说明

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

## 四、实验过程

### （一）数据准备阶段

代码位于`./data_pre.py`

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

#### 1. 函数 `data_pre`

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

#### 2. 函数 `main`

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

### 4. 示例输出

输出如下：

```
(base) PS E:\Projects\WebInfo-2024-Fall\lab3> & D:/ProgramData/Miniconda3/python.exe e:/Projects/WebInfo-2024-Fall/lab3/data_pre.py      
加载法律数据...
加载问答数据...
分割法律数据...
添加标签...
向量化并存储文档...
法律条文示例：

文档 1:
标签: law
内容: 民法商法-农民专业合作社法2017-12-27：第十五条 农民专业合作社章程应当载明下列事项：,（一）名称和住所；,（二）业务范围；,（三）成员资格及入社、退社和除名；,（四）成员的权利和义务；,（五）组织机构及其产生办法、职权、任期、议事规则；,（六）成员的出资方式、出资额，成员出资的转让、继承、担保；,（七）财务管理和盈余分配、亏损处理；,（八）章程修改程序；,（九）解散事由和清算办法；,（十）公告事项及发布方式 ；,（十一）附加表决权的设立、行使方式和行使范围；,（十二）需要载明的其他事项。

文档 2:
标签: law
内容: 民法商法-个人独资企业法1999-08-30：第十七条 个人独资企业投资人对本企业的财产依法享有所有权，其有关权利可以依法进行转让或继承 。

文档 3:
标签: law
内容: 民法商法-个人独资企业法1999-08-30：第二十六条 个人独资企业有下列情形之一时，应当解散：,（一）投资人决定解散；,（二）投资人死 亡或者被宣告死亡，无继承人或者继承人决定放弃继承；,（三）被依法吊销营业执照；,（四）法律、行政法规规定的其他情形。

问答示例：

文档 1:
标签: qa
内容: 盗窃罪的犯罪客体是什么，盗窃罪的犯罪主体
盗窃罪的客体要件本罪侵犯的客体是公私财物的所有权。侵犯的对象，是国家、集体或个人的财物，一般是指动产而言，但不动产上之附着物，可与不动产 分离的，例如，田地上的农作物，山上的树木、建筑物上之门窗等，也可以成为本罪的对象。另外，能源如电力、煤气也可成为本罪的对象。盗窃罪侵犯的 客体是公私财物的所有权。所有权包括占有、使用、收益、处分等权能。这里的所有权一般指合法的所有权，但有时也有例外情况。根据《最高人民法院关 于审理盗窃案件具体应用法律若干问题的解释》(以下简称《解释》)的规定：“盗窃违禁品，按盗窃罪处理的，不计数额，根据情节轻重量刑。盗窃违禁品或犯罪分子不法占有的财物也构成盗窃罪。”

文档 2:
标签: qa
内容: 高利贷还不还不知如何是好。
我在微信有凭证上借了6000块钱，到手4800，因为之前总借而且付着高额利息不说还要给他们私人红包，我只想还他赢得的利息，因为是高利我不想给他， 但是他们放高利贷的实在讨厌，骚扰家人，我想打官司可以吗？
没必要的，还没到官司这步，建议你跟家人把这事情说清楚，免得他们经常利用你家人威胁到你，最好的办法是把借款还清，从此和他们没有任何一个关系 了，大家都安心。

文档 3:
标签: qa
内容: 请问，父亲生前欠下债务要还吗
你继承遗产的话，有可能需要

```

输出符合预期。

## 五、实验总结

