import os
import faiss
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

def data_pre(law_csv_path, qa_csv_path, faiss_index_path):
    """
    数据预处理函数，用于加载、分割、向量化，并将文档存储到 FAISS 数据库。

    输入:
        law_csv_path (str): law_data.csv 文件的路径。
        qa_csv_path (str): qa_data.csv 文件的路径。
        faiss_index_path (str): 生成的 FAISS 索引文件保存路径。

    输出:
        - 返回：List[Document]: 包含所有分割、打标签后的文档列表，用于main函数展示示例文档。
        - 保存：生成的 FAISS 索引文件存储在 faiss_index_path 目录下。
        PS：如果 faiss_index_path 目录已存在，则不会重新生成，直接预览示例

    数据库结构 (FAISS 本地索引):
        - index.faiss: 存储向量和索引信息的二进制文件
        - index.pkl: 存储文档 metadata 的序列化文件
          (包括每个文档的 page_content、metadata 等信息)，
          便于在检索时通过索引找到对应原文。

    documents 结构:
        - page_content (str): 文档的文本内容，为一条法条或一组问答。
        - metadata (dict): 文档的元数据，包括以下字段:
            - label (str): 文档的标签，用于区分 "law"（法条） 和 "qa"（问答）
    """
    # 1. 加载CSV文件
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

    qa_loader = CSVLoader(
        file_path=qa_csv_path, 
        encoding="utf-8", 
        source_column="data",
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'escapechar': '\\'
        }
    )

    print("加载法律数据...")
    law_docs = law_loader.load()
    print("加载问答数据...")
    qa_docs = qa_loader.load()
    
    # 去除每个文档内容中的 "data: " 前缀
    for doc in law_docs:
        doc.page_content = doc.page_content.lstrip('data: ')
    for doc in qa_docs:
        doc.page_content = doc.page_content.lstrip('data: ')

    # 2. 文本分割
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=128,
        chunk_overlap=0,
        length_function=len,
    )
    
    # 对法条按行分割
    print("分割法律数据...")
    law_splits = text_splitter.split_documents(law_docs)

    # 3. 添加标签
    print("添加标签...")
    for doc in law_splits:
        doc.metadata["label"] = "law"
    for doc in qa_docs:
        doc.metadata["label"] = "qa"
        
    documents = law_splits + qa_docs

    # 4. 向量化和数据入库
    if not os.path.exists(faiss_index_path):
        os.makedirs(faiss_index_path)
        print("加载预训练模型...")
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        print("向量化文档...")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("保存本地索引...")
        vectorstore.save_local(faiss_index_path)
    
    return documents

def data_pre_process(file_path, faiss_index_path):
    data = CSVLoader(file_path=file_path, encoding='utf-8').load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=256, chunk_overlap=0)
    data = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    if data:
        sample_text = data[0].page_content  # 使用实际数据中的一个样本
        vector_dim = len(embeddings.embed_query(sample_text))
    else:
        raise ValueError("No data available to determine vector dimension.")

    index = faiss.IndexFlatL2(vector_dim)
    # print("Index dimension:", vector_dim)
    # index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_documents(data)
    
    vector_store.save_local(faiss_index_path)
    
    print("FAISS index saved at", faiss_index_path)


def main():
    law_csv_path = "law_data.csv"
    qa_csv_path = "qa_data.csv"
    faiss_index_path = "faiss_index_2"
    
    documents = data_pre(law_csv_path, qa_csv_path, faiss_index_path)
    
    # 分组文档
    law_examples = [doc for doc in documents if doc.metadata['label'] == 'law'][:10]
    qa_examples = [doc for doc in documents if doc.metadata['label'] == 'qa'][:10]
    
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

if __name__ == "__main__":
    main()
    # data_pre_process("law_data_3k.csv", "faiss_index")