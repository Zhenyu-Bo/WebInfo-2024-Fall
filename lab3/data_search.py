"""
这是一个用于文档检索的模块，包含相似性检索和全文检索功能。
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from data_pre import data_pre

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import os

def retrieve_similar_documents(query, faiss_index_path, top_k=5):
    """
    根据查询从 FAISS 索引中检索相似文档。

    输入:
        query (str): 用户的查询。
        faiss_index_path (str): FAISS 索引文件的保存路径。
        top_k (int): 返回的最相似文档数量。

    输出:
        List[Document]: 检索到的相似文档列表。
    """
    # 加载向量化模型
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # 加载 FAISS 索引
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 执行相似性检索
    similar_docs = vectorstore.similarity_search(query, k=top_k)
    
    return similar_docs

def create_search_index(documents, index_dir="indexdir"):
    """
    创建 Whoosh 全文检索索引。

    输入:
        documents (List[Document]): 要索引的文档列表。
        index_dir (str): 索引目录。

    输出:
        None
    """
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    
    schema = Schema(title=ID(stored=True), content=TEXT(stored=True))
    ix = create_in(index_dir, schema)
    writer = ix.writer()
    
    for i, doc in enumerate(documents):
        writer.add_document(title=str(i), content=doc.page_content)
    
    writer.commit()

def search_documents(query_str, index_dir="indexdir", top_k=5):
    """
    使用 Whoosh 进行全文检索。

    输入:
        query_str (str): 用户的查询。
        index_dir (str): 索引目录。
        top_k (int): 返回的最相似文档数量。

    输出:
        List[dict]: 检索到的相似文档列表。
    """
    ix = open_dir(index_dir)
    qp = QueryParser("content", schema=ix.schema)
    query = qp.parse(query_str)
    
    with ix.searcher() as searcher:
        results = searcher.search(query, limit=top_k)
        return [{"title": r["title"], "content": r["content"]} for r in results]

def main():
    """
    主函数，执行示例查询。
    """
    law_csv_path = "law_data.csv"
    qa_csv_path = "qa_data.csv"
    faiss_index_path = "faiss_index"
    
    documents = data_pre(law_csv_path, qa_csv_path, faiss_index_path)
    
    # 创建全文检索索引
    create_search_index(documents)
    
    # 示例查询
    query = "消费者权益保护法"
    print(f"查询: {query}")
    print("相似性检索结果:")
    similar_docs = retrieve_similar_documents(query, faiss_index_path)
    
    for i, doc in enumerate(similar_docs, 1):
        print(f"文档 {i}:")
        print(f"标签: {doc.metadata['label']}")
        content = doc.page_content.replace("data: ", "")
        print(f"内容: {content}\n")
        
    print("全文检索结果:")
    search_results = search_documents(query)
    for i, doc in enumerate(search_results, 1):
        print(f"文档 {i}:")
        print(f"内容: {doc['content']}\n")

if __name__ == "__main__":
    main()