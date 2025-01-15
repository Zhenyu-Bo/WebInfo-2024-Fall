"""
这是一个用于文档检索的模块，包含相似性检索和全文检索功能。
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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
    faiss_index_path = "faiss_index_2"
    
    # data_pre_process(file_path, faiss_index_path)
    
    # 示例查询
    questions = ["借款人去世，继承人是否应履行偿还义务？",
                 "如何通过法律手段应对民间借贷纠纷？",
                 "没有赡养老人就无法继承财产吗？",
                 "谁可以申请撤销监护人的监护资格？",
                 '''你现在是一个精通中国法律的法官，请对以下案件做出分析：经审理查
                    明：被告人 xxx 于 2017 年 12 月，多次在本市 xxx 盗窃财物。具体事实如下：（一）2017 年 12 月 9 日 15 时许，被告人 xxx 在 xxx 店内，盗窃
                    白色毛衣一件（价值人民币 259 元）。现赃物已起获并发还。（二）20
                    17 年 12 月 9 日 16 时许，被告人 xx 在本市 xxx 店内，盗窃米白色大衣
                    一件（价值人民币 1199 元）。现赃物已起获并发还。（三）2017 年 12
                    月 11 日 19 时许，被告人 xxx 在本市 xxx 内，盗窃耳机、手套、化妆镜
                    等商品共八件（共计价值人民币 357.3 元）。现赃物已起获并发还。（四）
                    2017 年 12 月 11 日 20 时许，被告人 xx 在本市 xxxx 内，盗窃橙汁、牛
                    肉干等商品共四件（共计价值人民币 58.39 元）。现赃物已起获并发还。
                    2017 年 12 月 11 日，被告人 xx 被公安机关抓获，其到案后如实供述了
                    上述犯罪事实。经鉴定，被告人 xxx 被诊断为精神分裂症，限制刑事责
                    任能力，有受审能力。''',
                "你现在是一个精通中国法律的法官，请对以下案件做出分析：2012 年 5月 1 日，原告 xxx 在被告 xxxx 购买“玉兔牌”香肠 15 包，其中价值 558.6 元的 14 包香肠已过保质期。xxx 到收银台结账后，即径直到服务台索赔，后因协商未果诉至法院，要求 xxxx 店支付 14 包香肠售价十倍的赔偿金 5586 元。"]
    
    # 加载向量化模型和FAISS索引到内存中
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    faiss_index_path = "faiss_index_2"
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    for query in questions:
        print(f"Question: {query}")
        print("相似性检索结果:")
        # similar_docs = retrieve_similar_documents(query, faiss_index_path)
        similar_docs = vectorstore.similarity_search(query, k=5)
        for i, doc in enumerate(similar_docs, 1):
            print(f"文档 {i}:")
            content = doc.page_content.replace("data: ", "")
            print(f"内容: {content}\n")

if __name__ == "__main__":
    main()