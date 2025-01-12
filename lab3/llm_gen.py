from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

def rag_answer(query: str, faiss_index_path):
    # 1. 检索相关文档
    # 加载向量化模型
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # 加载 FAISS 索引
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 执行相似性检索
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    # print(retrieved_docs)

    # 2. 构造上下文文本
    context_str = "\n".join([doc.page_content.replace("data: ", "") for doc in retrieved_docs])
    # print(context_str)

    # 3. 构建 Prompt
    prompt = ChatPromptTemplate([
        ("system", "你是法律助理，以下是检索到的上下文，禁止使用常识回答："),
        ("system", context_str),
        ("system", "只能基于以上内容进行回答，否则回答'未找到相关答案'。"),
        ("user", query),
    ])

    # 4. 创建 LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.2,
        ),
    )

    # 5. 使用提示+LLM生成结果
    retriever = vectorstore.as_retriever()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    res = rag_chain.invoke(query)
    # print(res)
    
    return res

# 示例调用
if __name__ == "__main__":
    faiss_index_path = "faiss_index"
    # question = "借款人去世，继承人是否应履行偿还义务？"
    # answer = rag_answer(question, faiss_index_path)
    # print(answer)
    questions = ["借款人去世，继承人是否应履行偿还义务？",
                 "如何通过法律手段应对民间借贷纠纷？",
                 "没有赡养老人就无法继承财产吗？"]
    for question in questions:
        answer = rag_answer(question, faiss_index_path)
        print(answer)
