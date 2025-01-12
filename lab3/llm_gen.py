from data_search import search_documents
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

def rag_answer(query: str):
    # 1. 检索相关文档
    retrieved_docs = search_documents(query_str=query, index_dir="indexdir", top_k=5)

    # 2. 构造上下文文本
    context_str = "\n".join([doc["content"] for doc in retrieved_docs])

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
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.2,
        },
    )

    # 5. 使用提示+LLM生成结果
    rag_chain = (
        {"context": retrieved_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(query)

# 示例调用
if __name__ == "__main__":
    question = "借款人去世，继承人是否应履行偿还义务？如果借款人有财产，则如何处理？"
    answer = rag_answer(question)
    print(answer)