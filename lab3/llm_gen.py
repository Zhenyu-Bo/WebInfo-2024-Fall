from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

def rag_answer(query: str, vectorstore):
    # 1. 检索相关文档
    # # 加载向量化模型
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # # 加载 FAISS 索引
    # vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    
    # 执行相似性检索
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    # print(retrieved_docs)

    # 2. 构造上下文文本
    context_str = "\n".join([doc.page_content.replace("data: ", "") for doc in retrieved_docs])
    # print(context_str)

    # 3. 构建 Prompt
    # ChatPromptTemplate 支持的消息类型: 'human', 'user', 'ai', 'assistant', or 'system'.
    prompt = ChatPromptTemplate([
        ("system", "你是专业的法律知识问答助手。你需要使用以下检索到的上下文片段来回答问题，禁止根据常识和已知信息回答问题。如果你不知道答案，直接回答“未找到相关答案”。检索到的上下文如下："),
        ("system", context_str),
        ("system", "你只能使用上述上下文来回答下面的问题，禁止根据常识和已知信息回答问题。如果你不知道答案，直接回答“未找到相关答案”。"),
        ("user", query),
        # ("ai", "AI's answer:"),
        # ("assistant", "Assistant's answer:"),
    ])

    # 4. 创建 LLM
    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=512,
            # do_sample=False,
            repetition_penalty=1.2,
            
            top_p=0.8,
            top_k=20,
            do_sample=True, # 设置为 True 以避免警告
        ),
        device=0  # 使用 GPU
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
    # 加载向量化模型
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # 加载 FAISS 索引
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    for question in questions:
        answer = rag_answer(question, vectorstore)
        print(answer)
