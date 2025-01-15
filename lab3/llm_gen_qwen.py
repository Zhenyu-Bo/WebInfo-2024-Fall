from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

GDH_API_KEY = "sk-66ccd9858bc24cce93e1b5f9ae542262"
BZY_API_KEY = "sk-a11ed5e5b3a3473e86b42adb46571436"

# 加载向量化模型和FAISS索引到内存中
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
faiss_index_path = "faiss_index_2"
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

def rag_answer(query: str):
    # 1. 检索相关文档
    retrieved_docs = vectorstore.similarity_search(query, k=5)

    # 2. 构造上下文文本
    context_str = "\n".join([doc.page_content.replace("data: ", "") for doc in retrieved_docs])

    # 3. 接入通义千问qwen-turbo API，构建 Prompt
    try:
        client = OpenAI(
            # api_key= GDH_API_KEY,
            api_key = BZY_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        context_str = "\n\n".join([f"{i+1}. {doc.page_content.replace('data: ', '')}" for i, doc in enumerate(retrieved_docs)])
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
            {'role': 'system', 'content': '你是专业的法律知识问答助手。你需要使用以下检索到的上下文片段来回答问题，检索到的上下文如下：'},
            {'role': 'system', 'content': context_str},
            {'role': 'system', 'content': '你只能使用上述上下文来回答下面的问题，禁止根据常识和已知信息回答问题。如果上下文中没有足够依据，直接回答“未找到相关答案”。'},
            {'role': 'user', 'content': query},
            {'role': 'system', 'content': '请根据上下文和问题一步一步推导出答案。'}
            ]
        )
        print("=================New QA===================")
        # print("User:", query)
        print("Question:", query)
        # print("Context:\n", context_str)
        # print("=================QA===================")
        res = completion.choices[0].message.content
        print("Answer:", res)
        # print()
        # print()
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        res = "ERROR"

# 示例调用
if __name__ == "__main__":
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
    for question in questions:
        rag_answer(question)