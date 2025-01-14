from openai import OpenAI


GDH_API_KEY = "sk-66ccd9858bc24cce93e1b5f9ae542262"

def direct_answer(question: str):
    try:
        client = OpenAI(
            api_key=GDH_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {'role': 'system', 'content': '你是一名专业的法律顾问，请直接回答下面的问题：'},
                {'role': 'user', 'content': question}
            ]
        )
        res = response.choices[0].message.content
        print("=================New QA===================")
        print("User:", question)
        print("Assistant:", res)
        # 将输出存储到文件中
        with open("answer_no_rag.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {question}\n")
            f.write(f"Assistant: {res}\n\n")
    except Exception as e:
        print("错误信息：", e)
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")


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
        direct_answer(question)