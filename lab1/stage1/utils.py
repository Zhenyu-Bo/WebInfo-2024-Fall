# -*- coding: utf-8 -*-
# Author: Zhenyu Bo
# Date: 2024-11-14

import pickle
import jieba

class Utils:
    def __init__(self) -> None:
        # 加载自定义词典
        with open("data/word_dict.pkl", "rb") as f:
            custom_words = pickle.load(f)
        
        # 将自定义词添加到 jieba 词典中
        for word in custom_words:
            jieba.add_word(word)
        
        # 加载停用词
        with open('data/stopwords.txt', encoding='utf8') as f:
            self.stopwords = set(line.strip() for line in f)
        
        # 添加额外的停用词
        self.stopwords.update(['', '\n', ' ', '\u3000'])

    def split(self, text: str) -> set:
        # 使用 jieba 对文本进行分词
        words = set(jieba.cut_for_search(text))
        # 去除停用词
        words -= self.stopwords
        return words

if __name__ == "__main__":
    util = Utils()
    print(util.split("鞠婧祎演的电影"))