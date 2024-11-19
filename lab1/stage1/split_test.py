import csv
import jieba
import pkuseg
import ast
import time

def is_word_valid(word):
    """
    检查词语是否包含非法字符。
    """
    for char in word:
        if char in ILLEGAL_CHARS:
            return False
    return True

def escape_illegal_chars(text):
    """
    将非法字符转换为 Unicode 转义序列。
    """
    return text.encode('unicode_escape').decode('utf-8')

# 定义非法字符集
ILLEGAL_CHARS = {'§', '#', '$', '%', '&', '*', '!', '@', '^', '(', ')', '-', '=', '+', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/'}

# 选择分词工具
tool_choice = input("请选择分词工具（输入 'jieba' 或 'pkuseg'）：")
if tool_choice.lower() == 'jieba':
    use_jieba = True
    print("使用 Jieba 分词")
elif tool_choice.lower() == 'pkuseg':
    use_jieba = False
    print("使用 PKUSeg 分词")
else:
    print("输入有误，默认使用 Jieba 分词")
    use_jieba = True

# 加载停用词表
stopwords_file = 'data/cn_stopwords.txt'
stopwords = set()
with open(stopwords_file, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip()
        if word:
            stopwords.add(word)

# 加载同义词词典
synonyms_file = 'data/syno_from_baidu_hanyu.txt'
synonym_dict = {}
with open(synonyms_file, 'r', encoding='utf-8') as f:
    for line in f:
        words = line.strip().split()
        if words:
            representative = words[0]
            for word in words:
                synonym_dict[word] = representative

# 定义文件列表
file_list = [
    {'input': '../data/selected_book_top_1200_data_tag.csv', 'output': 'data/book_output.csv'},
    {'input': '../data/selected_movie_top_1200_data_tag.csv', 'output': 'data/movie_output.csv'}
]

# 初始化 pkuseg 分词器
if not use_jieba:
    seg = pkuseg.pkuseg()

for file_pair in file_list:
    input_file = file_pair['input']
    output_file = file_pair['output']
    print(f"正在处理文件：{input_file}")

    start_time = time.time()

    # 初始化统计信息
    total_tokens = 0
    removed_stopwords = 0
    replaced_synonyms = 0

    with open(input_file, 'r', encoding='utf-8') as csvfile_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as csvfile_out:
        reader = csv.reader(csvfile_in)
        writer = csv.writer(csvfile_out, quoting=csv.QUOTE_MINIMAL)

        # 跳过输入文件的第一行
        next(reader)

        # 写入输出文件的表头
        header = ['id', 'words']
        writer.writerow(header)

        for row in reader:
            item_id = row[0]
            tags_str = row[1]

            # 解析 Tags 字段
            try:
                tags_set = ast.literal_eval(tags_str)
            except Exception as e:
                print(f"解析文件 {input_file} 中的 ID {item_id} 的标签时出错: {e}")
                continue

            # 分词并收集所有词语
            all_words = []
            for tag in tags_set:
                if use_jieba:
                    words = jieba.lcut(tag)
                else:
                    words = seg.cut(tag)
                all_words.extend(words)

            total_tokens += len(all_words)

            # 去除停用词和非法词语
            filtered_words = []
            for word in all_words:
                word = word.strip()
                if word and word not in stopwords and is_word_valid(word):
                    filtered_words.append(word)
                else:
                    removed_stopwords += 1

            # 替换同义词
            new_tags = set()
            for word in filtered_words:
                representative = synonym_dict.get(word, word)
                if representative != word:
                    replaced_synonyms += 1
                representative = representative.replace("'", "\\'")
                new_tags.add(representative)

            # 重建 Tags 字符串
            new_tags_str = "{" + ", ".join(f"'{tag}'" for tag in new_tags) + "}"
            writer.writerow([item_id, new_tags_str])

    elapsed_time = time.time() - start_time

    print(f"文件 {input_file} 处理完成，结果已保存到 {output_file}")
    print(f"分词过程运行时间：{elapsed_time:.2f} 秒")
    print(f"总词数：{total_tokens}")
    print(f"去除的停用词和非法词语数量：{removed_stopwords}")
    print(f"替换的同义词数量：{replaced_synonyms}\n")

print("所有文件处理完成！")