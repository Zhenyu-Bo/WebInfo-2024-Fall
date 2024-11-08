import csv
import pandas as pd
import ast
import re

def read_inverted_index(file_path):
    """
    读取倒排索引表，返回字典格式的数据，包含 id_list 和 skip_table。
    """
    inverted_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row['word']
            id_list = ast.literal_eval(row['id_list'])
            skip_table = ast.literal_eval(row['skip_table'])
            # 保存为字典，包含 id_list 和 skip_table
            inverted_index[word] = {
                'id_list': id_list,
                'skip_table': skip_table
            }
    return inverted_index

def read_all_ids(file_path):
    """
    读取所有ID，返回ID的集合。
    """
    all_ids = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            id = int(line.strip())
            all_ids.add(id)
    return all_ids

def tokenize(expression):
    """
    将布尔表达式分词。
    """
    tokens = re.findall(r'AND|OR|NOT|\w+|\(|\)', expression)
    return tokens

def infix_to_postfix(tokens):
    """
    将中缀表达式转换为后缀表达式（逆波兰表达式）。
    """
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
    output = []
    operator_stack = []
    operators = {'AND', 'OR', 'NOT'}
    for token in tokens:
        if token not in operators and token not in {'(', ')'}:
            output.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()  # 弹出 '('
        else:  # 操作符
            while operator_stack and operator_stack[-1] != '(' and precedence[operator_stack[-1]] >= precedence[token]:
                output.append(operator_stack.pop())
            operator_stack.append(token)
    while operator_stack:
        output.append(operator_stack.pop())
    return output

def intersect_with_skips(p1, p2, skip_p1, skip_p2):
    """
    使用跳表的交集操作，优化 AND 操作。
    """
    answer = []
    i = j = 0
    len1, len2 = len(p1), len(p2)
    while i < len1 and j < len2:
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            # 检查是否可以跳过
            next_skip_idx = get_next_skip_idx(skip_p1, i)
            if next_skip_idx != -1 and p1[next_skip_idx] <= p2[j]:
                i = next_skip_idx
            else:
                i += 1
        else:
            # p2[j] < p1[i]
            next_skip_idx = get_next_skip_idx(skip_p2, j)
            if next_skip_idx != -1 and p2[next_skip_idx] <= p1[i]:
                j = next_skip_idx
            else:
                j += 1
    return answer

def get_next_skip_idx(skip_table, current_idx):
    """
    获取跳表中下一个可跳转的索引。
    """
    for skip in skip_table:
        if skip['index'] > current_idx:
            return skip['index']
    return -1  # 没有可跳转的索引

def evaluate_postfix(postfix_tokens, inverted_index, all_ids):
    """
    评估后缀表达式，返回符合条件的ID列表。
    """
    stack = []
    operators = {'AND', 'OR', 'NOT'}
    for token in postfix_tokens:
        if token not in operators:
            if token in inverted_index:
                stack.append((inverted_index[token]['id_list'], inverted_index[token]['skip_table']))
            else:
                stack.append(([], []))
        elif token == 'NOT':
            operand_ids, _ = stack.pop()
            result = list(sorted(all_ids - set(operand_ids)))
            stack.append((result, []))
        else:
            right_ids, right_skips = stack.pop()
            left_ids, left_skips = stack.pop()
            if token == 'AND':
                result = intersect_with_skips(left_ids, right_ids, left_skips, right_skips)
            elif token == 'OR':
                result = sorted(set(left_ids) | set(right_ids))
            stack.append((result, []))
    result_ids, _ = stack.pop() if stack else ([], [])
    return set(result_ids)

def display_results(result_ids, words_df):
    """
    根据查询结果的ID，从 DataFrame 中查找并打印ID和标签。
    """
    df = words_df[words_df['id'].isin(result_ids)]
    if not df.empty:
        for index, row in df.iterrows():
            id = row['id']
            words = ast.literal_eval(row['words'])
            print(f"ID: {id}")
            print(f"标签: {', '.join(words)}\n")
    else:
        print("没有符合条件的结果。")

def main():
    print("正在加载数据，请稍候...")

    # 加载书籍数据
    book_inverted_index = read_inverted_index('book_inverted_index_table.csv')
    book_all_ids = read_all_ids('Book_id.txt')
    book_words_df = pd.read_csv('book_words.csv', dtype={'id': int, 'words': str})

    # 加载电影数据
    movie_inverted_index = read_inverted_index('movie_inverted_index_table.csv')
    movie_all_ids = read_all_ids('Movie_id.txt')
    movie_words_df = pd.read_csv('movie_words.csv', dtype={'id': int, 'words': str})

    print("数据加载完成！")

    while True:
        choice = input("请选择查询类型（1 - 书籍，2 - 电影）：\n")
        if choice == '1':
            inverted_index = book_inverted_index
            all_ids = book_all_ids
            words_df = book_words_df
        elif choice == '2':
            inverted_index = movie_inverted_index
            all_ids = movie_all_ids
            words_df = movie_words_df
        else:
            print("输入错误，请输入 1 或 2。")
            continue  # 重新开始循环

        expression = input("请输入布尔查询表达式：\n")
        tokens = tokenize(expression)
        postfix_tokens = infix_to_postfix(tokens)
        result_ids = evaluate_postfix(postfix_tokens, inverted_index, all_ids)
        if result_ids:
            print("查询结果：\n")
            display_results(result_ids, words_df)
        else:
            print("没有符合条件的结果。")

        cont = input("是否继续查询？(Y/N): ")
        if cont.strip().lower() != 'y':
            print("感谢您的使用！")
            break  # 退出循环，结束程序

if __name__ == "__main__":
    main()