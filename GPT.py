import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from openai import OpenAI
import json
from difflib import get_close_matches
import numpy as np
import jieba

file_path = "example.xlsx"
xls = pd.ExcelFile(file_path)
sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
standard_dept_file_path = 'D:/标准科室列表.xlsx'
std_xls = pd.ExcelFile(standard_dept_file_path)
std_sheets = {sheet_name: std_xls.parse(sheet_name) for sheet_name in std_xls.sheet_names}

client = OpenAI(
    api_key = 'sk-w003ue0vYegC7cbFVRYbkOd1kpCJB3AHkIljE3n3AT7OvXWj',
    base_url = "https://api.moonshot.cn/v1",
)

def determine_source(new_dialog, sheets):
    # 将所有工作表的数据合并
    all_data = []
    for sheet_name, df in sheets.items():
        for dialogue in df['dialogue']:
            all_data.append((sheet_name, dialogue))
    
    # 创建一个 DataFrame 存储所有数据
    all_df = pd.DataFrame(all_data, columns=['sheet_name', 'dialogue'])
    
    # 使用 TfidfVectorizer 对所有数据进行向量化
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_df['dialogue'].astype(str))
    
    # 对新的对话进行向量化
    new_dialog_tfidf = vectorizer.transform([new_dialog])
    
    # 计算相似度
    cos_sim = cosine_similarity(new_dialog_tfidf, tfidf_matrix).flatten()
    max_sim_index = cos_sim.argmax()
    
    # 返回最相似对话的工作表名称
    best_source = all_df.loc[max_sim_index, 'sheet_name']
    
    return best_source

def find_similar_examples(new_dialog, df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['dialogue'].astype(str))
    new_dialog_tfidf = vectorizer.transform([new_dialog])
    cos_sim = cosine_similarity(new_dialog_tfidf, tfidf_matrix)
    similar_indices = cos_sim[0].argsort()[-3:][::-1]  # 获取最相似的三个索引
    second_most_similar_index = similar_indices[1]  # 取第二个相似的索引
    return df.iloc[[second_most_similar_index]]


# 生成COT
def generate_cot(dialogue, answer):
    prompt = f"""对话: {dialogue}
               问题: 患者应该挂什么科？
               请根据{dialogue}和{answer}来生成详细的推理过程和最终的建议，建议去的专业是{answer},输出不要包含对话，只包含推理过程和建议"""
    response =  client.chat.completions.create(
        model= "moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "你是一名医生助手。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.7,
        top_p=0.9,
        n=1,
        stop=None
    )
    cot = response.choices[0].message.content
    lines = [line.strip() for line in  cot.split('\n') if line.strip()]  
    formatted_output = ' '.join(lines)
    return formatted_output


# 生成Few Shot
def generate_few_shot(similar_examples):
    few_shot = ""
    for _, row in similar_examples.iterrows():
        dialogue = row['dialogue']
        answer = row['answer']
        cot = generate_cot(dialogue, answer)
        few_shot += f"""对话: {dialogue}
        问题: 患者应该挂什么科？
        思维链: {cot}"""
    return few_shot


# 定义一个函数来获取候选科室
def get_candidate_departments(department, std_sheets):
    if department in std_sheets:
        return std_sheets[department].iloc[:, 0].tolist()  # 获取第一列的所有值
    return []
    


# 生成最终Prompt
def generate_final_prompt(new_dialog, few_shot,departments):
    prompt = (f"{few_shot}"
              f"新的对话: {new_dialog}"
              f"问题: 患者应该挂什么科？"
              f"请根据以上示例生成详细的推理过程和最终的建议，不要包含对话，只包含推理过程和建议，最终的建议挂的科室在{departments}寻找。")
    return prompt


# API输出
def get_api_output(prompt):
    response =  client.chat.completions.create(
        model = "moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "你是一名医生助手。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
        temperature=0.7,
        top_p=0.9,
        n=1,
        stop=None
    )
    api_out = response.choices[0].message.content

    lines = [line.strip() for line in api_out.split('\n') if line.strip()] 
    formatted_output = ' '.join(lines)
    return formatted_output


def find_closest_department(api_out_content, candidate_departments):
    # 确保 api_out_content 是一个字符串
    if isinstance(api_out_content, tuple):
        api_out_content = ''.join(api_out_content)   
    # 找到并提取建议内容
    suggestion_start = api_out_content.find('建议')
    if suggestion_start != -1:
        suggestion = api_out_content[suggestion_start:].replace('建议', '').strip()
    else:
        return "无建议内容"
    #print(suggestion)
    # 分词
    suggestion_tokens = ' '.join(jieba.lcut(suggestion))
    department_tokens = [' '.join(jieba.lcut(dept)) for dept in candidate_departments]

    # 使用TF-IDF计算相似度
    vectorizer = TfidfVectorizer().fit_transform([suggestion_tokens] + department_tokens)
    vectors = vectorizer.toarray()

    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[0][1:]  # 与候选科室的相似度

    # 输出相似度分数，便于调试
    # for dept, score in zip(candidate_departments, similarity_scores):
    #     print(f"科室: {dept}, 相似度得分: {score}")

    # 找到最相似的科室
    best_match_index = np.argmax(similarity_scores)
    return candidate_departments[best_match_index] if similarity_scores[best_match_index] > 0 else "其它"


# 将结果保存为 JSON 文件
def save_to_json(new_dialog, departments, few_shot, api_output, answer_department, source, file_path):
   
    data = {
        "dialogue": new_dialog,
        "department": departments,
        "few_shot": few_shot,
        "question": "患者应该挂什么科？",
        "api_output": api_output,
        "answer_department": answer_department,
        "sample_id": "sample_12",  # 更新样本 ID
        "source": source
    }
    # 使用 json.dumps 格式化 JSON 字符串
    formatted_json = json.dumps(data, ensure_ascii=False, indent=4)

    # 自定义处理数组的输出格式
    # 查找所有数组的内容
    formatted_json = re.sub(r'(\[[^\[\]]*\])', lambda m: m.group(1).replace('\n', '').replace('  ', ''), formatted_json)
    formatted_json = formatted_json.replace('\\n', '\n')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(formatted_json)

def main(file_path,new_dialogue):
    source = determine_source(new_dialog, sheets)
    similar_examples = find_similar_examples(new_dialog, sheets[source])
    few_shot = generate_few_shot(similar_examples)
    departments = get_candidate_departments(source, std_sheets)
    final_prompt = generate_final_prompt(new_dialog, few_shot,departments )
    api_out = get_api_output(final_prompt)
    answer_department = find_closest_department(api_out, departments)
    save_to_json(new_dialog, departments, few_shot, api_out, answer_department, source, file_path)


file_path = 'D:/dialog-T.json'
new_dialog =  "患者:2岁小孩嘴里长了很多小泡,应该去哪个科？"
main(file_path, new_dialog)


