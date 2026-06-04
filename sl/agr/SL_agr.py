import numpy as np
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def cos_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_top_n_results(query_emb, embend_list, top_n):
    scores = [
        (cos_sim(query_emb, item["embedding"]), item["text"])
        for item in embend_list
    ]
    scores.sort(reverse=True, key=lambda x: x[0])
    return [t for _, t in scores[:top_n]]

def load_embend_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_embeddings(texts, api_url="http://localhost:8000/embeddings"):
    payload = {"texts": texts}
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"]

def get_completion(messages, model="deepseek/deepseek-chat-v3-0324:free"):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        tools=[],
    )
    return response.choices[0].message

def answer_question_impl(query, std_type="cn", province=None):
    # query: 用户输入问题
    # std_type: "cn" | "loc"
    # province: 如果是地方标准，传省名
    query_emb = get_embeddings([query])[0]

    if std_type == "cn":
        embend_list = load_embend_file(r"E:\aaacode\ssll\josn\cn\cn.json")
        context = get_top_n_results(query_emb, embend_list, top_n=4)
    elif std_type == "loc" and province:
        loc_path = os.path.join(r"E:\aaacode\ssll\embend\loc", f"{province}.json")
        cn_path = r"E:\aaacode\ssll\josn\cn\cn.json"
        loc_embend = load_embend_file(loc_path)
        cn_embend = load_embend_file(cn_path)
        context = get_top_n_results(query_emb, loc_embend, top_n=3)
        context += get_top_n_results(query_emb, cn_embend, top_n=2)
    else:
        raise Exception("地方标准必须指定省份")

    # 拼prompt
    user_content = (
        "请根据以下标准内容回答我的问题。如果相关内容不存在则说明查无结果。\n"
        "后台知识库里的文本结构为"
        "第一句话\n"   
        "第二句话\n"
        "第三句话\n"   
        "第四句话\n"           
        "后台知识库里的表格都存成如下 JSON 条目格式：\n"
        "    \"<表号>\",\n"
        "    \"<表标题>\",\n"
        "    \"<列名1>\",\n"
        "    \"<列名2>\",\n"
        "    …\n"
        "    \"<row1_col1>\",\n"
        "    \"<row1_col2>\",\n"
        "    …\n"
        "    \"<row2_col1>\",\n"
        "    \"<row2_col2>\",\n"
        "    …\n"
        "请务必遵守以下解析规则：\n"
        "1. text[0] 是表号；text[1] 是表标题；\n"
        "2. text[2:] 前 n 项为列名（n = 列数）；剩余元素依次为各行单元格，按列序循环排列；\n"
        "3. 如某行“代码”列为空，则沿用上一行对应值；\n"
        "4. 输出格式如下：\n"
        "{\n"
        "  \"table_number\": \"<表号>\",\n"
        "  \"table_title\" : \"<表标题>\",\n"
        "  \"columns\": [\"<列名1>\", \"<列名2>\", …],\n"
        "  \"rows\": [\n"
        "    {\"<列名1>\": \"<值>\", …},\n"
        "    …\n"
        "  ]\n"
        "}\n"
        + "\n\n".join(context)
        + f"\n\n我的问题是：{query}"
    )
    messages = [
        {"role": "system", "content": "你是水利工程标准机器人，善于从标准文件内容中给出详细答案。"},
        {"role": "user", "content": user_content},
    ]
    return get_completion(messages)

def answer_question(query):
    std_type, prov = detect_standard_type(query)
    if std_type == "special":
        return f"目前{prov}的地方标准尚未收录。"
    elif std_type == "loc":
        return answer_question_impl(query, std_type="loc", province=prov)
    else:
        return answer_question_impl(query, std_type="cn", province=None)

def detect_standard_type(query):
    """返回 ('cn', None) 或 ('loc', 省份) 或 ('special', '香港/澳门')"""
    for reg in SPECIAL_REGION:
        if reg in query:
            return "special", reg
    for prov in PROVINCE_LIST:
        if prov in query:
            return "loc", prov
    return "cn", None


import re

PROVINCE_LIST = [
    "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "广西", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "宁夏", "内蒙古", "新疆", "西藏"
]
SPECIAL_REGION = ["香港", "澳门"]


if __name__ == "__main__":
    #query = "水闸的安全监测有哪些国家标准要求？"
    query="市单元工程划分标准是什么"
    print(answer_question(query))