import os
import json
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

# 加载 OpenAI 环境
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 本地向量生成
def get_embeddings(texts, api_url="http://localhost:8000/embeddings", batch_size=32):
    embeddings = []
    filtered = [t for t in texts if isinstance(t, str) and t.strip()]
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i : i + batch_size]
        payload = {"texts": batch}
        resp = requests.post(api_url, json=payload)
        if resp.status_code != 200:
            print(f"⚠️ Embedding API 批次 {i//batch_size} 返回错误 {resp.status_code}：{resp.text}")
            resp.raise_for_status()
        data = resp.json()
        if "embeddings" not in data:
            raise KeyError(f"接口返回缺少 'embeddings' 字段：{data}")
        embeddings.extend(data["embeddings"])
    return embeddings

# 替代原来的文件搜索逻辑，用 Chroma 查询
def answer_question_impl(query, collection, top_n):
    embedding = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_n,
        include=["documents"]
    )
    docs = results["documents"][0] if results["documents"] else []
    context = "\n\n".join(docs)

    prompt = (
        """
        请根据以下标准内容回答我的问题。如果相关内容不存在则说明查无结果。\n
        后台知识库里的文本结构为
        第一句话\n   
        第二句话\n
        第三句话\n
        第四句话\n      
        后台知识库里的表格都存成如下 JSON 条目格式：\n
            \"<表号>\",\n
            \"<表标题>\",\n
            \"<列名1>\",\n
            \"<列名2>\",\n
            …\n
            \"<row1_col1>\",\n
            \"<row1_col2>\",\n
            …\n
            \"<row2_col1>\",\n
            \"<row2_col2>\",\n
            …\n
        请务必遵守以下解析规则：\n
        1. text[0] 是表号；text[1] 是表标题；\n
        2. text[2:] 前 n 项为列名（n = 列数）；剩余元素依次为各行单元格，按列序循环排列；\n
        3. 如某行“代码”列为空，则沿用上一行对应值；\n
        4. 输出格式如下：\n
        {\n
          \"table_number\": \"<表号>\",\n
          \"table_title\" : \"<表标题>\",\n
          \"columns\": [\"<列名1>\", \"<列名2>\", …],\n
          \"rows\": [\n
            {\"<列名1>\": \"<值>\", …},\n
            …\n
          ]\n
        }\n
        """
    
        "请根据以下标准内容回答我的问题，如果相关内容不存在则说明查无结果：\n\n"
        f"{context}\n\n"
        f"我的问题是：{query}\n\n"
        "（回答问题，并说明相关资料来源-文件名和页码（可以是多个）"
    )
    messages = [
        {"role": "system", "content": "你是文件检索机器人，擅长基于 embedding 做相似度检索。"},
        {"role": "user",   "content": prompt},
    ]
    return get_completion(messages)

# 主调用逻辑，根据省份判断使用哪个集合
def answer_question(query):
    std_type, prov = detect_standard_type(query)

    client = PersistentClient(path=r"E:\aaacode\ssll\chromadb")
    coll_loc = client.get_or_create_collection("coll-loc")
    coll_cn = client.get_or_create_collection("coll_cn")

    if std_type == "special":
        return f"目前该的地方标准尚未收录。"
    elif std_type == "loc":
        ctx_loc = answer_question_impl(query, coll_loc, top_n=6)
        ctx_cn  = answer_question_impl(query, coll_cn,  top_n=4)
        return ctx_loc + "\n\n" + ctx_cn
    else:
        return answer_question_impl(query, coll_cn, top_n=10)

# LLM 接口调用
def get_completion(messages, model="deepseek/deepseek-chat-v3-0324:free"):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content

# 识别标准类型和地区
PROVINCE_LIST = [
    "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "广西", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "宁夏", "内蒙古", "新疆", "西藏"
]

def detect_standard_type(query):
    type = detect_type(query)
    for prov in PROVINCE_LIST:
        if prov == type:
            return "loc", prov
    if type == "cn":
        return "cn", None
    if type == "special":
        return "special", None
    print(f"type 检测出错，输出为{type}")
    return "cn", None

# 语言识别地区类型
def detect_type(query):
    prompt = (
        """
        请判断以下内容实是在询问有关哪个地区的问题，你的回答只能从下面的选项中挑选：\n\n
        第一种可能
        [
        "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
        "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
        "广东", "广西", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
        "宁夏", "内蒙古", "新疆", "西藏"
        ]
        询问中包含这些省份中的地区，你应当返回对应省市的名字
        例如
        我想在西安建水坝
        宝鸡的桥梁建设要求
        你应该返回 陕西

        澜沧江对外开上水域航道
        你应该返回 云南

        第二种可能
        询问中包含的区域不在这些省份之中（如台湾，香港，澳门，泰国，纽约）则返回 special
        例如
        伦敦的桥梁建设规范
        香港货船进港管理方式
        你应该返回 special

        第三种可能
        询问中包含中国，国家等字样，或未提及具体地区，应返回 cn
        例如
        中国公共水道管理条例

        桥梁建设规划

        中国的水文检测标准
        你应该返回 cn
        """
        f"我的问题是：{query}\n\n"
        "（只回答问题，不要说多余的话）"
    )
    messages = [
        {"role": "system", "content": "你是文件检索机器人，擅长分析语言中描述的对应地区。"},
        {"role": "user",   "content": prompt},
    ]
    return get_completion(messages)

# 调试入口
if __name__ == "__main__":
    query = "内蒙古坡面格宾网箱选型"
    print(answer_question(query))