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

def _fetch_docs(collection, query, top_n, prefix: str):
    """统一检索文档并加前缀"""
    emb = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_n,
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    return [f"{prefix}{d}" for d in docs]

def answer_question(query,chat_history=None):
    std_type, prov = detect_standard_type(query)

    client_db = PersistentClient(path=r"E:\aaacode\ssll\chromadb")
    coll_loc = client_db.get_or_create_collection("coll-loc")   # 地方标准
    coll_cn  = client_db.get_or_create_collection("coll_cn")    # 国家标准

    if std_type == "special":
        return "目前该地区标准尚未收录。"

    # 根据类型决定检索策略
    if std_type == "loc":
        docs_loc = _fetch_docs(coll_loc, query, top_n=6, prefix="【地方标准】")
        docs_cn  = _fetch_docs(coll_cn,  query, top_n=4, prefix="【国家标准】")
        docs_all = docs_loc + docs_cn
    else:
        docs_all = _fetch_docs(coll_cn, query, top_n=10, prefix="【国家标准】")

    context = "\n\n".join(docs_all)
    print(context)
    if std_type == "loc":
        prompt = (
            "请根据以下内容用自然语言回答我的问题\n\n"
            f"{context}\n\n"
            f"我的问题是：{query}\n\n"
            "使用“地方标准”回答问题，之后从“国家标准”中找出相同主题或相同领域的的全国通用性标准信息（不要说多余的话）。" 
            "（回答问题，并说明来源-文件（可以是多个）"
            "（请在回答中用自然语言作答，并在必要时说明来源文件。如果所有内容都与问题无关，则分别回复“地方标准中查无结果”或“国家标准中查无结果”）"
        )
    else:
        prompt = (
            "请根据以下内容用自然语言回答我的问题\n\n"
            f"{context}\n\n"
            f"我的问题是：{query}\n\n"
            "回答问题，不要说多余的话。" 
            "（回答问题，并说明来源-文件（可以是多个）"
            "（请在回答中用自然语言作答，并在必要时说明来源文件。如果所有内容都与问题无关，则分别回复“地方标准中查无结果”或“国家标准中查无结果”）"
        )


    messages = [
        {"role": "system", "content": "你是文件检索机器人，擅长从给出的文本中分析并回答相关问题。"},
        {"role": "user",   "content": prompt},
    ]
    if chat_history:
        messages.extend(chat_history)

    # 当前提问
    messages.append({"role": "user", "content": prompt})

    return get_completion(messages)


# LLM 接口调用
def get_completion(messages, model="deepseek-chat"):
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
    std_type = detect_type(query)
    for prov in PROVINCE_LIST:
        if prov == std_type:
            return "loc", prov
    if std_type == "cn":
        return "cn", None
    if std_type == "special":
        return "special", None
    return "cn", None

def detect_type(query):
    prompt = (
        f"""
        请判断以下内容实是在询问有关哪个地区的问题，你的回答只能从下面的选项中挑选：\n\n
        第一种可能
        {PROVINCE_LIST}
        询问中包含这些省份中的地区，你应当返回对应省市的名字
        例如
        我想在西安建水坝
        宝鸡的桥梁建设要求
        你应该返回 陕西

        澜沧江对外开上水域航道
        你应该返回 云南

        第二种可能
        询问中包含中国，国家等字样，或未提及具体地区，应返回 cn
        例如
        国家公共水道管理条例

        桥梁建设规划

        中国的水文检测标准
        你应该返回 cn

        第三种可能
        询问中包含的区域不在这些省份之中（如台湾，香港，澳门，泰国，纽约），或者乱码之类无法理解的语句则返回 special
        例如
        伦敦的桥梁建设规范
        香港货船进港管理方式
        你应该返回 special        
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