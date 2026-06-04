import os
import requests
import time
import logging
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import PersistentClient

# —— OpenAI 配置 —— #
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# —— 本地向量生成函数 —— #
def get_embeddings(texts, api_url="http://localhost:8000/embeddings", batch_size=32):
    embeddings = []
    filtered = [t for t in texts if isinstance(t, str) and t.strip()]
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i : i + batch_size]
        resp = requests.post(api_url, json={"texts": batch})
        if resp.status_code != 200:
            print(f"⚠️ Embedding API 批次 {i//batch_size} 返回错误 {resp.status_code}：{resp.text}")
            resp.raise_for_status()
        data = resp.json()
        if "embeddings" not in data:
            raise KeyError(f"接口返回缺少 'embeddings' 字段：{data}")
        embeddings.extend(data["embeddings"])
    return embeddings

# —— 批量检索函数（支持元数据过滤） —— #
def _fetch_docs(collection, query, top_n, prefix: str, where: dict = None):
    emb = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_n,
        where=where or {},
        include=["documents"],
    )
    docs = results["documents"][0] if results.get("documents") else []
    return [f"{prefix}{d}" for d in docs]


def _fetch_docs_with_distance(collection, query, top_n, where: dict = None):
    emb = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_n,
        where=where or {},
        include=["documents", "distances"]
    )
    docs = results["documents"][0] if results.get("documents") else []
    dists = results["distances"][0] if results.get("distances") else []
    return docs, dists

# —— Chroma 客户端与集合 —— #
PERSIST_DIR = r"E:\aaacode\ssll\chromadb_all"  # 与数据导入路径保持一致
client_db = PersistentClient(path=PERSIST_DIR)
ALL_COLL = client_db.get_or_create_collection("all_standards")

# —— 省份中文→拼音映射 —— #
PROVINCE_PY_MAP = {
    "北京": "beijing", "天津": "tianjin", "上海": "shanghai", "重庆": "chongqing",
    "河北": "hebei", "山西": "shanxi", "辽宁": "liaoning", "吉林": "jilin", "黑龙江": "heilongjiang",
    "江苏": "jiangsu", "浙江": "zhejiang", "安徽": "anhui", "福建": "fujian", "江西": "jiangxi",
    "山东": "shandong", "河南": "henan", "湖北": "hubei", "湖南": "hunan", "广东": "guangdong",
    "广西": "guangxi", "海南": "hainan", "四川": "sichuan", "贵州": "guizhou", "云南": "yunnan",
    "陕西": "shaanxi", "甘肃": "gansu", "青海": "qinghai", "宁夏": "ningxia", "内蒙古": "neimenggu",
    "新疆": "xinjiang", "西藏": "xizang"
}
PROVINCE_LIST = list(PROVINCE_PY_MAP.keys())


def answer_question(query, selected_provinces=None, chat_history=None):
    # 1) 检测用户问的是“地方”还是“国家”还是混合
    std_type, prov_zh = detect_standard_type(query, selected_provinces)
    docs_all = []

    # —— 地方 + 国家 模式 —— 
    if std_type in ("loc", "mixed"):
        # 支持用户 UI 传入多个省份，或者 detect 出一个省份
        prov_list = selected_provinces or [prov_zh]

        # 先搜地方标准
        local_cands = []
        for p in prov_list:
            py = PROVINCE_PY_MAP.get(p)
            if not py:
                # 如果没映射到拼音，跳过或报警
                st.warning(f"无法找到 {p} 对应的拼音，跳过地方检索")
                continue

            # 从同一个 all_standards 集合里按 metadata 过滤
            docs, dists = _fetch_docs_with_distance(
                ALL_COLL,
                query,
                top_n=6,
                where={"province": py}
            )
            for doc, dist in zip(docs, dists):
                local_cands.append({"province": p, "doc": doc, "dist": dist})

        # 按距离排序，取最靠前的 6 条
        local_cands.sort(key=lambda x: x["dist"])
        for item in local_cands[:6]:
            docs_all.append(f"【地方标准·{item['province']}】{item['doc']}")

        # 然后搜国家标准（province="cn"）
        cn_docs = _fetch_docs(
            ALL_COLL,
            query,
            top_n=4,
            prefix="【国家标准】",
            where={"province": "cn"}
        )
        docs_all.extend(cn_docs)

    # —— 纯国家标准 模式 —— 
    elif std_type == "cn":
        docs_all = _fetch_docs(
            ALL_COLL,
            query,
            top_n=10,
            prefix="【国家标准】",
            where={"province": "cn"}
        )

    context = "\n\n".join(docs_all)
    print(context)

    if selected_provinces or std_type == "loc":
        prompt = (
            "请根据以下内容用自然语言回答我的问题：\n\n"
            f"{context}\n\n"
            f"我的问题是：{query}\n\n"
            "先基于“地方标准”回答，再补充“国家标准”中的相关内容；说明来源文件；若无相关内容，则分别回复“地方标准中查无结果”或“国家标准中查无结果”。"
        )
    else:
        prompt = (
            "请根据以下内容用自然语言回答我的问题：\n\n"
            f"{context}\n\n"
            f"我的问题是：{query}\n\n"
            "仅基于“国家标准”回答；说明来源文件；若无相关内容，则回复“国家标准中查无结果”。"
        )

    messages = [
        {"role": "system",  "content": "你是文件检索机器人，擅长从给出的文本中准确回答问题。"},
        {"role": "user",    "content": prompt},
    ]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    return get_completion(messages)


def get_completion(messages, model="deepseek-chat"):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content


def detect_standard_type(query, selected_provinces):
    std_type = detect_type(query)
    if std_type in PROVINCE_PY_MAP:
        return "loc", std_type
    if std_type == "cn":
        return "cn", None
    if std_type == "noo":
        return ("loc", std_type) if selected_provinces else ("cn", None)
    return "special", None


def detect_type(query):
    prompt = (
        f"请判断以下问题是询问哪个地区，回答只能是：\n"
        f"1. 如果询问{PROVINCE_LIST} 中的省市名，输出对应省市名\n"
        "2. 如果询问与国家相关的话题，输出cn\n"
        "3. 如果询问中包含的区域不在以上条件之中或无法读懂，输出special\n"
        "4. 如果询问中未提及地点，输出noo\n"
        f"问题：{query}\n\n"
        "只输出对应标识或省市名。"
    )
    messages = [
        {"role": "system", "content": "你是擅长识别地名的助手。"},
        {"role": "user",   "content": prompt},
    ]
    return get_completion(messages).strip()

# —— 调试入口 —— #
if __name__ == "__main__":
    print(answer_question("内蒙古坡面格宾网箱选型", selected_provinces=["山东", "陕西"]))
