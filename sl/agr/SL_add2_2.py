# SL_add2.py
import os
import requests
import time
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

def _fetch_docs(collection, query, top_n, prefix: str):
    emb = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_n,
        include=["documents"],
    )
    docs = results["documents"][0] if results["documents"] else []
    return [f"{prefix}{d}" for d in docs]

# —— Chroma 客户端与集合 —— #
PERSIST_DIR = r"E:\aaacode\ssll\chromadb"
client_db = PersistentClient(path=PERSIST_DIR)
COLL_CN = client_db.get_or_create_collection("coll_cn")

# —— 省份中文→拼音映射 —— #
PROVINCE_PY_MAP = {
    "北京": "beijing", "天津": "tianjin", "上海": "shanghai", "重庆": "chongqing",
    "河北": "hebei", "山西": "shanxi", "辽宁": "liaoning", "吉林": "jilin", "黑龙江": "heilongjiang",
    "江苏": "jiangsu", "浙江": "zhejiang", "安徽": "anhui", "福建": "fujian", "江西": "jiangxi",
    "山东": "shandong", "河南": "henan", "湖北": "hubei", "湖南": "hunan", "广东": "guangdong",
    "广西": "guangxi", "海南": "hainan", "四川": "sichuan", "贵州": "guizhou", "云南": "yunnan",
    "陕西": "shaanxi", "甘肃": "gansu", "青海": "qinghai", "宁夏": "ningxia", "内蒙古": "neimenggu",
    "新疆": "xinjiang", "西藏": "xizang","loc":""
}
PROVINCE_LIST = list(PROVINCE_PY_MAP.keys())

def _fetch_docs_with_distance(collection, query, top_n):
    """
    返回 (docs, distances)，其中 docs 是纯文本列表，distances 是相似度距离列表
    """
    emb = get_embeddings([query])[0]
    results = collection.query(
        query_embeddings=[emb],
        n_results=top_n,
        include=["documents", "distances"]
    )
    docs = results["documents"][0] if results["documents"] else []
    dists = results["distances"][0] if results["distances"] else []
    return docs, dists


def answer_question(query, selected_provinces=None, chat_history=None):
    """
    query: 用户问题
    selected_provinces: UI 传入的中文省份列表
    chat_history: 历史对话（可选）
    """
    std_type, prov = detect_standard_type(query, selected_provinces)

    # special 类型且未选省份，直接返回
    if std_type == "special" and not selected_provinces:
        return "目前该地区标准尚未收录。"

    docs_all = []

    # —— 1. 地方 + 国家 模式 —— 
    if selected_provinces or std_type == "loc":
        # 省份列表（中文）
        if selected_provinces:
            prov_list = selected_provinces
        else:
            prov_list = [prov]

        # 先每省各取 up to 6 条做候选
        local_candidates = []
        for p in prov_list:
            py = PROVINCE_PY_MAP[p]
            coll_loc = client_db.get_or_create_collection(f"loc_{py}")
            docs, dists = _fetch_docs_with_distance(coll_loc, query, top_n=6)
            for doc, dist in zip(docs, dists):
                local_candidates.append({
                    "province": p,
                    "doc": doc,
                    "dist": dist
                })

        # 合并所有省份候选，按距离升序（越小越相关）排序
        local_candidates.sort(key=lambda x: x["dist"])
        # 取最前面的 6 条
        top_local = local_candidates[:6]
        # 添加前缀并放入最终列表
        for item in top_local:
            docs_all.append(f"【地方标准·{item['province']}】{item['doc']}")

        # 再检索 4 条国家标准
        docs_cn = _fetch_docs(
            COLL_CN,
            query,
            top_n=4,
            prefix="【国家标准】"
        )
        docs_all.extend(docs_cn)

    # —— 2. 纯国家 标准 模式 —— 
    elif std_type == "cn":
        docs_all = _fetch_docs(
            COLL_CN,
            query,
            top_n=10,
            prefix="【国家标准】"
        )

    # … 下半部分保持不变，构建 prompt 并调用 LLM …

    context = "\n\n".join(docs_all)
    print(context)

    if selected_provinces or std_type == "loc":
        prompt = (
            "请根据以下内容用自然语言回答我的问题：\n\n"
            f"{context}\n\n"
            f"我的问题是：{query}\n\n"
            "先基于“地方标准”回答，再补充“国家标准”中的相关内容；"
            "说明来源文件；若无相关内容，则分别回复“地方标准中查无结果”或“国家标准中查无结果”。"
        )
    else:
        prompt = (
            "请根据以下内容用自然语言回答我的问题：\n\n"
            f"{context}\n\n"
            f"我的问题是：{query}\n\n"
            "仅基于“国家标准”回答；说明来源文件；"
            "若无相关内容，则回复“国家标准中查无结果”。"
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

def detect_standard_type(query, selected_provinces=None):
    std_type = detect_type(query)
    if std_type in PROVINCE_PY_MAP:
        return "loc", std_type
    if std_type == "cn":
        return "cn", None
    if std_type == "noo":
        if selected_provinces:
            return "cn", None
        return "loc", std_type
    return "special", None

def detect_type(query):
    prompt = (
        f"请判断以下问题是询问哪个地区，回答只能是：\n"
        f"1. 如果询问{PROVINCE_LIST} 中的省市名，输出对省市名\n"
        "2. 如果询问与国家相关的话题，输出cn\n"
        "3. 如果询问询问中包含的区域不在以上的条件之中或者是无法读懂的乱码，输出special\n"
        "4. 如果讯问中未提及具体的地点，输出noo\n"
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