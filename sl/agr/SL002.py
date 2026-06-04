import os
import json
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
from numpy.linalg import norm
from numpy import dot

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def l2(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def get_embeddings(texts, api_url="http://localhost:8000/embeddings", batch_size=32):
    """
    分批向本地 embeddings 服务请求，避免一次性 payload 太大导致 422。
    跳过空文本，并打印出 HTTP 错误详情。
    """
    embeddings = []
    # 过滤掉空字符串
    filtered = [t for t in texts if isinstance(t, str) and t.strip()]
    for i in range(0, len(filtered), batch_size):
        batch = filtered[i : i + batch_size]
        payload = {"texts": batch}
        resp = requests.post(api_url, json=payload)
        if resp.status_code != 200:
            # 打印出完整的错误信息，帮助定位问题
            print(f"⚠️ Embedding API 批次 {i//batch_size} 返回错误 {resp.status_code}：{resp.text}")
            resp.raise_for_status()
        data = resp.json()
        if "embeddings" not in data:
            raise KeyError(f"接口返回缺少 'embeddings' 字段：{data}")
        embeddings.extend(data["embeddings"])
    return embeddings



def search_top_n(query_vec, doc_vecs, documents, n=3):
    """
    返回 top-n 最近邻文档及其距离
    """
    dists = [(i, cos_sim(query_vec, vec)) for i, vec in enumerate(doc_vecs)]
    dists.sort(key=lambda x: -x[1])

    #dists = [(i, l2(query_vec, vec)) for i, vec in enumerate(doc_vecs)]
    #dists.sort(key=lambda x: x[1])
    top = dists[:n]
    return {
        "documents": [documents[i] for i, _ in top],
        "indices": [i for i, _ in top],
        "scores": [s for _, s in top]
    }

def load_documents(path):
    """
    支持两种格式：
     1) 整个文件是一个 JSON 数组 -> 直接 json.load
     2) 每行一个 JSON 对象 -> 逐行 json.loads，并跳过空行和解析失败行
    返回列表，每个元素都是一个 dict（如 {'page': ..., 'text': ...}）。
    """
    import json, os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    # 先尝试整体解析
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"▶ 以 JSON 数组格式加载，共 {len(data)} 条记录")
            return data
        else:
            # 读到的不是 list（例如一个 dict），继续走后面的逐行逻辑
            print("▶ 整体解析结果不是列表，退回到逐行解析")
    except Exception as e:
        print(f"▶ 整体解析失败，退回到逐行解析：{e}")

    # 逐行解析，兼容 JSONL
    docs = []
    with open(path, "r", encoding="utf-8-sig") as fin:
        for lineno, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            # 如果行以 '[' 或 ']' 开头，也跳过
            if line.startswith('[') or line.startswith(']'):
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️ 跳过第 {lineno} 行（非 JSON 对象）：{repr(line[:30])}… 错误：{e}")
    print(f"▶ 逐行解析完成，共读取 {len(docs)} 条记录")
    return docs


def answer_question_impl(query, doc_path, top_n):
    """
    # 1. 读取文档
    documents = load_documents(doc_path)

    # —— 新增：把 text 字段从 list[str] 展平成一个长字符串 —— 
    for d in documents:
        txt = d.get("text", "")
        if isinstance(txt, list):
            # 用空格合并，也可以按你需求换成 "\n".join(txt)
            d["text"] = " ".join(txt)

    # 2. 提取所有文本
    texts = [d["text"] for d in documents]

    # 3. 批量生成文档 embeddings
    doc_vecs = get_embeddings(texts)

    # 4. 生成 query embedding
    query_vec = get_embeddings([query])[0]

    # 5. 检索 top_n
    results = search_top_n(query_vec, doc_vecs, documents, n=top_n)
    context_texts = "|".join(d["text"] for d in results["documents"])
    """

    raw_docs = load_documents(doc_path)

    # 2. 过滤掉 text 为空或全空白的文档，同时构造有效文档列表和对应的文本列表
    valid_docs = []
    valid_texts = []
    for d in raw_docs:
        txt = d.get("text", "")
        # 如果 text 是列表，先展平
        if isinstance(txt, list):
            txt = " ".join(txt)
        if isinstance(txt, str) and txt.strip():
            d["text"] = txt
            valid_docs.append(d)
            valid_texts.append(txt)
    print(f"▶ 过滤后，共 {len(valid_docs)} 条有效文档（去除空文本）")

    # 3. 对有效文本批量生成 embeddings
    doc_vecs = get_embeddings(valid_texts)

    # 4. 生成 query 的 embedding
    query_vec = get_embeddings([query])[0]

    # 5. 在有效文档中检索 top_n
    results = search_top_n(query_vec, doc_vecs, valid_docs, n=top_n)

    # 6. 构造 prompt 等后续逻辑……
    #    这里的 results["documents"] 就和 results["scores"] 一一对应
    #    无需担心索引错位
    context_texts = "\n\n".join(d["text"] for d in results["documents"])

    # 6. 拼 prompt
    prompt_header = (
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
    )
    prompt_body = "\n\n".join(context_texts)
    prompt_tail = f"\n\n我的问题是(直接回答问题，不要解释来源）：{query}"

    messages = [
        {"role": "system", "content": "你是文件整理机器人，善于从标准文件内容中找出详细答案。"},
        {"role": "user", "content": prompt_header + prompt_body + prompt_tail},
    ]
    return get_completion(messages)

def answer_question(query):
    std_type, prov = detect_standard_type(query)
    if std_type == "special":
        return f"目前{prov}的地方标准尚未收录。"
    elif std_type == "loc":
        # 地方标准优先检索当地，再补充国家标准
        loc_path = rf"E:\aaacode\ssll\josn\loc\{prov}.json"
        cn_path  = r"E:\aaacode\ssll\josn\cn\cn.json"
        # 先取 3 条省级，再取 2 条国家级
        ctx_loc = answer_question_impl(query, loc_path, top_n=6)
        ctx_cn  = answer_question_impl(query, cn_path,  top_n=4)
        return ctx_loc + "\n\n" + ctx_cn
    else:
        # 只检索国家标准
        cn_path = r"E:\aaacode\ssll\josn\cn\cn.json"
        return answer_question_impl(query, cn_path, top_n=10)

def get_completion(messages, model="deepseek/deepseek-chat-v3-0324:free"):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content

# 省份列表 & 特殊区域检测
PROVINCE_LIST = [
    "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "广西", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "宁夏", "内蒙古", "新疆", "西藏"
]
SPECIAL_REGION = ["香港", "澳门","台湾"]

def detect_standard_type(query):
    for reg in SPECIAL_REGION:
        if reg in query:
            return "special", reg
    for prov in PROVINCE_LIST:
        if prov in query:
            return "loc", prov
    return "cn", None

if __name__ == "__main__":
    query = "内蒙古坡面格宾网箱选型"
    print(answer_question(query))