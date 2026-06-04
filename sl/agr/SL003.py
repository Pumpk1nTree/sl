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
    且要求每条记录里已经有 "embedding" 字段，返回列表，每个元素都是一个 dict。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    # 先尝试整体解析
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            docs = data
            print(f"▶ 以 JSON 数组格式加载，共 {len(docs)} 条记录")
        else:
            raise ValueError("整体解析结果不是列表")
    except Exception as e:
        print(f"▶ 整体解析失败，退回到逐行解析：{e}")
        docs = []
        with open(path, "r", encoding="utf-8-sig") as fin:
            for lineno, raw in enumerate(fin, start=1):
                line = raw.strip()
                if not line or line.startswith('[') or line.startswith(']'):
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError as err:
                    print(f"  ⚠️ 跳过第 {lineno} 行：{err}")
        print(f"▶ 逐行解析完成，共读取 {len(docs)} 条记录")

    # 验证 embedding 字段
    for idx, doc in enumerate(docs, start=1):
        if "embedding" not in doc:
            raise KeyError(f"第 {idx} 条记录缺少 'embedding' 字段")

    return docs


def answer_question_impl(query, doc_path, top_n):
    """
    1. load_documents 会直接加载包含 embedding 的文档列表；
    2. 只需对 query 做一次 embedding，然后用 search_top_n 检索即可。
    """
    # 1. 读入所有文档（每条 doc 已有 'text' 和 'embedding'）
    raw_docs = load_documents(doc_path)

    # 2. 过滤空文本 & 展平 text 列表
    valid_docs = []
    valid_embs = []
    for d in raw_docs:
        txt = d.get("text", "")
        if isinstance(txt, list):
            txt = " ".join(txt)
        if not isinstance(txt, str) or not txt.strip():
            continue
        d["text"] = txt
        valid_docs.append(d)
        valid_embs.append(d["embedding"])
    print(f"▶ 过滤后，共 {len(valid_docs)} 条有效文档")

    # 3. 对 query 生成 embedding
    query_vec = get_embeddings([query])[0]

    # 4. 检索 top_n
    results = search_top_n(query_vec, valid_embs, valid_docs, n=top_n)

    # 5. 构造上下文 & 调用 LLM
    context = "\n\n".join(d["text"] for d in results["documents"])
    print(context)
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


def answer_question(query):
    std_type, prov = detect_standard_type(query)
    if std_type == "special":
        return f"目前该的地方标准尚未收录。"
    elif std_type == "loc":
        # 地方标准优先检索当地，再补充国家标准
        loc_path = rf"E:\aaacode\ssll\embend\loc\{prov}.json"
        cn_path  = r"E:\aaacode\ssll\embend\cn\cn.json"
        # 先取 3 条省级，再取 2 条国家级
        ctx_loc = answer_question_impl(query, loc_path, top_n=6)
        ctx_cn  = answer_question_impl(query, cn_path,  top_n=4)
        return ctx_loc + "\n\n" + ctx_cn
    else:
        # 只检索国家标准
        cn_path = r"E:\aaacode\ssll\embend\cn\cn.json"
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

if __name__ == "__main__":
    query = "内蒙古坡面格宾网箱选型"
    print(answer_question(query))