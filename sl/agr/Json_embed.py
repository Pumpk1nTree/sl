import os
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# 加载环境变量（如果你依赖 OpenAI SDK，也可以在这里初始化 client）
load_dotenv()

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
        resp = requests.post(api_url, json={"texts": batch})
        if resp.status_code != 200:
            print(f"⚠️ Embedding API 批次 {i//batch_size} 返回错误 {resp.status_code}：{resp.text}")
            resp.raise_for_status()
        data = resp.json()
        if "embeddings" not in data:
            raise KeyError(f"接口返回缺少 'embeddings' 字段：{data}")
        embeddings.extend(data["embeddings"])
    return embeddings

def load_documents(path):
    """
    支持两种格式：
     1) 整个文件是一个 JSON 数组 -> 直接 json.load
     2) 每行一个 JSON 对象 -> 逐行 json.loads，并跳过空行和解析失败行
    返回列表，每个元素都是一个 dict。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    # 尝试整体解析
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"▶ 以 JSON 数组格式加载，共 {len(data)} 条记录")
            return data
        else:
            print("▶ 整体解析结果不是列表，退回到逐行解析")
    except Exception as e:
        print(f"▶ 整体解析失败，退回到逐行解析：{e}")

    # 逐行解析 JSONL
    docs = []
    with open(path, "r", encoding="utf-8-sig") as fin:
        for lineno, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line or line.startswith('[') or line.startswith(']'):
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️ 跳过第 {lineno} 行（非 JSON 对象）：{repr(line[:30])}… 错误：{e}")
    print(f"▶ 逐行解析完成，共读取 {len(docs)} 条记录")
    return docs

def process_json_file(input_path):
    """
    加载单个 JSON 文件，过滤空文本，生成 embeddings，
    并把 embedding 附加到每个文档 dict 的 "embedding" 字段中。
    返回处理后的列表。
    """
    raw_docs = load_documents(input_path)

    # 过滤并展平 text 字段
    valid_docs = []
    texts = []
    for d in raw_docs:
        txt = d.get("text", "")
        if isinstance(txt, list):
            txt = " ".join(txt)
        if isinstance(txt, str) and txt.strip():
            d["text"] = txt
            valid_docs.append(d)
            texts.append(txt)
    print(f"▶ 过滤后，共 {len(valid_docs)} 条有效文档")

    # 批量生成 embeddings
    vecs = get_embeddings(texts)

    # 合并结果
    for doc, emb in zip(valid_docs, vecs):
        doc["embedding"] = emb
    return valid_docs

def process_folder_loc_json(input_dir, save_dir):
    """
    遍历 input_dir 下所有以省份命名的 JSON 文件，
    对它们进行 embedding 并分别保存到 save_dir，
    文件名为 <省份>_embed.json。
    """
    os.makedirs(save_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".json"):
            continue
        province = os.path.splitext(filename)[0]
        print(f"\n▶ 正在处理省份：{province}")
        in_path = os.path.join(input_dir, filename)
        items = process_json_file(in_path)
        out_path = os.path.join(save_dir, f"{province}.json")
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(items, fout, ensure_ascii=False, indent=2)
        print(f"✅ 已保存 {province} 的 embedding 到：{out_path}")

def process_cn_json(input_path, save_dir):
    """
    对单个 cn.json 进行 embedding，并保存到 save_dir/cn_embed.json。
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n▶ 正在处理 cn.json")
    items = process_json_file(input_path)
    out_path = os.path.join(save_dir, "cn.json")
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(items, fout, ensure_ascii=False, indent=2)
    print(f"✅ 已保存 cn 的 embedding 到：{out_path}")

if __name__ == "__main__":
    # 1) province-level JSONs:
    process_folder_loc_json(
        r"E:\aaacode\ssll\josn\loc",
        r"E:\aaacode\ssll\embend\loc"
    )

    # 2) single cn.json:
    #process_cn_json(
    #    r"E:\aaacode\ssll\josn\cn\cn.json",
    #    r"E:\aaacode\ssll\embend\cn"
    #)