import os
import json
from chromadb import PersistentClient

def load_embedding_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    embeddings = []
    ids = []
    for idx, item in enumerate(data):
        text = item.get("text")
        embedding = item.get("embedding")
        if text and embedding:
            docs.append(text)
            embeddings.append(embedding)
            ids.append(f"{os.path.basename(file_path)}_{idx}")
    return docs, embeddings, ids

def ingest_loc_folder(loc_folder, client):
    for filename in os.listdir(loc_folder):
        if not filename.endswith(".json"):
            continue
        province_name = os.path.splitext(filename)[0]  # 以文件名作为省份名（不含.json）
        path = os.path.join(loc_folder, filename)
        print(f"▶ 加载地方标准：{province_name}")
        
        # 为每个省份创建独立的 collection
        collection = client.get_or_create_collection(name=f"loc_{province_name}")
        
        docs, vectors, ids = load_embedding_json(path)
        collection.add(
            documents=docs,
            embeddings=vectors,
            ids=ids
        )
        print(f"✅ 写入 {len(docs)} 条记录到 collection [loc_{province_name}]")

def ingest_cn_folder(cn_folder, collection):
    for filename in os.listdir(cn_folder):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(cn_folder, filename)
        print(f"▶ 加载国家标准：{filename}")
        docs, vectors, ids = load_embedding_json(path)
        collection.add(
            documents=docs,
            embeddings=vectors,
            ids=ids
        )
        print(f"✅ 写入 {len(docs)} 条记录到国家标准集合 (文件: {filename})")

if __name__ == "__main__":
    persist_dir = r"E:\aaacode\ssll\chromadb_sub"
    client = PersistentClient(path=persist_dir)

    # 统一国家标准集合
    coll_cn = client.get_or_create_collection("coll_cn")

    # 地方标准：根据省份分别创建集合
    ingest_loc_folder(r"E:\aaacode\ssll\embend\loc", client)

    # 国家标准：仍统一写入一个集合
    ingest_cn_folder(r"E:\aaacode\ssll\embend\cn", coll_cn)

    print(f"✅ 所有标准已完成持久化，路径：{persist_dir}")