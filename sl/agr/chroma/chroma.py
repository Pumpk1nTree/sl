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

def ingest_folder(folder, collection):
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(folder, filename)
        print(f"▶ 加载 {filename}")
        docs, vectors, ids = load_embedding_json(path)
        collection.add(
            documents=docs,
            embeddings=vectors,
            ids=ids
        )
        print(f"✅ 已写入 {len(docs)} 条记录到 Chroma (文件: {filename})")

if __name__ == "__main__":
    persist_dir = r"E:\aaacode\ssll\chromadb_sub"

    # 只需要这样
    client = PersistentClient(path=persist_dir)

    # 获取集合
    coll_loc = client.get_or_create_collection("coll-loc")
    coll_cn = client.get_or_create_collection("coll_cn")

    ingest_folder(r"E:\aaacode\ssll\embend\loc", coll_loc)
    ingest_folder(r"E:\aaacode\ssll\embend\cn", coll_cn)

    print(f"✅ Chroma 持久化完毕，已保存在 {persist_dir}")