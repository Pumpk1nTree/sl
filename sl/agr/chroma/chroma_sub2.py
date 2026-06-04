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

def ingest_folder(base_folder, source, collection):
    """
    将 base_folder 中的 JSON 文件批量导入到同一个 collection，
    metadata 中记录来源：province（地方标准为省名，国家标准固定为 'cn'）。
    """
    for filename in os.listdir(base_folder):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(base_folder, filename)
        print(f"▶ 加载 {source} 标准：{filename}")

        docs, vectors, ids = load_embedding_json(path)
        # 构造 metadata
        if source == "loc":
            province_name = os.path.splitext(filename)[0]
            metadatas = [{"province": province_name} for _ in docs]
        else:
            # 国家标准
            metadatas = [{"province": "cn"} for _ in docs]

        collection.add(
            documents=docs,
            embeddings=vectors,
            ids=ids,
            metadatas=metadatas
        )
        print(f"✅ 写入 {len(docs)} 条记录到集合 [{collection.name}]，来源：{source}")

if __name__ == "__main__":
    persist_dir = r"E:\aaacode\ssll\chromadb_all"
    client = PersistentClient(path=persist_dir)

    # 合并管理：单一 collection
    coll_all = client.get_or_create_collection("all_standards")

    # 批量导入地方标准
    ingest_folder(r"E:\aaacode\ssll\embend\loc", source="loc", collection=coll_all)

    # 批量导入国家标准
    ingest_folder(r"E:\aaacode\ssll\embend\cn", source="cn", collection=coll_all)

    print(f"✅ 所有标准已完成持久化，路径：{persist_dir}")