from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
# 加载模型到 GPU
model = SentenceTransformer("BAAI/bge-large-zh-v1.5", device="cuda")

class EmbeddingRequest(BaseModel):
    texts: list[str]

@app.post("/embeddings")
def get_embeddings(req: EmbeddingRequest):
    vectors = model.encode(req.texts, normalize_embeddings=True)
    return {"embeddings": vectors.tolist()}
#cd \aaacode\ssll\agr
#uvicorn embedding_api:app --host 0.0.0.0 --port 8000
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
