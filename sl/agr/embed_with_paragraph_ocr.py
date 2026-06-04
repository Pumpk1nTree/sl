# embed_with_paragraph_ocr.py

import os
import io
import json
import fitz                # pip install pymupdf
import numpy as np
from PIL import Image      # pip install pillow
import cv2                 # pip install opencv-python
import requests
import tempfile
from paddleocr import PaddleOCR  # pip install paddleocr
from tqdm import tqdm

# ——— 1. 初始化 PaddleOCR ———
print("▶ 初始化 PaddleOCR …")
ocr = PaddleOCR(use_textline_orientation=True, lang="ch")
print("    ✓ 模型加载完毕\n")

def ocr_page_to_paragraph(img: np.ndarray) -> str:
    """
    对一张 BGR numpy 图像跑 OCR，并把同一页的所有识别文字按顺序拼成一段。
    """
    raw = ocr.predict(img, use_textline_orientation=True)

    # 如果是新版“页面字典”格式
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        page = raw[0]
        texts = page.get("rec_texts", [])
        return " ".join(texts)

    # 否则兼容旧版结构
    texts = []
    for item in raw:
        # 扁平：[box, (txt,score)]
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], (list, tuple)):
            _, (txt, _) = item
            texts.append(txt)
        # 扁平：[box, txt, score]
        elif isinstance(item, list) and len(item) == 3 and isinstance(item[1], str):
            _, txt, _ = item
            texts.append(txt)
        else:
            # 嵌套多行模式
            for sub in item:
                if len(sub) == 2 and isinstance(sub[1], (list, tuple)):
                    _, (txt, _) = sub
                    texts.append(txt)
                elif len(sub) == 3 and isinstance(sub[1], str):
                    _, txt, _ = sub
                    texts.append(txt)
    return " ".join(texts)


# ——— 2. 嵌入 API 调用函数 ———
EMBEDDING_API = "http://localhost:8000/embeddings"

def get_embeddings(texts, api_url=EMBEDDING_API):
    payload = {"texts": texts}
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"]


# ——— 3. 处理单个 PDF 并返回每页的文字与向量 ———
def process_pdf(pdf_path, index=None, total=None):
    if index is not None and total is not None:
        print(f"[{index}/{total}] 处理 PDF: {os.path.basename(pdf_path)}")

    doc = fitz.open(pdf_path)
    page_texts = []

    for page_num in range(len(doc)):
        # 渲染成 RGBA 图片
        pix = doc.load_page(page_num).get_pixmap(dpi=300, alpha=False)
        img_data = pix.tobytes("png")

        # PIL -> OpenCV BGR
        pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 调用自定义 OCR
        paragraph = ocr_page_to_paragraph(bgr)
        page_texts.append(paragraph)

        print(f"    ✓ 页 {page_num+1} 文本长度: {len(paragraph)}")

    # 批量获取向量
    embeddings = get_embeddings(page_texts)
    print(f"[{index}/{total}] 完成：{os.path.basename(pdf_path)}")

    return [
        {"page": i+1, "text": txt, "embedding": emb}
        for i, (txt, emb) in enumerate(zip(page_texts, embeddings))
    ]


# ——— 4. 批量处理函数示例 ———
def process_folder_loc(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    provinces = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for p_idx, province in enumerate(provinces, start=1):
        prov_path = os.path.join(input_dir, province)
        print(f"开始省份 {p_idx}/{len(provinces)}: {province}")
        embeds = []
        pdfs = [f for f in os.listdir(prov_path) if f.lower().endswith('.pdf')]
        total_pdfs = len(pdfs)
        for idx, pdf in enumerate(pdfs, start=1):
            pdf_path = os.path.join(prov_path, pdf)
            embeds.extend(process_pdf(pdf_path, idx, total_pdfs))
        save_path = os.path.join(output_dir, f"{province}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(embeds, f, ensure_ascii=False, indent=2)
        print(f"省份 {province} 完成，保存至 {save_path}\n")

def process_folder_cn(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    total = len(pdfs)
    print(f"开始处理国家标准，共 {total} 个 PDF")
    for idx, pdf in enumerate(pdfs, start=1):
        pdf_path = os.path.join(input_dir, pdf)
        embeds = process_pdf(pdf_path, idx, total)
        base = os.path.splitext(pdf)[0]
        out_path = os.path.join(output_dir, f"{base}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(embeds, f, ensure_ascii=False, indent=2)
        print(f"[{idx}/{total}] {pdf} → {out_path}")
    print(f"国家标准处理完成，结果保存在 {output_dir}")





if __name__ == "__main__":
    # 示例：按省份分文件夹处理
    process_folder_loc(
        r"E:\aaacode\ssll\data\PDF\loc_spearate",
        r"E:\aaacode\ssll\embend\loc"
    )
    # 示例：所有国家标准同一目录处理
    process_folder_cn(
        r"E:\aaacode\ssll\data\PDF\cn",
        r"E:\aaacode\ssll\embend\cn"
    )