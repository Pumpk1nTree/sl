# ocr_pdf_to_paragraph_simple.py

import os
import io
import json
import fitz                # pip install pymupdf
import numpy as np
from PIL import Image      # pip install pillow
import cv2                 # pip install opencv-python
from paddleocr import PaddleOCR  # pip install paddleocr

# ——— 配置路径 ———
INPUT_DIR  = r"E:\aaacode\OCR\PDF"
OUTPUT_DIR = r"E:\aaacode\OCR\emb_paragraph_simple"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ——— 初始化 PaddleOCR ———
print("▶ 初始化 PaddleOCR …")
ocr = PaddleOCR(use_textline_orientation=True, lang="ch")
print("    ✓ 模型加载完毕\n")

def ocr_page_to_paragraph(img: np.ndarray) -> str:
    """
    对一张 BGR numpy 图像跑 OCR，并把同一页的所有识别文字按顺序拼成一段。
    """
    raw = ocr.predict(img, use_textline_orientation=True)

    # 如果返回新版“页面字典”格式
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        page = raw[0]
        texts = page.get("rec_texts", [])
        # 直接按顺序拼接
        return " ".join(texts)

    # 否则按旧版扁平/嵌套列表解析
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
            # 嵌套：[[box, (txt,score)], ...] 或 [[box, txt, score], ...]
            for sub in item:
                if len(sub) == 2 and isinstance(sub[1], (list, tuple)):
                    _, (txt, _) = sub
                    texts.append(txt)
                elif len(sub) == 3 and isinstance(sub[1], str):
                    _, txt, _ = sub
                    texts.append(txt)
    return " ".join(texts)

def process_pdf(pdf_path: str, json_path: str):
    """
    将 PDF 每一页渲染成图片，OCR 识别，
    并把同一页的所有文字合并成一段。
    最终输出 JSON:
    {
      "file": "...",
      "pages": [
        { "page_number": 0, "text": "…" },
        …
      ]
    }
    """
    doc = fitz.open(pdf_path)
    result = {"file": os.path.basename(pdf_path), "pages": []}

    for i, page in enumerate(doc):
        print(f"▶ 处理 {os.path.basename(pdf_path)} 第 {i} 页")
        # 渲染成 PNG bytes
        pix = page.get_pixmap(dpi=300, alpha=False)
        img_data = pix.tobytes("png")
        # PIL -> OpenCV BGR
        pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        rgb = np.array(pil)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        paragraph = ocr_page_to_paragraph(img)
        result["pages"].append({
            "page_number": i,
            "text": paragraph
        })
        print(f"    ✓ 本页文字长度 {len(paragraph)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"▶ 完成：已写入 {json_path}\n")

if __name__ == "__main__":
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path  = os.path.join(INPUT_DIR, fname)
        json_name = os.path.splitext(fname)[0] + ".json"
        json_path = os.path.join(OUTPUT_DIR, json_name)

        print(f"[OCR] {fname} → {json_name}")
        try:
            process_pdf(pdf_path, json_path)
        except Exception as e:
            print("  ⚠️  处理失败:", e)