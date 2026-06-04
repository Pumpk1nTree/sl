import os
import re
import io
import shutil

import fitz                         # pip install pymupdf
from PIL import Image              # pip install pillow
import numpy as np
import cv2                          # pip install opencv-python
from paddleocr import PaddleOCR    # pip install paddleocr

# ——— 配置 ———
INPUT_DIR  = r"E:\aaacode\ssll\data\PDF\cn"
OUTPUT_DIR = r"E:\aaacode\ssll\data\PDF\recn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ——— 初始化 PaddleOCR ———
print("▶ 初始化 PaddleOCR …")
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="ch"
)
print("    ✓ 模型加载完毕\n")

def ocr_page_to_lines(img: np.ndarray) -> list[str]:
    """
    对一张 BGR numpy 图像跑 OCR，并把同一页的识别结果按行返回列表。
    """
    raw = ocr.predict(img, use_textline_orientation=True)

    # 新版：页面字典格式
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0].get("rec_texts", [])

    lines = []
    for item in raw:
        # 扁平：[box, (txt, score)]
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], (list, tuple)):
            _, (txt, _) = item; lines.append(txt)
        # 扁平：[box, txt, score]
        elif isinstance(item, list) and len(item) == 3 and isinstance(item[1], str):
            _, txt, _ = item; lines.append(txt)
        else:
            # 嵌套多行
            for sub in item:
                if len(sub) == 2 and isinstance(sub[1], (list, tuple)):
                    _, (txt, _) = sub; lines.append(txt)
                elif len(sub) == 3 and isinstance(sub[1], str):
                    _, txt, _ = sub; lines.append(txt)
    return lines

def extract_first_page_lines(pdf_path: str) -> list[str]:
    """
    渲染 PDF 第 1 页为图像，手动缩放最长边 ≤4000 px，然后跑 OCR 返回行列表。
    """

    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=300, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    w, h = img.size
    max_side = max(w, h)
    if max_side > 4000:
        scale = 4000 / max_side
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return ocr_page_to_lines(bgr)

# ——— 编号正则 ———
# 1) 从文件名提取：允许 GB+1234.56-2014、SL_31-2003、DL T 5514—2015 等
filecode_pattern = re.compile(
    r'\b([A-Z]+(?:/[A-Z])?)[\s+_\-]?(\d+(?:\.\d+)*[-–—]\d+)\b',
    re.IGNORECASE
)
# 2) OCR 识别提取
ocr_pattern = re.compile(
    r'\b'
    r'[A-Z]+'
    r'(?:/[A-Z])?'
    r'\s*'
    r'\d+(?:\.\d+)*'
    r'\s*'
    r'[-–—]'
    r'\s*'
    r'\d+'
    r'\b'
)

def normalize_code(prefix: str, numeral: str) -> str:
    """
    合并前缀和编号、去空格并大写
    """
    return (prefix + numeral).upper().replace(" ", "")

if __name__ == "__main__":
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue

        src = os.path.join(INPUT_DIR, fname)
        code = None

        # —— 1. 文件名优先 —— 
        m0 = filecode_pattern.search(fname)
        if m0:
            code = normalize_code(m0.group(1), m0.group(2))
            print(f"[Filename] {fname} → {code}.pdf")
        else:
            # —— 2. OCR 回退 —— 
            try:
                lines = extract_first_page_lines(src)
                print(f"[OCR] {fname} 第1页 行数: {len(lines)}")
                text = " ".join(lines)
                m1 = ocr_pattern.search(text)
                if m1:
                    code = m1.group().replace(" ", "")
                    print(f"[OCR] {fname} → {code}.pdf")
                else:
                    print(f"[Skip] {fname}：未识别到编号，跳过")
            except Exception as e:
                print(f"[Error] 处理 {fname} 时出错：{e}")

        # —— 3. 复制并重命名 —— 
        if code:
            dst = os.path.join(OUTPUT_DIR, f"{code}.pdf")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"[Error] 复制 {fname} → {dst} 失败：{e}")

    print("🎉 批量重命名完成。")