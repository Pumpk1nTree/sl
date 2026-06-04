import os
import re
import io
import shutil

import fitz                          # pip install pymupdf
import numpy as np
from PIL import Image               # pip install pillow
import cv2                           # pip install opencv-python
from paddleocr import PaddleOCR     # pip install paddleocr

# ——— 配置 ———
INPUT_DIR  = r"E:\aaacode\ssll\data\PDF\cn"
OUTPUT_DIR = r"E:\aaacode\ssll\data\PDF\recn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ——— 初始化 PaddleOCR ———
print("▶ 初始化 PaddleOCR …")
ocr = PaddleOCR(
    use_textline_orientation=True,
    lang="ch",
    det_limit_side_len=9000   # 兼容旧版参数，提升内部检测阈值
)
print("    ✓ 模型加载完毕\n")

def ocr_page_to_lines(img: np.ndarray) -> list[str]:
    """
    对一张 BGR numpy 图像跑 OCR，并把同一页的识别结果按行返回列表。
    """
    raw = ocr.predict(img, use_textline_orientation=True)
    # 如果新版返回页面字典
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0].get("rec_texts", [])

    lines = []
    for item in raw:
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], (list, tuple)):
            _, (txt, _) = item; lines.append(txt)
        elif isinstance(item, list) and len(item) == 3 and isinstance(item[1], str):
            _, txt, _ = item; lines.append(txt)
        else:
            for sub in item:
                if len(sub) == 2 and isinstance(sub[1], (list, tuple)):
                    _, (txt, _) = sub; lines.append(txt)
                elif len(sub) == 3 and isinstance(sub[1], str):
                    _, txt, _ = sub; lines.append(txt)
    return lines

# ——— 正则：先文件名，再 OCR 匹配 ———
filecode_pattern = re.compile(r'([A-Z]+(?:/[A-Z])?\s*\d+(?:\.\d+)*[-–—]\d+)', re.IGNORECASE)
ocr_pattern      = re.compile(
    r'\b'          # 边界
    r'[A-Z]+'      # 前缀
    r'(?:/[A-Z])?' # 可选 "/T"
    r'\s*'
    r'\d+(?:\.\d+)*' # 支持带小数点
    r'\s*'
    r'[-–—]'       # 中横、en-dash、em-dash
    r'\s*'
    r'\d+'         # 修订号
    r'\b'
)

# ——— 批量处理 ———
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith('.pdf'):
        continue

    src_path = os.path.join(INPUT_DIR, fname)

    # 1. 优先从文件名里提取
    m0 = filecode_pattern.search(fname)
    if m0:
        code = m0.group(1).replace(" ", "")
        print(f"[Filename] {fname} → {code}.pdf")
    else:
        # 2. 文件名里没找到，再做 OCR
        try:
            doc = fitz.open(src_path)
            page = doc.load_page(0)
            # 动态计算 DPI，确保渲染后最大边 ≤ 4000px
            rect = page.rect
            MAX_SIDE = 4000
            dpi = int(MAX_SIDE * 72 / max(rect.width, rect.height))
            pix = page.get_pixmap(dpi=dpi, alpha=False)

            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            lines = ocr_page_to_lines(bgr)
            print(f"[OCR] {fname} 第1页 行数: {len(lines)}")

            text_block = "\n".join(lines)
            m1 = ocr_pattern.search(text_block)
            if not m1:
                print(f"[Skip] {fname}，未识别到编号")
                continue

            code = m1.group().replace(" ", "")
            print(f"[OCR] {fname} → {code}.pdf")

        except Exception as e:
            print(f"[Error] 处理 {fname} 时出错：{e}")
            continue

    # 3. 复制并重命名
    new_fname = f"{code}.pdf"
    dst_path = os.path.join(OUTPUT_DIR, new_fname)
    shutil.copy2(src_path, dst_path)

print("🎉 批量重命名完成。")