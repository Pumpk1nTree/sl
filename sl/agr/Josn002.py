import os
import io
import json
import fitz                
import numpy as np
from PIL import Image      
import cv2                
from paddleocr import PaddleOCR  
from tqdm import tqdm

# ——— 初始化 PaddleOCR ———
print("▶ 初始化 PaddleOCR …")
ocr = PaddleOCR(use_textline_orientation=True, lang="ch")
print("    ✓ 模型加载完毕\n")

def ocr_page_to_lines(img: np.ndarray) -> list[str]:
    """
    对一张 BGR numpy 图像跑 OCR，并把同一页的识别结果按行返回列表。
    """
    raw = ocr.predict(img, use_textline_orientation=True)

    # 如果返回新版“页面字典”格式，直接取 rec_texts 列表
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        page = raw[0]
        return page.get("rec_texts", [])

    lines = []
    for item in raw:
        # 扁平模式：[box, (txt, score)]
        if isinstance(item, list) and len(item) == 2 and isinstance(item[1], (list, tuple)):
            _, (txt, _) = item
            lines.append(txt)
        # 扁平模式：[box, txt, score]
        elif isinstance(item, list) and len(item) == 3 and isinstance(item[1], str):
            _, txt, _ = item
            lines.append(txt)
        else:
            # 嵌套多行模式
            for sub in item:
                if len(sub) == 2 and isinstance(sub[1], (list, tuple)):
                    _, (txt, _) = sub
                    lines.append(txt)
                elif len(sub) == 3 and isinstance(sub[1], str):
                    _, txt, _ = sub
                    lines.append(txt)
    return lines


def process_pdf(pdf_path):
    """
    对单个 PDF 文件进行 OCR，返回每页的文字列表。
    """
    print(f"处理 PDF: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    results = []

    for page_num in range(len(doc)):
        pix = doc.load_page(page_num).get_pixmap(dpi=300, alpha=False)
        img_data = pix.tobytes("png")

        pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lines = ocr_page_to_lines(bgr)
        results.append({
            "page": page_num + 1,
            "text": lines
        })
        print(f"    ✓ 页 {page_num+1} 行数: {len(lines)}")

    return results

def process_folder_loc(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    provinces = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for province in provinces:
        prov_path = os.path.join(input_dir, province)
        pdfs = [f for f in os.listdir(prov_path) if f.lower().endswith('.pdf')]
        embeds = []

        for pdf in tqdm(pdfs, desc=f"省份 {province}"):
            pdf_path = os.path.join(prov_path, pdf)
            embeds.extend(process_pdf(pdf_path))

        save_path = os.path.join(output_dir, f"{province}.json")
        with open(save_path, "w", encoding="utf-8") as fout:
            json.dump(embeds, fout, ensure_ascii=False, indent=2)
        print(f"省份 {province} 完成，保存至 {save_path}\n")

"""
def process_folder_cn(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    for pdf in tqdm(pdfs, desc="国家标准处理"):
        pdf_path = os.path.join(input_dir, pdf)
        results = process_pdf(pdf_path)
        base = os.path.splitext(pdf)[0]
        out_path = os.path.join(output_dir, f"{base}.json")
        with open(out_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, ensure_ascii=False, indent=2)
        print(f"{pdf} → {out_path}")

    print(f"国家标准处理完成，结果保存在 {output_dir}")
"""


def process_folder_cn(input_dir, output_file):
    """
    批量处理目录下所有国家标准 PDF，将所有结果汇总到一个 JSON 文件中。
    """
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    all_data = []

    for idx, pdf in enumerate(pdfs, start=1):
        path_in = os.path.join(input_dir, pdf)
        base = os.path.splitext(pdf)[0]
        print(f"[{idx}/{len(pdfs)}] 处理 PDF: {pdf}")
        pages = process_pdf(path_in)
        for page in pages:
            entry = {"file": base, "page": page['page'], "text": page['text']}
            all_data.append(entry)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"全部 PDF OCR 完成，结果保存在: {output_file}")

if __name__ == "__main__":
    # 示例：按省份分文件夹处理
    process_folder_loc(
        r"E:\aaacode\ssll\data\PDF\loc_spearate",
        r"E:\aaacode\ssll\josn\loc"
    )
    # 示例：所有国家标准同一目录处理
    process_folder_cn(
        r"E:\aaacode\ssll\data\PDF\cn", 
        r"E:\aaacode\ssll\josn\cn\cn.json"
    )