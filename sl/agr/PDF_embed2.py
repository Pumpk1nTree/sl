import os
import fitz
import pdfplumber
import requests
import json
from paddleocr import PaddleOCR
from tqdm import tqdm
import numpy as np
import cv2

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  
EMBEDDING_API = "http://localhost:8000/embeddings"

def get_embeddings(texts, api_url=EMBEDDING_API):
    payload = {"texts": texts}
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"]

def extract_text_from_pdf(pdf_path):
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return texts


def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_bytes = pix.tobytes()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                continue
            # 严格转换为 (H, W, 3) 且 uint8
            if len(img_cv.shape) == 2:  # 灰度
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
            elif img_cv.shape[2] == 4:  # BGRA
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)
            if img_cv.dtype != np.uint8:
                img_cv = img_cv.astype(np.uint8)
            # 再次校验
            if len(img_cv.shape) != 3 or img_cv.shape[2] != 3:
                print(f"[警告] 图片shape异常，跳过: {img_cv.shape}")
                continue
            images.append(img_cv)
            pix = None
    return images

def ocr_pdf(pdf_path):
    result = ocr.predict(pdf_path)
    text_list = []
    # result 是 list，每一页都是 list，每行为 (box, (text, score))
    for page in result:
        for line in page:
            if isinstance(line, list):
                for item in line:
                    if isinstance(item, tuple) and len(item) > 1:
                        text_list.append(item[1][0])
            elif isinstance(line, tuple) and len(line) > 1:
                text_list.append(line[1][0])
    text = '\n'.join(text_list)
    return [text]

def process_pdf(pdf_path):
    print(f"\n[流程] 开始处理: {pdf_path}")
    texts = extract_text_from_pdf(pdf_path)
    print(f"[流程] 文本抽取完成，共{len(texts)}条")

    # 用更稳定的 OCR
    ocr_texts = ocr_pdf(pdf_path)
    print(f"[流程] OCR识别完成，共{len(ocr_texts)}段")

    all_texts = list(set(t.strip() for t in texts + ocr_texts if t and t.strip()))
    print(f"[流程] 合并去重后待embedding文本数：{len(all_texts)}")

    embeddings = get_embeddings(all_texts)
    print(f"[流程] embedding生成完成，本文件所有流程结束\n")
    return [{"text": t, "embedding": emb} for t, emb in zip(all_texts, embeddings)]

def process_folder_loc(folder_path, save_dir):
    # 每省单独一个embedding文件
    for province in os.listdir(folder_path):
        prov_path = os.path.join(folder_path, province)
        if not os.path.isdir(prov_path):
            continue
        print(f"正在处理省份：{province}")
        embend_list = []
        for pdf in tqdm(os.listdir(prov_path), desc=f"处理{province}下PDF"):
            pdf_path = os.path.join(prov_path, pdf)
            if not pdf_path.endswith('.pdf'):
                continue
            print(f"  开始embedding文件：{pdf}")
            embend_list.extend(process_pdf(pdf_path))
            print(f"  完成embedding文件：{pdf}")
        save_path = os.path.join(save_dir, f"{province}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(embend_list, f, ensure_ascii=False, indent=2)
        print(f"已完成省份 {province} 的全部embedding，保存至 {save_path}")

def process_folder_cn(folder_path, save_path):
    embend_list = []
    pdfs = [pdf for pdf in os.listdir(folder_path) if pdf.endswith('.pdf')]
    for pdf in tqdm(pdfs, desc="国家标准PDF"):
        pdf_path = os.path.join(folder_path, pdf)
        print(f"开始embedding国家标准文件：{pdf}")
        embend_list.extend(process_pdf(pdf_path))
        print(f"完成embedding国家标准文件：{pdf}") 
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(embend_list, f, ensure_ascii=False, indent=2)
    print(f"已完成全部国家标准embedding，保存至 {save_path}")

# 执行
if __name__ == "__main__":
    # 地方
    process_folder_loc(
        r"E:\aaacode\ssll\data\PDF\loc_spearate",
        r"E:\aaacode\ssll\embend\loc"
    )
    # 国家
    process_folder_cn(
        r"E:\aaacode\ssll\data\PDF\cn",
        r"E:\aaacode\ssll\embend\cn\cn.json"
    )
