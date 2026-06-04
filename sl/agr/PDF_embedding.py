import os
import fitz  # PyMuPDF
import pdfplumber
import requests
import json
from paddleocr import PaddleOCR
from tqdm import tqdm


ocr = PaddleOCR(use_angle_cls=True, lang="ch")
EMBEDDING_API = "http://localhost:8000/embeddings"

def get_embeddings(texts, api_url=EMBEDDING_API):
    payload = {"texts": texts}
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"]

def extract_text_from_pdf(pdf_path):
    # 提取文本
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return texts

def extract_images_from_pdf(pdf_path):
    # 提取图片，返回图片对象list
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # this is GRAY or RGB
                imgdata = pix.tobytes()
                images.append(imgdata)
            pix = None
    return images


def ocr_images(images):
    ocr_results = []
    for img_path in images:
        result = ocr.predict(img_path)    # 正确写法
        # ...解析 result
        # result 格式: [ [ [bbox, (text, confidence)], ... ], ...]
        text_list = []
        for line in result:
            for res in line:
                text_list.append(res[1][0])  # 提取识别到的文本
        ocr_results.append('\n'.join(text_list))
    return ocr_results

def process_pdf(pdf_path):
    texts = extract_text_from_pdf(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    ocr_texts = ocr_images(images) if images else []
    # 合并所有内容用于embedding
    all_texts = texts + ocr_texts
    embeddings = get_embeddings(all_texts)
    # 返回 [ {"text": t, "embedding": emb} ... ]
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