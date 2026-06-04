import os
import fitz  # PyMuPDF
import pdfplumber
import requests
import json
from paddleocr import PaddleOCR
from tqdm import tqdm
import tempfile

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
                texts.append(txt.strip())
    return texts


def extract_images_from_pdf(pdf_path):
    image_paths = []
    doc = fitz.open(pdf_path)
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n < 5:  # 灰度或RGB
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                pix.save(tmp.name)
                image_paths.append(tmp.name)
            pix = None
    return image_paths


def ocr_images(image_paths):
    ocr_results = []
    for img_path in image_paths:
        print(f"OCR processing image: {os.path.basename(img_path)}")
        result = ocr.predict(img_path)
        text_list = []
        for line in result:
            for res in line:
                text_list.append(res[1][0])
        ocr_results.append("\n".join(text_list))
        os.remove(img_path)
    return ocr_results


def process_pdf(pdf_path, index=None, total=None):
    if index is not None and total is not None:
        print(f"[{index}/{total}] 正在处理PDF: {os.path.basename(pdf_path)}")
    texts = extract_text_from_pdf(pdf_path)
    image_paths = extract_images_from_pdf(pdf_path)
    ocr_texts = ocr_images(image_paths) if image_paths else []
    all_texts = texts + ocr_texts
    embeddings = get_embeddings(all_texts)
    print(f"[{index}/{total}] 完成处理: {os.path.basename(pdf_path)}")
    return [{"text": t, "embedding": emb} for t, emb in zip(all_texts, embeddings)]


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
        print(f"省份 {province} 完成，共处理 {total_pdfs} 个文件，结果保存至 {save_path}\n")


def process_folder_cn(input_dir, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    total_pdfs = len(pdfs)
    embeds = []
    print(f"开始处理国家标准，共 {total_pdfs} 个PDF")
    for idx, pdf in enumerate(pdfs, start=1):
        pdf_path = os.path.join(input_dir, pdf)
        embeds.extend(process_pdf(pdf_path, idx, total_pdfs))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(embeds, f, ensure_ascii=False, indent=2)
    print(f"国家标准处理完成，共 {total_pdfs} 个文件，结果保存至 {save_path}")


if __name__ == "__main__":
    process_folder_loc(
        r"E:\aaacode\ssll\data\PDF\loc_spearate",
        r"E:\aaacode\ssll\embend\loc"
    )
    process_folder_cn(
        r"E:\aaacode\ssll\data\PDF\cn",
        r"E:\aaacode\ssll\embend\cn\cn.json"
    )