import os
import fitz  # PyMuPDF
import requests
import json
from paddleocr import PaddleOCR
from tqdm import tqdm
import tempfile

ocr = PaddleOCR(use_textline_orientation=True, lang="ch")  # 使用新参数替代use_angle_cls
EMBEDDING_API = "http://localhost:8000/embeddings"


def get_embeddings(texts, api_url=EMBEDDING_API):
    payload = {"texts": texts}
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def process_pdf(pdf_path, index=None, total=None):
    """
    对单个 PDF 按页进行 OCR 并生成 embedding，返回 [{'page': 页号, 'text': 文本, 'embedding': 向量}, ...]
    """
    if index is not None and total is not None:
        print(f"[{index}/{total}] 正在处理PDF: {os.path.basename(pdf_path)}")

    doc = fitz.open(pdf_path)
    page_texts = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_path = f"{tempfile.gettempdir()}/page_{page_num+1}.png"
        pix.save(img_path)

        # OCR 整页内容
        result = ocr.predict(img_path)
        lines = []
        for line in result:
            for res in line:
                lines.append(res[1][0])
        text = "\n".join(lines)
        page_texts.append(text)

        os.remove(img_path)

    embeddings = get_embeddings(page_texts)
    print(f"[{index}/{total}] 完成处理: {os.path.basename(pdf_path)}")

    return [
        {"page": i+1, "text": txt, "embedding": emb}
        for i, (txt, emb) in enumerate(zip(page_texts, embeddings))
    ]


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


def process_folder_cn(input_dir, output_dir):
    """
    For each PDF in input_dir, run process_pdf(...) to get its embeddings,
    and save them to output_dir/<pdf_basename>.json
    """
    os.makedirs(output_dir, exist_ok=True)
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    total = len(pdfs)
    print(f"开始处理国家标准，共 {total} 个PDF")
    
    for idx, pdf in enumerate(pdfs, start=1):
        pdf_path = os.path.join(input_dir, pdf)
        # process_pdf should return a list of embeddings for that single PDF
        embeds = process_pdf(pdf_path, idx, total)
        
        # build output filename: replace .pdf with .json
        base = os.path.splitext(pdf)[0]
        out_path = os.path.join(output_dir, f"{base}.json")
        
        # write per‐file JSON
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(embeds, f, ensure_ascii=False, indent=2)
        
        print(f"[{idx}/{total}] {pdf} → {out_path}")

    print(f"国家标准处理完成，结果保存在 {output_dir}")


if __name__ == "__main__":
    """
    process_folder_loc(
        r"E:\aaacode\ssll\data\PDF\loc_spearate",
        r"E:\aaacode\ssll\embend\loc"
    )
    """
    process_folder_cn(
        r"E:\aaacode\ssll\data\PDF\cn",
        r"E:\aaacode\ssll\embend\cn\cn.json"
    )