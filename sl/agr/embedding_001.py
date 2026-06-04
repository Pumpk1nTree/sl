#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 PaddleOCR GPU/CPU 自动切换 + 本地 Embedding API 的 OCR & Embedding 管道
新增：
  - 原生文本提取优先
  - pdfplumber 表格提取
  - 图示提取
  - 全局唯一 ID 管理
"""
import os
import io
import json
import requests
import fitz           # PyMuPDF
import pdfplumber
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# 可选 CLIP 模型：若不可用则跳过示意图嵌入
has_clip = False
try:
    from transformers import CLIPProcessor, CLIPModel
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    has_clip = True
    print("CLIP: 已启用示意图嵌入")
except Exception as e:
    print(f"Warning: CLIPModel unavailable ({e}). Diagram embedding will be skipped.")

# 本地 Embedding API
EMBEDDING_API = "http://localhost:8000/embeddings"

# 初始化 OCR
ocr = PaddleOCR(use_textline_orientation=True, lang="ch")


def extract_paragraphs(lines, y_thr=20):
    """基于行框的简单聚合，合并纵坐标接近的文字为段落。"""
    sorted_lines = sorted(lines, key=lambda x: x[0][1])
    paras, cur_y, cur_text = [], None, ''
    for (x, y), txt in sorted_lines:
        if cur_y is None or abs(y - cur_y) < y_thr:
            cur_text += txt
            cur_y = y if cur_y is None else cur_y
        else:
            paras.append(cur_text)
            cur_text, cur_y = txt, y
    if cur_text:
        paras.append(cur_text)
    return paras


def chunk_text(text: str, size=500, overlap=50):
    """将长文本切成 size 大小、overlap 重叠的若干块。"""
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks


def embed_text_local(text: str):
    """调用本地文本嵌入服务，单条请求。"""
    try:
        resp = requests.post(EMBEDDING_API, json={"texts":[text]})
        resp.raise_for_status()
        return resp.json().get("embeddings", [[]])[0]
    except Exception as e:
        print(f"Warning: 文本嵌入失败: {e}")
        return []


def embed_image(img: Image.Image):
    """当 CLIP 可用时，生成图像嵌入；否则返回空。"""
    if not has_clip:
        return []
    try:
        inputs = clip_processor(images=img, return_tensors="pt")
        feats  = clip_model.get_image_features(**inputs)
        return feats.detach().cpu().numpy().tolist()[0]
    except Exception as e:
        print(f"Warning: 图片嵌入失败: {e}")
        return []


def render_page_to_image(page, dpi=300) -> Image.Image:
    """将 PDF 页面渲染为 PIL Image。"""
    zoom = dpi / 72
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def process_pdf(path: str, start_id: int = 1):
    """
    处理单个 PDF：原生文本／OCR 回退／表格抽取／示意图抽取。
    返回： (records 列表, 下一个可用 ID)
    """
    records = []
    rid = start_id

    # 同时打开 PyMuPDF & pdfplumber
    doc = fitz.open(path)
    pl  = pdfplumber.open(path)

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1

        # 1) 原生文本提取
        native = page.get_text("text") or ''
        if len(native.strip()) > 20:
            # 切块嵌入
            for chunk in chunk_text(native):
                records.append({
                    'id':     rid,
                    'type':   'paragraph',
                    'vector': embed_text_local(chunk),
                    'meta':   {'page': page_num}
                })
                rid += 1
        else:
            # 2) OCR 回退
            img = render_page_to_image(page)
            arr = np.array(img)
            try:
                ocr_out = ocr.ocr(arr)  # v3.x 返回 list of dict
            except Exception as e:
                print(f"Warning: OCR 失败 (page {page_num}): {e}")
                ocr_out = []

            # 解析 OCR 输出
            for block in ocr_out:
                # dt_boxes + rec_texts 可能在 dict 内
                polys = block.get('dt_polys') or block.get('rec_polys') or []
                texts = block.get('rec_texts') or []
                lines = [((int(p[0][0]), int(p[0][1])), txt)
                         for p,txt in zip(polys, texts)]
                paras = extract_paragraphs(lines)
                for para in paras:
                    for chunk in chunk_text(para):
                        records.append({
                            'id':     rid,
                            'type':   'paragraph',
                            'vector': embed_text_local(chunk),
                            'meta':   {'page': page_num}
                        })
                        rid += 1
        # 3) 表格提取
        pl_page = pl.pages[page_idx]
        tables  = pl_page.extract_tables()
        for ti, table in enumerate(tables):
            # 转为制表符分行纯文本
            txt = "\n".join(["\t".join(cell or "" for cell in row) for row in table])
            for chunk in chunk_text(txt):
                records.append({
                    'id':     rid,
                    'type':   'table',
                    'vector': embed_text_local(chunk),
                    'meta':   {'page': page_num, 'table': ti}
                })
                rid += 1
        # 4) 示意图嵌入
        if has_clip:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                pix  = fitz.Pixmap(doc, xref)
                if pix.n >= 5:  # CMYK 转 RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                diag = Image.open(io.BytesIO(pix.tobytes("png")))
                records.append({
                    'id':     rid,
                    'type':   'diagram',
                    'vector': embed_image(diag),
                    'meta':   {'page': page_num, 'xref': xref}
                })
                rid += 1

    pl.close()
    doc.close()
    return records, rid


def process_folder_cn(input_dir: str, save_path: str):
    """遍历文件夹，依次处理每个 PDF，并将所有嵌入结果合并为一个 JSON 文件。"""
    all_recs = []
    rid = 1
    for fn in os.listdir(input_dir):
        if not fn.lower().endswith('.pdf'):
            continue
        path = os.path.join(input_dir, fn)
        print(f"-> Processing {fn}")
        recs, rid = process_pdf(path, start_id=rid)
        print(f"   Extracted {len(recs)} records")
        all_recs.extend(recs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_recs, f, ensure_ascii=False, indent=2)
    print(f"=== All done: {len(all_recs)} embeddings saved to {save_path}")


if __name__ == '__main__':
    process_folder_cn(
        r"E:\aaacode\OCR\PDF",
        r"E:\aaacode\OCR\emb\cn.json"
    )