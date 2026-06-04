import os
import glob
import json

# 你的PDF目录
pdf_folder = r"E:\aaacode\shuili\source"
save_path = r"E:\aaacode\shuili\data\rag_docs.jsonl"

# 你给的抽取函数（假设extract_text_from_pdf已定义好）
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_text_from_pdf(filename, page_numbers=None, min_line_length=10):
    paragraphs = []
    for i, page_layout in enumerate(extract_pages(filename)):
        if page_numbers is not None and i not in page_numbers:
            continue
        page_text = ''
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text += element.get_text()
        # 过滤掉纯空白页或只有很短内容的页
        if len(page_text.strip()) >= min_line_length:
            paragraphs.append(page_text.strip())
    return paragraphs

# 扫描PDF文件
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

# 批量处理并保存为JSONL
import os
import json

with open(save_path, "w", encoding="utf-8") as fout:
    total = 0
    for pdf_file in pdf_files:
        try:
            paragraphs = extract_text_from_pdf(pdf_file)
        except Exception as e:
            print(f"处理文件出错: {pdf_file}，错误信息: {e}")
            continue
        file_name = os.path.basename(pdf_file)
        for idx, para in enumerate(paragraphs):
            doc = {
                "doc_id": f"{file_name}-{idx+1}",
                "source": file_name,
                "paragraph_id": idx+1,
                "text": para.strip()
            }
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total += 1
print("所有文档已处理并保存为:", save_path)
print(f"总计处理段落数: {total}")