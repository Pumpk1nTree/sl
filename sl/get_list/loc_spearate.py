import pandas as pd
import os
import shutil
import re

# 路径定义
excel_path = r"E:\aaacode\ssll\data\list\loc\local_list.xlsx"
pdf_root = r"E:\aaacode\ssll\data\PDF\local"
save_root = r"E:\aaacode\ssll\data\PDF\loc_sub"

# 加载excel
df = pd.read_excel(excel_path)
df['省份'] = df['省份'].astype(str)
df['code'] = df['code'].astype(str)

# 获取所有PDF文件名
all_pdfs = [f for f in os.listdir(pdf_root) if f.lower().endswith('.pdf')]

classified = set()   # 记录被分类的文件
not_classified = set(all_pdfs)  # 初始为全部，后续分类成功就移除

def extract_xxxx_from_code(code):
    match = re.search(r'[Dd][Bb][A-Z]*[\s\+]*([0-9]{3,6})', code)
    if match:
        return match.group(1)
    else:
        match = re.search(r'([0-9]{3,6})', code)
        if match:
            return match.group(1)
    return None

for idx, row in df.iterrows():
    charge_dept = row['省份'].strip()
    code = row['code']
    xxxx = extract_xxxx_from_code(code)
    if not xxxx:
        continue

    matched_files = [f for f in all_pdfs if xxxx in f]
    if not matched_files:
        continue

    # 目标文件夹
    target_dir = os.path.join(save_root, charge_dept)
    os.makedirs(target_dir, exist_ok=True)

    for pdf_file in matched_files:
        src_path = os.path.join(pdf_root, pdf_file)
        dst_path = os.path.join(target_dir, pdf_file)
        if not os.path.exists(src_path):
            print(f"文件不存在，跳过：{pdf_file}")
            continue
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
            classified.add(pdf_file)
            not_classified.discard(pdf_file)

# 统计输出
print(f"分类完成，共 {len(classified)} 个文件被分类，{len(not_classified)} 个文件未分类。")
if not_classified:
    print("未分类文件如下：")
    for fname in not_classified:
        print(fname)