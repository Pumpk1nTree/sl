import os
import re
import shutil
import pandas as pd

# —— 路径配置 —— 
excel_path   = r"E:\aaacode\ssll\data\list\loc\local_list.xlsx"
old_root     = r"E:\aaacode\ssll\data\PDF\local_sub" # 分类后的原文件夹
new_root     = r"E:\aaacode\ssll\data\PDF\rename"       # 重命名后存放的位置

# —— 工具函数 —— 
def extract_digits(code: str) -> str:
    """
    从 code 中抽取连续的 3–6 位数字，用于匹配文件名
    """
    m = re.search(r'([0-9]{3,6})', code)
    return m.group(1) if m else None

def sanitize_filename(name: str) -> str:
    """
    把 Windows 下保留的非法字符(\ / : * ? " < > |)替换为下划线，生成合法文件名
    """
    return re.sub(r'[\\/:\*\?"<>\|]', '_', name)

def debug_rename_and_copy():
    df = pd.read_excel(excel_path, dtype={'省份': str, 'code': str})
    df['省份'] = df['省份'].str.strip()
    df['code']       = df['code'].str.strip()

    total = len(df)
    for idx, row in df.iterrows():
        dept   = row['省份']
        code   = row['code']
        digits = extract_digits(code)

        print(f"\n[{idx+1}/{total}] dept={dept!r}, code={code!r}, digits={digits!r}")

        src_dir = os.path.join(old_root, dept)
        if not os.path.isdir(src_dir):
            print("  ✗ 源目录不存在:", src_dir)
            continue
        print("  ✓ 源目录:", src_dir)

        all_pdfs = [f for f in os.listdir(src_dir) if f.lower().endswith('.pdf')]
        print(f"  • 目录下共 {len(all_pdfs)} 个 PDF 文件")

        # 1) 尝试完整 code 匹配
        cands = [fn for fn in all_pdfs if code in fn]
        method = "full code"
        # 2) 如果没命中，再尝试数字匹配
        if not cands and digits:
            cands = [fn for fn in all_pdfs if digits in fn]
            method = "digits fallback"

        if not cands:
            print(f"  ✗ 无匹配 ({method})，跳过")
            continue
        print(f"  ✓ 匹配到 {len(cands)} 个文件 ({method}): {cands}")

        # 确保目标子目录存在
        dst_dir = os.path.join(new_root, dept)
        os.makedirs(dst_dir, exist_ok=True)

        # 清洗 code，生成安全文件名
        safe_code = sanitize_filename(code)

        for i, old_fn in enumerate(cands, start=1):
            src_path = os.path.join(src_dir, old_fn)
            if len(cands) == 1:
                new_fn = f"{safe_code}.pdf"
            else:
                new_fn = f"{safe_code}_{i}.pdf"
            dst_path = os.path.join(dst_dir, new_fn)

            if os.path.exists(dst_path):
                print("    → [跳过] 目标已存在:", new_fn)
                continue

            shutil.copy2(src_path, dst_path)
            print("    → [复制重命名]", old_fn, "→", new_fn)

    print("\n=== 全部处理完成 ===")

if __name__ == "__main__":
    debug_rename_and_copy()