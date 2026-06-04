import os
import shutil

def copy_all_pdfs(src_root: str, dst_folder: str):
    """
    遍历 src_root 下所有子目录，复制其中的 PDF 文件到 dst_folder。

    :param src_root: 源目录，里面有多个以省份命名的子文件夹
    :param dst_folder: 目标目录，用于保存所有复制出来的 PDF
    """
    # 如果目标文件夹不存在，就创建
    os.makedirs(dst_folder, exist_ok=True)

    for root, dirs, files in os.walk(src_root):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                src_path = os.path.join(root, filename)
                dst_path = os.path.join(dst_folder, filename)

                # 若目标已有同名文件，则在文件名后加序号
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    idx = 1
                    while True:
                        new_name = f"{base}_{idx}{ext}"
                        dst_path = os.path.join(dst_folder, new_name)
                        if not os.path.exists(dst_path):
                            break
                        idx += 1

                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} → {dst_path}")

if __name__ == '__main__':
    # TODO: 根据实际路径修改这两个变量
    source_root = r"E:\aaacode\ssll\data\PDF\loc_spearate"     # 源目录，含省份子文件夹
    destination = r"E:\aaacode\ssll\data\PDF\local"      # 目标目录，保存所有PDF

    copy_all_pdfs(source_root, destination)