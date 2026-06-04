import os
import re

def delete_duplicate_files(folder_path):
    # 正则表达式匹配文件名末尾的 (数字) 格式
    pattern = re.compile(r"\(\d+\)\.[^\\/:*?\"<>|]+$")

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 检查文件是否匹配模式
        if pattern.search(filename):
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除失败: {file_path}, 错误: {e}")

if __name__ == "__main__":
    # 替换成你要操作的文件夹路径
    target_folder = r"E:\aaacode\ssll\data\PDF"
    delete_duplicate_files(target_folder)