import os
import requests
import pandas as pd

# 读取 Excel 文件
df = pd.read_excel(r"E:\aaacode\ssll\data\list.xlsx")

# 下载保存目录
save_dir = r"E:\aaacode\ssll\data\PDF"
os.makedirs(save_dir, exist_ok=True)

# 基础下载地址模板
BASE_URL = "https://dbba.sacinfo.org.cn/stdDetail/{}"

for idx, row in df.iterrows():
    pk = row["pk"]
    name = row["chName"]
    code = row["code"]
    
    url = BASE_URL.format(pk)
    filename = f"{code}_{name}.pdf".replace("/", "_").replace("\\", "_")
    filepath = os.path.join(save_dir, filename)
    
    try:
        print(f"正在下载: {name}")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"下载成功: {filepath}")
        else:
            print(f"下载失败: {name}, 状态码: {resp.status_code}")
    except Exception as e:
        print(f"下载异常: {name}, 错误: {e}")