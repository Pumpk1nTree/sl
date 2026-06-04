from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time
import os

# 文件路径
excel_path = r"E:\aaacode\ssll\data\list.xlsx"  # 已上传的文件路径
save_dir = r"E:\aaacode\ssll\data\PDF"
os.makedirs(save_dir, exist_ok=True)

# 读取 Excel 获取 pk 列表（假设列名为 'pk'，可改成实际列名）
df = pd.read_excel(excel_path)
pk_list = df['pk'].dropna().astype(str).tolist()

# 启动浏览器
driver = webdriver.Chrome()

for index, pk in enumerate(pk_list, start=1):
    url = f"https://dbba.sacinfo.org.cn/stdDetail/{pk}"
    driver.get(url)
    
    print(f"第 {index}/{len(pk_list)} 个标准：{pk}")
    print("请在浏览器中手工输入验证码并点击下载按钮...")

    # 等待用户手动完成下载
    time.sleep(10)  # 可以调整等待时间，也可以换成智能判断下载完成逻辑
    
    input("当前下载完成后，按 Enter 键继续下载下一个标准...")

print("所有标准处理完成！")
driver.quit()