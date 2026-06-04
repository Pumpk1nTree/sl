import pandas as pd
from selenium import webdriver
import os
import time
import re

# 配置
excel_path = r"E:\aaacode\ssll\data\list.xlsx"
save_dir = r"E:\aaacode\ssll\data\PDF"
os.makedirs(save_dir, exist_ok=True)

# 从 Excel 读取数据
df = pd.read_excel(excel_path)
df['code'] = df['code'].astype(str).fillna('')
df['chName'] = df['chName'].astype(str).fillna('')
df['pk'] = df['pk'].astype(str).fillna('')

# 你整理的标准号列表
standard_list = [
    "DB5323/T 76—2024",
    "DB5323/T 53—2024",
    "DB5323/T 52—2024",
    "DB34/T 4855-2024",
    "DB41/T 2690-2024",
    "DB37/T 4715—2024",
    "DB45/T 2758-2023",
    "DB41/T 2619-2024",
    "DB43/T 2428-2022",
    "DB63/T 2060-2022",
    "DB33/T 2512-2022",
    "DB32/T 4294-2022",
    "DB65/T 4468-2021",
    "DB41/T 2202-2021",
    "DB23/T1496.26-2021",
    "DB13/T 5294-2020",
    "DB32/T 3841-2020",
    "DB35/T 1915-2020",
    "DB41/T 1959-2020",
    "DB34/T 2990-2017",
    "DB50/T 867.10-2019",
    "DB34/T 2922-2017",
    "DB36/T 1084-2018",
    "DB22/T 134-2018",
    "DB21/T 2973-2018",
    "DB21/T 2972-2018",
    "DB21/T 2971-2018",
    "DB52/T 1218-2017",
    "DB34/T 2632-2016",
    "DB32/T 3261-2017",
    "DB32/T 3260-2017",
    "DB43/T 970-2014",
    "DB51/T 2202-2016",
    "DB41/T 1175-2015",
    "DB12/ 617-2016",
    "DB32/T 2949-2016",
    "DB32/T 2948-2016",
    "DB32/T 2931-2016",
    "DB21/T 2481-2015",
    "DB34/T 2206-2014",
    "DB50/T 576-2014",
    "DB34/T 2119-2014",
    "DB32/T 2709-2014",
    "DB32/T 2708-2014",
    "DB32/T 2707-2014",
    "DB23/T 1501-2013",
    "DB21/T 2241-2014",
    "DB37/T 2436-2013",
    "DB37/T 2437-2013",
    "DB45/T 952-2013",
    "DB41/T 818-2013",
    "DB32/T 2334.1-2013",
    "DB32/T 2334.4-2013",
    "DB32/T 2334.2-2013",
    "DB32/T 2334.3-2013",
    "DB32/T 2333-2013",
    "DB23/T 1496.26-2012",
    "DB63/T 1113-2012",
    "DB32/T 1713-2011",
    "DB32/T 1712-2011",
    "DB33/T 809-2010",
    "DB51/ 1178-2010",
    "DB11/T 685-2009",
    "DB43/ 417-2008",
    "DB37/T 1072-2008",
    "DB14/T 505-2008",
    "DB53/T 207.5-2007",
    "DB53/T 190.07-2006",
    "DB33/T 408-2003",
    "DB51/ 278.059-1998"
]

# 已完成记录文件
completed_file = os.path.join(save_dir, 'completed.txt')
if os.path.exists(completed_file):
    with open(completed_file, 'r', encoding='utf-8') as f:
        completed = set(line.strip() for line in f)
else:
    completed = set()

# 逐个处理
for std_code in standard_list:
    if std_code in completed:
        print(f"✅ 已下载：{std_code}")
        continue

    # 在 DataFrame 中查找对应 pk
    matched_row = df[df['code'].str.contains(re.escape(std_code.split()[0]), regex=True, na=False)]
    if matched_row.empty:
        print(f"⚠️ 没找到对应 pk：{std_code}，跳过")
        continue

    pk = matched_row.iloc[0]['pk']
    url = f"https://dbba.sacinfo.org.cn/stdDetail/{pk}"
    print(f"\n第 {standard_list.index(std_code)+1}/{len(standard_list)} 个标准：{std_code}")
    print(f"对应 pk：{pk}")
    print(f"访问：{url}")

    driver = webdriver.Chrome()
    try:
        driver.get(url)
        print("请手工输入验证码并点击下载按钮，如果失败按 Enter 跳过...")
        time.sleep(10)  # 等待用户操作
        input("如已完成下载，按 Enter 继续；如失败也按 Enter 跳过")
    except Exception as e:
        print(f"⚠️ 打开页面失败：{e}")
    finally:
        driver.quit()

    # 记录已完成
    with open(completed_file, 'a', encoding='utf-8') as f:
        f.write(std_code + '\n')

print("\n🎉 全部处理完毕！")