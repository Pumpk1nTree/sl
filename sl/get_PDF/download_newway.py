import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np  # 新增

# ---- 配置区 ----
BASE_URL   = "https://openstd.samr.gov.cn/bzgk/gb/std_list_type"
# 初始请求参数
params = {
    "r":        str(np.random.rand()),  # 随机防缓存，改为 np.random
    "page":     1,                      # 当前页码
    "pageSize": 10,                     # 每页条数
    "p.p1":     1,                      # 标准类别：1=强制性国家标准
    "p.p5":     "PUBLISHED",
    "p.p90":    "circulation_date",
    "p.p91":    "desc",
}
# 浏览器 Cookie
cookie_str = (
    "Hm_lvt_50758913e6f0dfc9deacbfebce3637e4=1749695065,1749723077,1749777606,1750038109; "
    "Hm_lpvt_50758913e6f0dfc9deacbfebce3637e4=1750141401; "
    "JSESSIONID=15B37557978195FC7716F93EEE4E0B2A; "
    "Hm_lvt_50758913e6f0dfc9deacbfebebce3637e4=1749458227,1750141363; "
    "HMACCOUNT=685F2073CA5DA7A1; "
    "Hm_lpvt_50758913e6f0dfc9deacbfebce3637e4=1750141378"
)

COOKIES = {
    kv.split("=", 1)[0].strip(): kv.split("=", 1)[1].strip()
    for kv in cookie_str.split(";") if "=" in kv
}
HEADERS     = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
OUTPUT_FILE = "std_list_all.xlsx"
# -------------------

all_rows = []
headers  = None

while True:
    print(f"抓取第 {params['page']} 页…")
    resp = requests.get(BASE_URL, params=params, cookies=COOKIES, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        print("无更多页面，退出。")
        break
    resp.raise_for_status()

    soup  = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", class_="result_list")
    if not table:
        print("未找到结果表格，结束。")
        break

    # 读取表头（仅第一页）
    if headers is None:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        headers.append("pk")

    # 读取第二个 <tbody> 的所有数据行
    tbodies   = table.find_all("tbody")
    data_rows = tbodies[1].find_all("tr") if len(tbodies) > 1 else []
    if not data_rows:
        print("本页无数据，结束。")
        break

    for tr in data_rows:
        tds = tr.find_all("td")
        if len(tds) != len(headers) - 1:
            continue
        row = [td.get_text(strip=True) for td in tds]
        btn = tr.find("button", onclick=True)
        pk  = ""
        if btn:
            m = re.search(r"showInfo\('([0-9A-F]+)'\)", btn["onclick"])
            if m:
                pk = m.group(1)
        row.append(pk)
        all_rows.append(row)

    # 下一页
    params["page"] += 1
    params["r"]     = str(np.random.rand())  # 每次更新随机数

# 写入 Excel
if all_rows:
    df = pd.DataFrame(all_rows, columns=headers)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"共保存 {len(df)} 条记录到 {OUTPUT_FILE}")
else:
    print("未抓取到任何记录。")