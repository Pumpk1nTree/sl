#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import datetime

import requests
import pandas as pd

# —— 配置区 —— 
BASE_URL    = "https://ba.sacinfo.org.cn/stdQueryList"
KEYWORD     = "水"
PAGE_SIZE   = 15
OUTPUT_FILE = "地方标准_水.xlsx"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}
# ————————

def ts_to_date(ms):
    """把 JS 毫秒时间戳转成 YYYY-MM-DD"""
    try:
        return datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d")
    except:
        return ""

def fetch_page(session, page_no):
    """
    调接口拿一页数据，返回 (total_count, records_list)
    records_list 是原始 JSON 里的 records 数组
    """
    payload = {
        "current":  page_no, 
        "size":     PAGE_SIZE,
        "key":      KEYWORD,
        "ministry": "",        # 省市区过滤，留空=全部
        "industry": "",        # 行业过滤，留空=全部
        "pubdate":  "",        # 备案日期过滤，留空=全部
        "date":     "",        # （备用字段，留空）
        "status":   "现行",     # 可选 '现行','有更新版','废止'，这里采用默认“现行”
    }
    resp = session.post(BASE_URL, data=payload, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    js = resp.json()
    return js["total"], js["records"]

def main():
    sess = requests.Session()

    # 1. 先请求第一页，拿 total
    total, recs1 = fetch_page(sess, 1)
    pages = math.ceil(total / PAGE_SIZE)
    print(f"总记录数 {total}，共 {pages} 页，每页 {PAGE_SIZE} 条。")

    all_recs = recs1.copy()

    # 2. 抓剩余页
    for p in range(2, pages + 1):
        print(f"抓取第 {p} / {pages} 页…")
        _, recs = fetch_page(sess, p)
        if not recs:
            break
        all_recs.extend(recs)

    # 3. 把 JSON 字段转成表格列
    rows = []
    for idx, r in enumerate(all_recs, start=1):
        rows.append({
            "#":       idx,
            "标准号":   r.get("code", ""),
            "标准名称": r.get("chName", ""),
            "省市区":   r.get("industry", ""),
            "状态":     r.get("status", ""),
            "批准日期": ts_to_date(r.get("issueDate")),
            "实施日期": ts_to_date(r.get("actDate")),
            "备案号":   r.get("recordNo", ""),
            "备案日期": ts_to_date(r.get("recordDate")),
        })

    # 4. 输出到 Excel
    df = pd.DataFrame(rows, columns=[
        "#", "标准号", "标准名称", "省市区", "状态",
        "批准日期", "实施日期", "备案号", "备案日期"
    ])
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"共保存 {len(df)} 条到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()