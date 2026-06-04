#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抓取 dbba.sacinfo.org.cn/stdQueryList 全量元数据
- 每页 15 条（接口硬限制）
- 自动断点续跑：若本地已有 partial CSV，则从最后一页 +1 继续
- 每 0.35 s 请求一页，跑完约 35 min
"""
import os, csv, time, datetime
import requests
from tqdm import tqdm
import pandas as pd

URL   = "https://dbba.sacinfo.org.cn/stdQueryList"
HEAD  = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}
PAGE_SIZE   = 15
CSV_NAME    = "dbba_all_standards_full.csv"

def ts2date(ms):
    return datetime.datetime.fromtimestamp(ms/1000).strftime("%Y-%m-%d")

def fetch_page(session, page_no):
    data = {
        "key": "",
        "current": page_no,
        "size": PAGE_SIZE,
        "searchCount": "true"
    }
    r = session.post(URL, data=data, headers=HEAD, timeout=15).json()

    # ⇩⇩ 关键修改：用 .get() 并做存在性检查 ⇩⇩
    for row in r["records"]:
        for k in ("actDate", "issueDate", "recordDate"):
            ts_val = row.get(k)
            if ts_val:                         # 既存在又非空才转
                row[k] = ts2date(ts_val)
            else:
                row[k] = ""                    # 或用 None

    return r

def main():
    sess = requests.Session()

    # ――― 决定从哪一页开始 ―――
    start_page = 1
    rows = []
    if os.path.exists(CSV_NAME):
        print(f"发现已有 {CSV_NAME}，尝试续跑…")
        rows = pd.read_csv(CSV_NAME).to_dict("records")
        start_page = len(rows)//PAGE_SIZE + 1
        print(f"   已有 {len(rows)} 条，将从第 {start_page} 页继续")

    # ――― 先拉一页，拿到总页数 ―――
    first = fetch_page(sess, start_page)
    total  = first["total"]
    pages  = first["pages"]
    if start_page == 1:
        rows = first["records"]          # 全新跑
    else:
        rows.extend(first["records"])    # 续跑把这一页拼进来

    print(f"目标总数 {total} 条，共 {pages} 页，开始抓取…")

    # ――― 主循环 ―――
    for pg in tqdm(range(start_page+1, pages+1), desc="抓取中"):
        resp = fetch_page(sess, pg)
        rows.extend(resp["records"])
        time.sleep(0.35)                 # 根据限速调节
        # 每 100 页落一次盘，防断电
        if pg % 100 == 0:
            pd.DataFrame(rows).drop_duplicates("pk").to_csv(
                CSV_NAME, index=False, encoding="utf-8")
    # ――― 最终保存 ―――
    pd.DataFrame(rows).drop_duplicates("pk").to_csv(
        CSV_NAME, index=False, encoding="utf-8")
    print(f"完毕！去重后 {len(set(r['pk'] for r in rows))} / {len(rows)} 行")

if __name__ == "__main__":
    main()