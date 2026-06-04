import os
import requests
import pandas as pd
from datetime import datetime

def timestamp_to_str(ts):
    if ts is None or ts == "" or ts == 0:
        return ""
    try:
        ts_int = int(ts)
    except (ValueError, TypeError):
        return ""
    if ts_int > 1e12:
        ts_sec = ts_int / 1000
    else:
        ts_sec = ts_int
    try:
        return datetime.fromtimestamp(ts_sec).strftime("%Y-%m-%d")
    except (OSError, OverflowError, ValueError):
        return ""

def fetch_water_records(page_size=100):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/x-www-form-urlencoded",
    })
    url = "https://ba.sacinfo.org.cn/stdQueryList"
    all_records = []
    page = 1

    while True:
        payload = {
            "current": page,
            "size": page_size,
            "key": "水",
            "ministry": "",
            "industry": "",
            "pubdate": "",
            "date": "",
            "status": ""
        }
        resp = session.post(url, data=payload)
        resp.raise_for_status()
        js = resp.json()
        records = js.get("records", [])
        if not records:
            break
        for r in records:
            r["标准号"]   = r.get("code", "")
            r["标准名称"] = r.get("chName", "")
            r["省市区"]   = r.get("industry", "")
            r["状态"]     = r.get("status", "")
            r["批准日期"] = timestamp_to_str(r.get("issueDate"))
            r["实施日期"] = timestamp_to_str(r.get("actDate"))
            r["备案号"]   = r.get("recordNo", "")
            r["备案日期"] = timestamp_to_str(r.get("recordDate"))
            r["pk"]       = r.get("pk", "")
            for f in ["code","chName","industry","status","issueDate","actDate","recordNo","recordDate"]:
                r.pop(f, None)

        all_records.extend(records)
        print(f"已抓取第 {page} 页，共 {len(records)} 条（含“水”）")
        if len(records) < page_size:
            break
        page += 1
    return all_records

def save_to_excel(records, output_path):
    df = pd.DataFrame(records)
    cols = [c for c in df.columns if c != "pk"] + ["pk"]
    df = df[cols]
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"保存完成：{output_path}")

if __name__ == "__main__":
    recs = fetch_water_records(page_size=100)
    out_path = os.path.join("data", "list", "loc", "loc_list.xlsx")
    save_to_excel(recs, out_path)