#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# —— 配置区 —— 
SEARCH_URL     = "https://ba.sacinfo.org.cn/stdList"
KEYWORD        = "水"
OUTPUT_FILE    = "地方标准_水.xlsx"
HEADLESS       = True
PAGE_LOAD_WAIT = 2    # 每次翻页后等待 JS 渲染的秒数

# 你的 chromedriver.exe 完整路径
CHROMEDRIVER_PATH = r"E:\chromedriver-win64\chromedriver.exe"
# ————————

def init_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless")
        opts.add_argument("--disable-gpu")
    # 你可以根据需要继续添加其他 ChromeOptions，例如：
    # opts.add_argument("--window-size=1920,1080")
    # opts.add_argument("--no-sandbox")

    # 显式指定 chromedriver 路径，绕过 Selenium Manager
    service = Service(executable_path=CHROMEDRIVER_PATH)
    return webdriver.Chrome(service=service, options=opts)

def scrape():
    driver = init_driver(HEADLESS)
    driver.get(SEARCH_URL)
    time.sleep(PAGE_LOAD_WAIT)

    # 1) 输入关键词并点击“查询”
    search_input = driver.find_element(By.TAG_NAME, "input")
    search_input.clear()
    search_input.send_keys(KEYWORD)

    search_btn = driver.find_element(By.XPATH, "//button[contains(text(),'查询')]")
    search_btn.click()
    time.sleep(PAGE_LOAD_WAIT)

    all_records = []
    page_idx    = 1

    while True:
        print(f"抓取第 {page_idx} 页…")
        # 2) 定位到结果表格
        table = driver.find_element(By.TAG_NAME, "table")

        # 3) 定位数据所在的 <tbody>
        tbodies = table.find_elements(By.TAG_NAME, "tbody")
        data_tbody = tbodies[1] if len(tbodies) > 1 else tbodies[0]

        # 4) 遍历每一行，抽取单元格文本
        rows = data_tbody.find_elements(By.TAG_NAME, "tr")
        if not rows:
            print("本页没有数据，结束。")
            break

        for tr in rows:
            cols = tr.find_elements(By.TAG_NAME, "td")
            if len(cols) < 8:
                continue
            record = [
                cols[0].text,               # 序号
                cols[1].text,               # 标准号
                cols[2].text,               # 标准名称
                cols[3].text,               # 省市区
                cols[4].text,               # 状态
                cols[5].text,               # 批准日期
                cols[6].text,               # 实施日期
                cols[7].text,               # 备案号
                cols[8].text if len(cols) > 8 else ""  # 备案日期（可选）
            ]
            all_records.append(record)

        # 5) 翻到下一页
        try:
            next_btn = driver.find_element(By.LINK_TEXT, "下一页")
            if "disabled" in next_btn.get_attribute("class"):
                break
            next_btn.click()
            page_idx += 1
            time.sleep(PAGE_LOAD_WAIT)
        except (NoSuchElementException, TimeoutException):
            break

    driver.quit()

    # 6) 保存到 Excel
    if all_records:
        columns = [
            "序号", "标准号", "标准名称",
            "省市区", "状态", "批准日期",
            "实施日期", "备案号", "备案日期"
        ]
        df = pd.DataFrame(all_records, columns=columns)
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"共抓取 {len(df)} 条，已保存到 {OUTPUT_FILE}")
    else:
        print("未抓取到任何数据。")

if __name__ == "__main__":
    scrape()