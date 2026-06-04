from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import os
import time

KEYWORDS = ['火灾', '危险', '水利']
SAVE_DIR = r'E:\aaacode\ssll\CH'
os.makedirs(SAVE_DIR, exist_ok=True)

COOKIES = {
    'Hm_lvt_50758913e6f0dfc9deacbfebce3637e4': '1749458227,1749695065,1749723077,1749777606',
    'HMACCOUNT': '39DF6F88EE726D53',
    'Hm_lpvt_50758913e6f0dfc9deacbfebce3637e4': '1749777608',
    'JSESSIONID': '18F8841BED84940DDE4879DC3456FE91',
}
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Cookie": "; ".join(f"{k}={v}" for k, v in COOKIES.items())
}

LIST_URL = "https://openstd.samr.gov.cn/bzgk/gb/std_list_type?p.p1=1&p.p90=circulation_date&p.p91=desc"
BASE_URL = "https://openstd.samr.gov.cn"

options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

driver.get(LIST_URL)
time.sleep(2)

while True:
    rows = driver.find_elements(By.CSS_SELECTOR, "table.tablelist tr")[1:]  # 跳过表头
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) < 4:
            continue
        std_name = cols[2].text.strip()
        if not any(kw in std_name for kw in KEYWORDS):
            continue
        print("匹配到:", std_name)

        # 点击“查看详细”按钮（button有onclick事件）
        try:
            detail_btn = row.find_element(By.XPATH, ".//button[contains(text(), '查看详细')]")
            driver.execute_script("arguments[0].click();", detail_btn)
            time.sleep(2)
            driver.switch_to.window(driver.window_handles[-1])
        except Exception as e:
            print("无法点击详情:", e)
            continue

        # 找到“下载标准”按钮，获取data-value
        try:
            download_btn = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//button[contains(@class,'xz_btn') and contains(text(), '下载标准')]")
                )
            )
            data_value = download_btn.get_attribute('data-value')
            print("data-value:", data_value)

            # 直接构造下载接口（猜测接口格式如下，若不行可F12网络分析查下实际下载请求）
            # 示例接口: /bzgk/gb/downloadGb?hcno=xxxx
            # 或可能需要post请求，见实际F12
            download_api = f"https://openstd.samr.gov.cn/bzgk/gb/downloadGb?hcno={data_value}"
            filename = os.path.join(SAVE_DIR, f"{std_name}.pdf")

            if os.path.exists(filename):
                print("文件已存在，跳过：", filename)
            else:
                # 请求pdf并保存
                resp = requests.get(download_api, headers=HEADERS, cookies=COOKIES, stream=True)
                resp.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in resp.iter_content(1024 * 100):
                        if chunk:
                            f.write(chunk)
                print("已下载：", filename)
        except Exception as e:
            print("下载标准按钮未找到或下载失败：", e)

        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        time.sleep(1)

    # 翻页
    try:
        next_btn = driver.find_element(By.LINK_TEXT, "下一页")
        if "disabled" in next_btn.get_attribute("class"):
            break
        driver.execute_script("arguments[0].click();", next_btn)
        time.sleep(2)
    except Exception:
        print("没有下一页了")
        break

driver.quit()