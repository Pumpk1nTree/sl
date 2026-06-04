from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
import time
import os
import requests

KEYWORDS = ['黄河', '水利']
SAVE_DIR = r'E:\std_download'
BASE_URL = 'https://openstd.samr.gov.cn'
DETAIL_URL = BASE_URL + '/bzgk/gb/newGbInfo?hcno={gid}'

os.makedirs(SAVE_DIR, exist_ok=True)

def keyword_in_title(title):
    return any(kw in title for kw in KEYWORDS)

def download_pdf(pdf_url, filename):
    file_path = os.path.join(SAVE_DIR, filename)
    if os.path.exists(file_path):
        print(f'已存在: {file_path}，跳过下载')
        return True
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        }
        r = requests.get(pdf_url, headers=headers, stream=True, timeout=15)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'下载完成: {file_path}')
        return True
    except Exception as e:
        print(f'下载失败: {pdf_url}, 错误: {e}')
        return False

def is_404_page(driver):
    page_text = driver.page_source
    return ("无法找到该页面" in page_text) or ("HTTP错误404" in page_text)

def crawl(max_pages=3):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(options=chrome_options)

    for page in range(1, max_pages + 1):
        url = f'{BASE_URL}/bzgk/gb/std_list_type?p.p1={page}&p.p90=circulation_date&p.p91=desc'
        driver.get(url)
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table.result_list"))
            )
        except Exception as e:
            print('等待表格加载超时:', e)
            with open(f'debug_{page}.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            driver.save_screenshot(f'debug_{page}.png')
            continue
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        table = soup.select_one('table.result_list')
        if not table:
            print('未找到标准列表表格')
            with open(f'debug_notable_{page}.html', 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            driver.save_screenshot(f'debug_notable_{page}.png')
            continue
        tbody = table.find('tbody')
        for row in tbody.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) < 5:
                continue
            std_num_a = cols[1].find('a')
            if not std_num_a:
                continue
            std_num = std_num_a.text.strip().replace(' ', '')
            std_title = cols[2].text.strip()
            status = cols[4].text.strip()
            if not keyword_in_title(std_title):
                continue
            onclick_str = std_num_a.get('onclick', '')
            gid_match = re.search(r"showInfo\('(.+?)'\)", onclick_str)
            if not gid_match:
                print('未找到gid')
                continue
            gid = gid_match.group(1)
            detail_url = DETAIL_URL.format(gid=gid)
            driver.get(detail_url)
            time.sleep(2)
            if is_404_page(driver):
                print(f'详情页404，跳过: {detail_url}')
                with open(f'debug_404_{std_num}.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                driver.save_screenshot(f'debug_404_{std_num}.png')
                continue
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "下载标准"))
                )
            except Exception as e:
                print(f'详情页等待下载按钮超时: {e}')
                with open(f'debug_detailtimeout_{std_num}.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                driver.save_screenshot(f'debug_detailtimeout_{std_num}.png')
                continue
            detail_html = driver.page_source
            detail_soup = BeautifulSoup(detail_html, 'lxml')
            btn = detail_soup.find('a', string="下载标准")
            if not btn:
                print(f'未找到下载链接: {detail_url}')
                with open(f'debug_nodownload_{std_num}.html', 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                driver.save_screenshot(f'debug_nodownload_{std_num}.png')
                continue
            href = btn['href']
            pdf_url = BASE_URL + href if not href.startswith('http') else href
            filename = f"{std_num}_{status}.pdf"
            filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
            print(f'匹配到: {filename} - {std_title}')
            download_pdf(pdf_url, filename)
            time.sleep(1.5)
        time.sleep(1)
    driver.quit()

if __name__ == '__main__':
    crawl(max_pages=5)