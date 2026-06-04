import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

excel_path = r"E:\aaacode\ssll\data\list\CN\CN_GB_list.xlsx"
url_tpl = "https://openstd.samr.gov.cn/bzgk/gb/newGbInfo?hcno={}"

df = pd.read_excel(excel_path)
pk_list = df['pk'].dropna().astype(str).tolist()

driver = webdriver.Chrome()
driver.maximize_window()

for i, pk in enumerate(pk_list[24:], start=25):
    print(f"\n[{i+1}/{len(pk_list)}] 打开标准：{pk}")
    url = url_tpl.format(pk)
    driver.get(url)

    try:
        # 修正：通过class定位“下载标准”按钮
        download_btn = driver.find_element(By.CLASS_NAME, "xz_btn")
        download_btn.click()
        print("已自动点击‘下载标准’，请手动输入验证码并保存文件。")
    except Exception as e:
        print("未找到‘下载标准’按钮，跳过此项。")
        continue

    input("完成本次下载后，按回车继续下一个...")

print("全部处理完毕！")
driver.quit()