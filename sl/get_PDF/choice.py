import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# ========== 基础配置 ==========
excel_path = r"E:\aaacode\ssll\data\list\CN\CN_GB_menu.xlsx"
save_path = r"E:\aaacode\ssll\data\list\CN\CN_GB_water_related.xlsx"

# ========== 加载模型客户端 ==========
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ========== 大模型判断函数 ==========
def is_water_related(standard_name):
    prompt = (
        f"请判断以下标准名称是否与水利或水电行业相关。"
        "判断要尽量宽泛，如果可能涉及水资源、水利工程、水电、水环境、防洪灌溉等，也算相关。"
        f"\n标准名称：{standard_name}\n"
        "请只回复：相关 或 不相关"
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages,
            temperature=0,
        )
        result = response.choices[0].message.content.strip()
        return "相关" in result
    except Exception as e:
        print(f"大模型调用失败：{e}，内容：{standard_name}")
        return False

# ========== 读取Excel & 逐条判断 ==========
df = pd.read_excel(excel_path)
if "标准名称" not in df.columns:
    raise ValueError("Excel中未找到'标准名称'列")

mask = []
for i, name in enumerate(df["标准名称"]):
    related = is_water_related(str(name))
    mask.append(related)
    print(f"{i+1}/{len(df)}：{name} → {'相关' if related else '不相关'}")
    # 推荐加延时，防止接口被限流
    time.sleep(0.3)

df_related = df[mask]
df_related.to_excel(save_path, index=False)
print(f"\n共筛选出 {len(df_related)} 条与水利水电相关的标准，已保存为 {save_path}")