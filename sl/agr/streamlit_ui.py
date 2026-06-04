import streamlit as st
import subprocess
import socket
import time
import io
import contextlib

# ============ 前置：确保本地 Embedding API 已启动 ============
def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """检查本地端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0

def start_embedding_api():
    """若 8000 端口未被占用则后台启动 uvicorn embedding_api:app"""
    if not is_port_open(8000):
        subprocess.Popen(
            ["uvicorn", "embedding_api:app", "--host", "0.0.0.0", "--port", "8000"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        # 预留 5 s 等待服务就绪
        time.sleep(5)

if "api_started" not in st.session_state:
    start_embedding_api()
    st.session_state.api_started = True

# ============ 导入核心检索函数 ============
from SL014 import answer_question

# ============ Streamlit UI ============
st.set_page_config(page_title="水利标准检索机器人", layout="wide")
st.title("📑 水利标准检索界面")
st.markdown("请输入你的问题，点击 **查询** 按钮开始检索。")

query = st.text_input("查询问题", placeholder="例如：内蒙古坡面格宾网箱选型")

# —— 主查询按钮 ——
if st.button("查询", type="primary"):
    if not query.strip():
        st.warning("❗ 请先输入查询内容。")
    else:
        with st.spinner("🔍 正在检索，请稍候…"):
            try:
                # 捕获 answer_question 内部的 print / 调试信息，便于展示检索上下文
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    answer = answer_question(query)
                logs = buf.getvalue()

                # --- 检索上下文 ---
                with st.expander("📄 检索到的原始文本（调试用）", expanded=False):
                    st.text(logs if logs.strip() else "（无调试输出）")

                # --- 输出答案 ---
                st.subheader("📝 检索结果")
                st.write(answer)

            except Exception as e:
                st.error(f"🚫 检索过程中出错：{e}")

# ============ 侧边栏 ============
with st.sidebar:
    st.header("关于")
    st.info(
        "本应用基于 Streamlit 构建，后端使用 Chroma 向量数据库 + OpenAI / DeepSeek Chat "
        "模型实现智能问答。Embedding 服务通过本地 `uvicorn` 暴露在端口 8000。"
    )

#cd E:\aaacode\ssll\agr
#streamlit run streamlit_ui.py