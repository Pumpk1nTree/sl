import streamlit as st
import subprocess
import socket
import time
import io
import contextlib
import logging

# ============ 配置 ============
EMB_HOST = "127.0.0.1"
EMB_PORT = 8000
STARTUP_TIMEOUT = 10  # 最大等待秒数

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 前置：确保本地 Embedding API 已启动 ============
def is_port_open(host: str, port: int) -> bool:
    """检查本地端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0

def start_embedding_api():
    """若端口未被占用，则后台启动 uvicorn embedding_api:app，并等待健康就绪"""
    if is_port_open(EMB_HOST, EMB_PORT):
        logger.info(f"Embedding API 已在 {EMB_HOST}:{EMB_PORT} 上运行")
        return

    cmd = ["uvicorn", "embedding_api:app", "--host", "0.0.0.0", f"--port={EMB_PORT}"]
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("已触发 Embedding API 启动，等待服务就绪…")
    except Exception as ex:
        logger.error(f"启动 Embedding API 失败：{ex}")
        return

    # 等待端口开放，最长等待 STARTUP_TIMEOUT 秒
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT:
        if is_port_open(EMB_HOST, EMB_PORT):
            logger.info("Embedding API 服务已就绪")
            return
        time.sleep(0.5)

    logger.warning(f"启动超时：{EMB_HOST}:{EMB_PORT} 在 {STARTUP_TIMEOUT}s 内未响应")

# 仅首次加载时启动
if "api_started" not in st.session_state:
    start_embedding_api()
    st.session_state.api_started = True

# ============ 导入核心检索函数 ============
from SL014 import answer_question  # 确保 embedding_api:app 中已注册 /embeddings 路由

# ============ Streamlit UI ============
st.set_page_config(page_title="水利标准检索机器人", layout="wide")
st.title("📑 水利标准检索界面")
st.markdown("请输入你的问题，点击 **查询** 按钮开始检索。")

query = st.text_input("查询问题", placeholder="例如：内蒙古坡面格宾网箱选型")

if st.button("查询", type="primary"):
    if not query.strip():
        st.warning("❗ 请先输入查询内容。")
    else:
        with st.spinner("🔍 正在检索，请稍候…"):
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    answer = answer_question(query)
                logs = buf.getvalue()

                with st.expander("📄 检索到的原始文本（调试用）", expanded=False):
                    st.text(logs or "（无调试输出）")

                st.subheader("📝 检索结果")
                st.write(answer)

            except Exception as e:
                st.error(
                    "🚫 检索过程中出错：" 
                    f"{e}\n\n"
                    "请确认 Embedding API 是否已正确启动，"
                    f"可尝试手动运行：\n\n"
                    "```bash\n"
                    "uvicorn embedding_api:app --reload --host 127.0.0.1 --port 8000\n"
                    "```"
                )

with st.sidebar:
    st.header("关于")
    st.info(
        "本应用基于 Streamlit + Chroma 向量数据库 + OpenAI/DeepSeek Chat 模型。\n\n"
        f"Embedding 服务地址：{EMB_HOST}:{EMB_PORT}"
    )

#cd E:\aaacode\ssll\agr
#streamlit run streamlit_ui_2.py