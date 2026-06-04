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
STARTUP_TIMEOUT = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ 启动 Embedding API ============
def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0

def start_embedding_api():
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

    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT:
        if is_port_open(EMB_HOST, EMB_PORT):
            logger.info("Embedding API 服务已就绪")
            return
        time.sleep(0.5)
    logger.warning(f"启动超时：{EMB_HOST}:{EMB_PORT} 在 {STARTUP_TIMEOUT}s 内未响应")

# 初始化 API
if "api_started" not in st.session_state:
    start_embedding_api()
    st.session_state.api_started = True

# ============ 导入主函数 ============
from SL_add import answer_question

# ============ 初始化状态 ============
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============ 页面 UI ============
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
                    answer = answer_question(query, chat_history=st.session_state.chat_history)
                logs = buf.getvalue()

                with st.expander("📄 检索到的原始文本（调试用）", expanded=False):
                    st.text(logs or "（无调试输出）")

                st.subheader("📝 检索结果")
                st.write(answer)

                # 更新历史
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(
                    "🚫 检索过程中出错："
                    f"{e}\n\n"
                    "请确认 Embedding API 是否已正确启动，"
                )

# 展示历史对话记录
with st.expander("🗂️ 多轮对话历史", expanded=False):
    for i, msg in enumerate(st.session_state.chat_history):
        role = "用户" if msg["role"] == "user" else "机器人"
        st.markdown(f"**{role}：** {msg['content']}")

# 侧边栏
with st.sidebar:
    st.header("关于")
    st.info(
        "本应用基于 Streamlit + Chroma 向量数据库 + OpenAI/DeepSeek Chat 模型。\n\n"
        f"Embedding 服务地址：{EMB_HOST}:{EMB_PORT}"
    )
    if st.button("🧹 清空对话历史"):
        st.session_state.chat_history = []
        st.success("已清空对话历史")


#cd E:\aaacode\ssll\agr
#streamlit run streamlit_add.py