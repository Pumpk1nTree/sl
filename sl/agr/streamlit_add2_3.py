import streamlit as st
import subprocess
import socket
import time
import io
import contextlib
import logging

# ============ 页面配置 ===========
st.set_page_config(page_title="水利标准检索机器人", layout="wide")

# ============ Embedding API 配置 ===========
EMB_HOST = "127.0.0.1"
EMB_PORT = 8000
STARTUP_TIMEOUT = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# 仅首次加载时启动
if "api_started" not in st.session_state:
    start_embedding_api()
    st.session_state.api_started = True

# ============ 登录状态 ===========
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 欢迎使用 水利标准检索机器人")
    st.markdown("""
    **产品介绍：**  
    随着人工智能技术的快速发展，大语言模型（LLM）已在政务、法务、工程等多个领域得到广泛应用。水利行业\n
    作为国家基础设施建设和生态保护的重要领域，涉及复杂的技术标准和法律法规体系，涵盖国家标准、行业标准、\n
    地方标准以及配套的法律法规文件。如何高效理解、查询和应用这些规范性文件，成为水利工程设计、审查、监管\n
    与运维中的核心难题。\n
    本项目旨在构建一个基于向量检索与大语言模型（LLM）结合的智能问答系统，用于对国家及地方水利标准与技术\n
    规范进行语义理解与精准回答。随着水利工程建设日益复杂，标准文件数量庞大、结构多样，传统基于关键词的检\n
    索手段难以满足工程技术人员对“语义相关”“结构化解析”“多源融合”的实际需求。本项目集成了300余项水利水电\n
    标准以及200余项水利相关法律法规。项目构建了一个智能问答系统，专门服务于国家及地方水利标准、法规和技术\n
    文档的语义查询任务。系统通过文本向量化检索（Chroma 向量数据.库）结合大语言模型（DeepSeek Chat），实\n
    现对用户自然语言提问的智能解答，适用于政府部门、工程设计院及科研单位的法规辅助检索工作，准确率较高。
  
    """)
    if st.button("开始查询"):
        st.session_state.logged_in = True
    # 停止执行，等待按钮触发的重跑
    st.stop()

# ============ 登录后主界面 ===========
from SL_add2_3 import answer_question, PROVINCE_LIST

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("📑 水利标准检索界面")
st.markdown("请输入你的问题，并可手动勾选省份范围，点击 **查询** 开始检索。\n"
            "(注意：如果未选择具体省份请在问题中明确指出具体地址，未勾选省份并且问题中没有明确的地点信息则按国家标准查询)")

selected_provinces = st.multiselect(
    "🔍 请选择要检索的省份（多选，留空则只检索国家标准）",
    options=PROVINCE_LIST,
    help="如选择，则在所选省份集合和国家标准集合中检索；留空则仅检索国家标准。"
)

query = st.text_input("查询问题", placeholder="例如：内蒙古坡面格宾网箱选型")

if st.button("查询", type="primary"):
    if not query.strip():
        st.warning("❗ 请先输入查询内容。")
    else:
        with st.spinner("🔍 正在检索，请稍候…"):
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    answer = answer_question(
                        query,
                        selected_provinces=selected_provinces,
                        chat_history=st.session_state.chat_history
                    )
                logs = buf.getvalue()

                with st.expander("📄 检索到的原始文本（调试用）", expanded=False):
                    st.text(logs or "（无调试输出）")

                st.subheader("📝 检索结果")
                st.write(answer)

                # 更新历史
                st.session_state.chat_history.append({"role": "user", "content": query})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"检索出错：{e}")

# 多轮对话历史
with st.expander("🗂️ 多轮对话历史", expanded=False):
    for msg in st.session_state.chat_history:
        role = "用户" if msg["role"] == "user" else "机器人"
        st.markdown(f"**{role}：** {msg['content']}")

# 侧边栏：清空历史
with st.sidebar:
    if st.button("🧹 清空多轮对话历史"):
        st.session_state.chat_history = []
        st.success("✅ 已清空多轮对话历史")

#cd E:\aaacode\ssll\agr
#streamlit run streamlit_add2_3.py