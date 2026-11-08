import streamlit as st

from research_copilot.config import Settings
from research_copilot.rag import (
    answer_question,
    create_chat_model,
    create_embeddings,
    open_vector_store,
)


@st.cache_resource(show_spinner="連接模型和持久化向量索引…")
def initialize(settings: Settings):
    embeddings = create_embeddings(settings)
    vector_store = open_vector_store(settings, embeddings)
    chat_model = create_chat_model(settings)
    return vector_store, chat_model


st.set_page_config(page_title="Data Science Research Copilot", page_icon="🔎")
st.title("Data Science Research Copilot")
st.caption("以公開示範文件作檢索問答。模型回答可能出錯；請查看下方來源並自行核實。")

try:
    settings = Settings.from_env()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

try:
    with st.spinner("正在連接持久化 Chroma 索引…"):
        vector_store, chat_model = initialize(settings)
except RuntimeError as error:
    st.error(str(error))
    st.stop()

st.caption("已連接持久化 Chroma 索引。")

with st.form("question_form"):
    question = st.text_area(
        "你想研究甚麼？",
        placeholder="例如：為甚麼不應反覆根據測試集調整模型？",
        max_chars=1000,
    )
    submitted = st.form_submit_button("搜尋並回答")

if submitted:
    if not question.strip():
        st.warning("請先輸入問題。")
    else:
        with st.spinner("正在檢索文件並生成回答…"):
            answer, sources = answer_question(question, vector_store, chat_model)

        st.subheader("回答")
        st.markdown(answer)

        st.subheader("檢索來源")
        if sources:
            for index, document in enumerate(sources, start=1):
                source = document.metadata.get("source", "unknown")
                title = document.metadata.get("title", source)
                page = document.metadata.get("page")
                location = f", PDF 第 {page} 頁" if page else ""
                with st.expander(f"[S{index}] {title} — {source}{location}"):
                    st.write(document.page_content)
        else:
            st.info("沒有檢索到來源。")
