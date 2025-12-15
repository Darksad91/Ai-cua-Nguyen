import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# --- CẤU HÌNH ---
st.set_page_config(page_title="AI Đọc Tài Liệu (Gemini)", page_icon="🤖")
st.header("🤖 Chat với tài liệu (Dùng Gemini Free)")

# --- SIDEBAR ---
with st.sidebar:
    st.title("Cài đặt")
    google_api_key = st.text_input("Nhập Google Gemini API Key:", type="password")
    uploaded_file = st.file_uploader("Tải lên file PDF", type="pdf")
    process_button = st.button("Xử lý dữ liệu")

# --- HÀM CHÍNH ---
def main():
    if uploaded_file and process_button:
        if not google_api_key:
            st.error("⚠️ Chưa nhập API Key.")
            return

        with st.spinner("Đang đọc tài liệu..."):
            # 1. Đọc PDF
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # 2. Cắt nhỏ văn bản
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)

            # 3. Tạo Vector (Dùng Google Embeddings)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            st.session_state.vector_store = vector_store
            st.success("✅ Xong! Hãy hỏi đi.")

    # --- CHAT ---
    query = st.text_input("Câu hỏi của bạn:")
    if query:
        if "vector_store" not in st.session_state:
            st.warning("⚠️ Hãy upload file trước.")
        elif not google_api_key:
            st.warning("⚠️ Thiếu API Key.")
        else:
            # 4. Tìm kiếm & Trả lời (Dùng Gemini Pro)
            docs = st.session_state.vector_store.similarity_search(query)
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            with st.spinner("Gemini đang nghĩ..."):
                response = chain.run(input_documents=docs, question=query)
                st.write(response)

if __name__ == '__main__':
    main()