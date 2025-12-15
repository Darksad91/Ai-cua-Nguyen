import streamlit as st
from langchain_openai import ChatOpenAI

# 1. Cấu hình trang web
st.set_page_config(page_title="AI Của Tôi", page_icon="🤖")
st.title("Chat với AI Riêng")

# 2. Nhập khóa bí mật (API Key)
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# 3. Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Chào bạn, tôi có thể giúp gì?"}]

# 4. Hiển thị tin nhắn cũ
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Xử lý khi người dùng nhập liệu
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Vui lòng nhập API Key để bắt đầu.")
        st.stop()

    # Lưu tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Gọi AI trả lời (Sử dụng Model)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    response = llm.invoke(prompt)
    msg_content = response.content

    # Lưu và hiện câu trả lời
    st.session_state.messages.append({"role": "assistant", "content": msg_content})
    st.chat_message("assistant").write(msg_content)