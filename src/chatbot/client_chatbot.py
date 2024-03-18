import streamlit as st
import requests
import json

url = "http://0.0.0.0:5000/chat"
headers = {'Content-Type': 'application/json'}

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello, How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    data = {'question': prompt}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    answ = response.json()['answer']

    st.session_state.messages.append({"role": "assistant", "content": answ})
    st.chat_message("assistant").write(answ)