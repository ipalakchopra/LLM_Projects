import requests
import streamlit as st

def get_llama_response(input_text, type):
    response = requests.post(f"http://localhost:8000/{type}/invoke",
    json={'input':{'topic':input_text}})

    return response.json()['output']

st.title('API Demo')
input_text_essay=st.text_input("write essay on")
input_text_poem=st.text_input("write poem on")

if input_text_essay:
    st.write(get_llama_response(input_text_essay,"essay"))

if input_text_poem:
    st.write(get_llama_response(input_text_poem,"poem"))
