import streamlit as st
from my_code.inference_from_user_prompt import inference_from_input

st.title("Transformer Language Model Demo")

user_text = st.text_input("Enter your prompt:", "")

if st.button("Generate"):
    if not user_text.strip():
        st.warning("Please enter a prompt first.")
    else:
        output = inference_from_input(user_text)
        st.subheader("Generated text:")
        st.write(output)
