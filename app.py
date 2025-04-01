import streamlit as st
import time
from gemini_api_key import generate_response  # ✅ Uses fine-tuned model
from qa import retrieve_context  # ✅ Retrieves content from PDFs
from streamlit_extras.let_it_rain import rain

# 🎨 Streamlit Page Configuration
st.set_page_config(page_title="Amazon Q&A Chatbot", page_icon="🚲", layout="wide")

# 🏢 Amazon-style Banner
st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)

# 🔹 User Role Selection
st.sidebar.title("🔹 User Role Selection")
user_role = st.sidebar.radio("Choose a role:", ["Buyer", "Seller", "Admin"])

# 🎯 Chat Section
st.title("🛒 Amazon Policy Q&A Chatbot")
st.write("🔍 Ask me anything about Amazon policies!")

query = st.text_input("💬 Type your question here and press Enter:")
submit = st.button("Ask 🚀")

# 💡 Rain Effect for Fun
rain(emoji="💵", font_size=20, falling_speed=3, animation_length="infinite")

if submit and query:
    with st.spinner("Thinking... 🤔"):
        time.sleep(1)  # Simulating API delay
        retrieved_docs = retrieve_context(query)
        response = generate_response(query, retrieved_docs, user_role)

    with st.expander("📄 Relevant Policy Sections", expanded=False):
        for doc in retrieved_docs:
            st.write(f"📄 {doc}")

    st.markdown(f"<div class='response-bubble'>🤖 AI: {response}</div>", unsafe_allow_html=True)

# 📢 Footer
st.markdown("---")
st.markdown("🔹 **Amazon Policy Q&A Chatbot** - Built with ❤️ using Streamlit & GPT-Neo")
