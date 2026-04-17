from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from ChatModels.chatmodel_huggingface_local import model



load_dotenv()
st.header("Research Assistant")


paper_input = st.selectbox("Select Research Paper", ["Attention Is All You Need",
                                                     "Word2Vec","Transformer","BERT",
                                                     "Few-Shot Learning with GPT-3"])

style_input = st.selectbox("Select Response Style",['Beginner-friendly', 'Technical',
                                                    "code-oriented", "concise", "detailed"])

length_input = st.selectbox("Select Response Length", ['Short(1-2 paragraphs)',
                                                       'medium(3-5 paragraphs)',
                                                       'long(6+ paragraphs)'])

# st.text_input("Enter your message/prompt:", key="input")

template = load_prompt('Prompts/template.json')





if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input

    })
    st.write(result.content)