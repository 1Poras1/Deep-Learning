from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#Creating my prompts
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please answer the user's question."),
        ("user", "QUESTION: {question}"),
    ]
)

# stremlit framework
st.title('Langchain with gemma:2b')
input_text = st.text_input("Enter your query and question do you have?: ")

#ollma model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))