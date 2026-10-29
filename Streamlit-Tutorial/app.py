import streamlit as st
import pandas as pd
import numpy as np

st.title("Streamlit Tutorial")
st.write("This is a simple Streamlit tutorial to demonstrate basic functionalities.")
st.text("Let's get started ")


#Conditional logic with widgets
name = st.text_input("Enter your name:")
if st.button("Greet"):
    st.success(f"Hello, {name}!")


#Displaying data and charts
data = pd.DataFrame(np.random.randn(10,2), columns = ['A', 'B'])
st.line_chart(data)
st.bar_chart(data)


#File uploading and caching topics
upload_file = st.file_uploader("Upload a csv file", type="csv")
if upload_file:
    df = pd.read_csv(upload_file)
    st.dataframe(df)

st.title("Text and Markdown")
st.header("This is a header")
st.subheader("This is a subheader")
st.markdown("This is **markdown** text. You can use *italic*, **bold**, `code`, and [links](https://www.streamlit.io/).")
st.code("for i in range(5):print(i)", language="python")

st.text_input("What is your name?")
st.text_area("Tell us about yourself:")
st.number_input("Enter a number:", min_value=0, max_value=100, step=1)
st.slider("Choose a range",0,100)
st.selectbox("Select a fruit", ["Apple", "Banana", "Orange", "Grapes", "Mango"])
st.multiselect("Select your favorite colors", ["Red", "Green", "Blue", "Yellow", "Purple"])
st.radio("Select your gender", ["Male", "Female", "Other"])
st.checkbox("I agree to the terms and conditions")

if st.checkbox("Show details"):
    st.info("Here are some additional details about the application. You can customize this section to provide more information to the users.")

option = st.radio("Choose view", ["Show cart", "Show table"])
if option == "Show cart":
    st.write("Chart would appear here")
else:
    st.write("Table would appear here")
with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submitted = st.form_submit_button("Login")

    if submitted:
        st.success(f"Welcome, {username}!")


st.image("https://static.streamlit.io/examples/dice.jpg", caption="Streamlit Dice Image", use_column_width=True)
st.video("https://www.youtube.com/watch?v=JwSS70SZdyM", format="video/mp4", start_time=0)