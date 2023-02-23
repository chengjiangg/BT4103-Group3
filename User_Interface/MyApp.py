import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the app
st.title("Multi-Lingual Text Classification")

# Add a sidebar
st.sidebar.subheader("Classification Settings")

# Select language
langauge = st.sidebar.selectbox(
    "Select language",
    ("English", "Chinese")
)

# Setup file upload
uploaded_file = st.sidebar.file_uploader(
    label="Upload your Excel file.", type=['xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.write(df)
        count_df = df['emotion'].value_counts().rename_axis(
            'emotion').reset_index(name="counts")
        fig = px.bar(count_df, x="emotion", y="counts",
                     color="emotion", text="counts", title="Emotion Counts", labels={
                         "emotion": "Emotion",
                         "counts": "Counts",
                     })
        st.plotly_chart(fig)
    except Exception as e:
        print(e)
else:
    st.write("Please upload an excel file to the application")
