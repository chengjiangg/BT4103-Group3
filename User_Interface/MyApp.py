import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import io

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
        # Read excel file
        df = pd.read_excel(uploaded_file)
        # Get counts for emotion
        emotion_count = df['emotion'].value_counts().rename_axis(
            'emotion').reset_index(name="counts")
        # Get counts for stance
        stance_count = df['stance'].value_counts().rename_axis(
            'stance').reset_index(name="counts")
        # Plot counts for emotion
        emotion_fig = px.bar(emotion_count, x="emotion", y="counts",
                             color="emotion", text="counts", title="Emotion Counts", labels={
                                 "emotion": "Emotion",
                                 "counts": "Counts",
                             })
        # Plot counts for stance
        stance_fig = px.bar(stance_count, x="stance", y="counts",
                            color="stance", text="counts", title="Stance Counts", labels={
                                "stance": "Stance",
                                "counts": "Counts",
                            })
        st.header('Label Statistics')
        st.plotly_chart(emotion_fig)
        st.plotly_chart(stance_fig)

        st.header('Output')
        st.write(df)
        # Export results to excel
        towrite = io.BytesIO()
        downloaded_file = emotion_count.to_excel(
            towrite, encoding='utf-8', index=False, header=True)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Results.xlsx">Export Result</a>'
        st.markdown(linko, unsafe_allow_html=True)
    except Exception as e:
        print(e)
else:
    st.write("Please upload an excel file to the application")
