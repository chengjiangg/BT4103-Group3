import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import io
from classes.utils import run_prediction
model_weight_path = "saved_models/model_weights.pth"


def load_dataset():
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
            return df
        except Exception as e:
            print(e)
    else:
        return None


def predict(df):
    prediction = run_prediction(df, model_weight_path, batch_size=1)
    text = prediction[0]
    entity = prediction[1]
    emotion = prediction[2]
    stance = prediction[3]
    prediction_df = pd.DataFrame(list(zip(text, entity, emotion, stance)), columns=[
                                 'text', 'entity', 'emotion', 'stance'])
    # Get counts for emotion
    emotion_count = prediction_df['emotion'].value_counts().rename_axis(
        'emotion').reset_index(name="counts")
    # Get counts for stance
    stance_count = prediction_df['stance'].value_counts().rename_axis(
        'stance').reset_index(name="counts")
    return prediction_df, emotion_count, stance_count


def plot_fig(df, xCol, yCol, title, labels):
    fig = px.bar(df, x=xCol, y=yCol,
                 color=xCol, text=yCol, title=title, labels=labels)
    return fig


def export_to_excel(df):
    towrite = io.BytesIO()
    df.to_excel(
        towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Results.xlsx">Export Result</a>'
    return linko


def main():
    # Title of the app
    st.title("Multi-Lingual Text Classification")
    dataset = load_dataset()
    result = st.sidebar.button("Run")
    if dataset is not None and result:
        # Get counts of emotion and stance of prediction
        prediction_df, emotion_count, stance_count = predict(dataset)
        # Plot bar chart for emotion
        emotion_labels = {"emotion": "Emotion", "counts": "Counts"}
        emotion_fig = plot_fig(emotion_count, "emotion",
                               "counts", "Emotion Counts", emotion_labels)
        # Plot bar chart for stance
        stance_labels = {"stance": "Stance", "counts": "Counts"}
        stance_fig = plot_fig(stance_count, "stance",
                              "counts", "Stance Counts", stance_labels)
        # Start of Output
        st.header('Label Statistics')
        st.plotly_chart(emotion_fig)
        st.plotly_chart(stance_fig)
        st.header('Output')
        st.write(prediction_df)
        linko = export_to_excel(prediction_df)
        st.markdown(linko, unsafe_allow_html=True)
    else:
        st.write("Please upload an excel file to the application")


if __name__ == '__main__':
    main()
