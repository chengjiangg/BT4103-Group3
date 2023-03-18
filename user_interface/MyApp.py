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
        ("English", "Mandarin")
    )
    # Setup file upload
    uploaded_file = st.sidebar.file_uploader(
        label="Upload your Excel file.", type=['xlsx'])
    if uploaded_file is not None:
        try:
            # Read excel file
            df = pd.read_excel(uploaded_file)
            # Column header must be strictyl 'text' and 'entity'
            if df.columns[0] != 'text' or df.columns[1] != 'entity':
                st.warning('File format incorrect. Refer to documentation.')
            else:
                return df
        except Exception as e:
            print(e)
    else:
        return None


def entities_info(df, filtered_entities):
    df_filtered = df[df['entity'].isin(filtered_entities)]
    entity_emotion_count = pd.DataFrame(
        df_filtered['emotion'].value_counts(dropna=True, sort=True)).reset_index()
    entity_emotion_count.columns = ['emotion', 'counts']
    entity_stance_count = pd.DataFrame(
        df_filtered['stance'].value_counts(dropna=True, sort=True)).reset_index()
    entity_stance_count.columns = ['stance', 'counts']
    return entity_emotion_count, entity_stance_count


def predict(df):
    prediction = run_prediction(df, model_weight_path, batch_size=1)
    text = prediction[0]
    entity = prediction[1]
    emotion_prob_df = prediction[2]
    stance_prob_df = prediction[3]
    emotion = prediction[4]
    stance = prediction[5]
    prediction_df = pd.DataFrame(list(zip(text, entity, emotion, stance)), columns=[
                                 'text', 'entity', 'emotion', 'stance'])
    prediction_df = pd.concat(
        [prediction_df, emotion_prob_df, stance_prob_df], axis=1)
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
    if 'dataset' in st.session_state and dataset is not None and dataset.equals(st.session_state['dataset']):
        dataset = st.session_state['dataset']
        result = st.session_state['result']
    if dataset is not None and result:
        # Get counts of emotion and stance of prediction
        if "prediction_df" in st.session_state and dataset.equals(st.session_state['dataset']):
            prediction_df = st.session_state['prediction_df']
            emotion_count = st.session_state['emotion_count']
            stance_count = st.session_state['stance_count']
        else:
            prediction_df, emotion_count, stance_count = predict(dataset)
            st.session_state['prediction_df'] = prediction_df
            st.session_state['emotion_count'] = emotion_count
            st.session_state['stance_count'] = stance_count
        st.session_state['dataset'] = dataset
        st.session_state['result'] = result
        entity_list = list(set(prediction_df['entity'].to_list()))
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
        st.metric('Total Labelled', len(dataset.index))
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(emotion_fig, use_container_width=True)
        with col2:
            st.plotly_chart(stance_fig, use_container_width=True)

        # For filter functions
        filter = st.multiselect('Select Entities For Breakdown Statistics:',
                                entity_list, key="my_multi")
        filtered_entities = st.session_state['my_multi']
        entity_emotion_count, entity_stance_count = entities_info(
            prediction_df, filtered_entities)
        entity_emotion_fig = plot_fig(entity_emotion_count, "emotion",
                                      "counts", "Selected Entity Emotion Counts", emotion_labels)
        entity_stance_fig = plot_fig(entity_stance_count, "stance",
                                     "counts", "Selected Entity Stance Counts", stance_labels)

        col11, col22 = st.columns(2)
        with col11:
            st.plotly_chart(entity_emotion_fig, use_container_width=True)
        with col22:
            st.plotly_chart(entity_stance_fig, use_container_width=True)

        st.header('Output')
        st.write(prediction_df)
        linko = export_to_excel(prediction_df)
        st.markdown(linko, unsafe_allow_html=True)
    else:
        st.write("Please upload an excel file to the application")


if __name__ == '__main__':
    main()
