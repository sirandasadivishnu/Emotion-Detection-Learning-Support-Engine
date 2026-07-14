import streamlit as st
import pandas as pd
import os
import json
from services.bilstm_model import BiLSTMClassifier
from services.bert_model import BERTClassifier
from services.mixed_emotion import MixedEmotionDetector
from services.prediction_schema import PredictionSchema
from services.gemini_service import get_gemini_response
from services.interaction_logger import InteractionLogger
import plotly.express as px

# -------------------------------------------------------
# Page Config
# -------------------------------------------------------

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

# -------------------------------------------------------
# Cached Model Loading
# -------------------------------------------------------


@st.cache_resource
def load_models():
    bilstm = BiLSTMClassifier()

    try:
        bert = BERTClassifier()
    except:
        bert = None

    return bilstm, bert


bilstm_model, bert_model = load_models()
logger = InteractionLogger()

class HistoryLoader:
    FILE_PATH = "data/interaction_history.csv"

    @staticmethod
    @st.cache_data
    def _load_cached(file_mtime):
        if not os.path.exists(HistoryLoader.FILE_PATH):
            return pd.DataFrame(columns=[
                "timestamp",
                "field",
                "problem",
                "cleaned_text",
                "emotion",
                "confidence",
                "primary_emotion",
                "primary_confidence",
                "secondary_emotions",
                "bilstm_scores",
                "bert_emotion",
                "bert_confidence",
                "bert_scores",
                "ai_enabled",
                "response"
            ])

        history_df = pd.read_csv(HistoryLoader.FILE_PATH)

        if "primary_emotion" not in history_df.columns and "emotion" in history_df.columns:
            history_df["primary_emotion"] = history_df["emotion"]

        if "primary_confidence" not in history_df.columns and "confidence" in history_df.columns:
            history_df["primary_confidence"] = history_df["confidence"]

        if "emotion" not in history_df.columns and "primary_emotion" in history_df.columns:
            history_df["emotion"] = history_df["primary_emotion"]

        if "confidence" not in history_df.columns and "primary_confidence" in history_df.columns:
            history_df["confidence"] = history_df["primary_confidence"]

        if "cleaned_text" not in history_df.columns:
            history_df["cleaned_text"] = ""

        for column, default in [
            ("secondary_emotions", []),
            ("bilstm_scores", {}),
            ("bert_scores", {})
        ]:
            if column in history_df.columns:
                history_df[column] = history_df[column].apply(
                    lambda value, default_value=default: default_value if pd.isna(value) or value == "" else json.loads(value)
                )
            else:
                history_df[column] = [default for _ in range(len(history_df))]

        return history_df

    @staticmethod
    def load():
        file_mtime = os.path.getmtime(HistoryLoader.FILE_PATH) if os.path.exists(HistoryLoader.FILE_PATH) else 0
        return HistoryLoader._load_cached(file_mtime)

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🎓 AI Learning Support Assistant")

st.markdown(
"""
Personalized learning assistant powered by

- BiLSTM
- BERT
- Gemini
"""
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:
    st.header("📊 Dashboard")

    st.success("Models Loaded")

    history = HistoryLoader.load()

    st.metric(
        "Total Sessions",
        len(history)
    )

    if st.button("Clear History"):
        if os.path.exists(HistoryLoader.FILE_PATH):
            os.remove(HistoryLoader.FILE_PATH)
        HistoryLoader._load_cached.clear()
        st.rerun()

    st.markdown("---")

    st.subheader("📚 Previous Sessions")

    history_placeholder = "-- Select Previous Session --"
    selected_history_index = None

    if not history.empty:
        selected_history_index = st.selectbox(
            "Choose a session",
            [history_placeholder] + history.index.tolist(),
            format_func=lambda option: (
                history_placeholder if option == history_placeholder else (
                    f"🕒 {pd.to_datetime(history.loc[option, 'timestamp']).strftime('%d %b %Y %H:%M')} | "
                    f"{history.loc[option, 'field']} | {history.loc[option, 'problem']}"
                )
            )
        )

selected_row = None
if selected_history_index is not None and selected_history_index != history_placeholder and not history.empty:
    selected_row = history.loc[selected_history_index]


def render_prediction_view(result, bert_result, ai_response, ai_title="🎓 AI Learning Support"):
    st.markdown("## 🧠 Emotion Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Primary Emotion",
            result["primary_emotion"]
        )
        st.metric(
            "Confidence",
            f"{result['primary_confidence']:.1%}"
        )

    with col2:
        if len(result["secondary_emotions"]) == 0:
            st.info("No secondary emotion detected.")
        else:
            st.write("Secondary Emotions")
            for emotion in result["secondary_emotions"]:
                st.write(f"{emotion['emotion']} ({emotion['confidence']:.1%})")

    st.markdown("---")

    st.subheader("BiLSTM Scores")

    score_df = pd.DataFrame(
        list(result["scores"].items()),
        columns=["Emotion", "Probability"]
    )

    fig = px.bar(
        score_df,
        x="Emotion",
        y="Probability",
        title="BiLSTM Emotion Probabilities"
    )

    st.plotly_chart(fig, width="stretch")

    if bert_result is not None:
        st.markdown("---")
        st.subheader("🤖 BERT Prediction")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Emotion", bert_result["emotion"])

        with c2:
            st.metric(
                "Confidence",
                f"{bert_result['confidence']:.1%}"
            )

        bert_df = pd.DataFrame(
            list(bert_result["scores"].items()),
            columns=["Emotion", "Probability"]
        )

        st.dataframe(bert_df, width="stretch")

    st.markdown("---")
    st.subheader(ai_title)
    st.write(ai_response)


def render_history_view(row):
    secondary_emotions = row.get("secondary_emotions", [])
    bilstm_scores = row.get("bilstm_scores", {})
    bert_scores = row.get("bert_scores", {})

    if not isinstance(secondary_emotions, list):
        secondary_emotions = []

    if not isinstance(bilstm_scores, dict):
        bilstm_scores = {}

    if not isinstance(bert_scores, dict):
        bert_scores = {}

    result = {
        "primary_emotion": row.get("primary_emotion", row.get("emotion", "Unknown")),
        "primary_confidence": float(row.get("primary_confidence", row.get("confidence", 0.0))),
        "secondary_emotions": secondary_emotions,
        "scores": bilstm_scores
    }

    bert_result = None
    if pd.notna(row.get("bert_emotion")) and pd.notna(row.get("bert_confidence")):
        bert_result = {
            "emotion": row["bert_emotion"],
            "confidence": float(row["bert_confidence"]),
            "scores": bert_scores
        }

    st.markdown("---")
    st.header("📜 Previous Interaction")
    st.write("Field:", row["field"])
    st.write("Problem:", row["problem"])

    render_prediction_view(result, bert_result, row["response"])

# -------------------------------------------------------
# Layout
# -------------------------------------------------------

left, right = st.columns([2, 1])

# -------------------------------------------------------
# Left Panel
# -------------------------------------------------------

with left:
    st.subheader("📚 Learning Challenge")

    field = st.selectbox(
        "What are you studying?",
        [
            "Computer Science",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Engineering",
            "Business",
            "Literature",
            "History",
            "Psychology",
            "Other"
        ]
    )

    problem = st.text_area(
        "Describe your learning problem",
        placeholder="Example: I don't understand Binary Search.",
        height=180
    )

# -------------------------------------------------------
# Right Panel
# -------------------------------------------------------

with right:
    st.subheader("⚙ Settings")

    use_ai = st.checkbox(
        "Use Gemini AI",
        value=True
    )

    save_csv = st.checkbox(
        "Save Prediction",
        value=True
    )

    show_details = st.checkbox(
        "Show Analysis Details",
        value=False
    )

# -------------------------------------------------------
# Analyze Button
# -------------------------------------------------------

col1, col2 = st.columns(2)

analyze = col1.button(
    "🚀 Get AI Learning Help",
    width="stretch"
)

clear = col2.button(
    "🗑 Clear",
    width="stretch"
)

if clear:
    st.rerun()

current_result = None
current_bert_result = None
current_ai_response = None

if analyze and selected_row is None:
    if problem.strip() == "":
        st.warning("Please enter your problem.")
    else:
        with st.spinner("Analyzing your learning challenge..."):
            bilstm_result = bilstm_model.predict(problem)

            if bert_model is not None:
                bert_result = bert_model.predict(problem)
            else:
                bert_result = None

            detector = MixedEmotionDetector()

            mixed = detector.detect(bilstm_result)

            result = PredictionSchema.build(
                model_name="BiLSTM",
                cleaned_text=mixed["cleaned_text"],
                primary_emotion=mixed["primary_emotion"],
                primary_confidence=mixed["primary_confidence"],
                secondary_emotions=mixed["secondary_emotions"],
                scores=mixed["scores"]
            )

            if show_details:
                st.info(f"Cleaned Text : {result['cleaned_text']}")
            if use_ai:
                ai_response = get_gemini_response(
                    field,
                    problem,
                    result["primary_emotion"],
                    result["primary_confidence"]
                )
            else:
                ai_response = (
                    "🤖 AI Response Disabled\n\n"
                    "Enable Gemini AI to receive personalized learning guidance."
                )

            # Save
            if save_csv:
                logger.save(
                    prediction={
                        "field": field,
                        "problem": problem,
                        "cleaned_text": result["cleaned_text"],
                        "primary_emotion": result["primary_emotion"],
                        "primary_confidence": result["primary_confidence"],
                        "secondary_emotions": json.dumps(result["secondary_emotions"]),
                        "bilstm_scores": json.dumps(result["scores"]),
                        "bert_emotion": None if bert_result is None else bert_result["emotion"],
                        "bert_confidence": None if bert_result is None else bert_result["confidence"],
                        "bert_scores": None if bert_result is None else json.dumps(bert_result["scores"]),
                        "ai_enabled": use_ai,
                        "ai_response": ai_response
                    }
                )

                st.success("Prediction saved successfully.")

            if save_csv:
                with open("data/interaction_history.csv", "rb") as file:
                    st.download_button(
                        "⬇ Download Interaction History",
                        data=file,
                        file_name="interaction_history.csv",
                        mime="text/csv"
                    )

            current_result = result
            current_bert_result = bert_result
            current_ai_response = ai_response

if selected_row is not None:
    render_history_view(selected_row)
elif current_result is not None:
    render_prediction_view(current_result, current_bert_result, current_ai_response)

# -------------------------------------------------------
# Analytics Dashboard
# -------------------------------------------------------
history_df = HistoryLoader.load()

if not history_df.empty:
    st.markdown("---")
    st.header("📊 Learning Analytics")

    tab1, tab2, tab3 = st.tabs([
        "😊 Emotions",
        "📚 Fields",
        "📈 Summary"
    ])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            emotion_counts = (
                history_df["primary_emotion"]
                .value_counts()
                .reset_index()
            )

            emotion_counts.columns = ["Emotion", "Count"]

            fig1 = px.pie(
                emotion_counts,
                names="Emotion",
                values="Count",
                title="Emotion Distribution"
            )

            st.plotly_chart(fig1, width="stretch")

        with col2:
            history_df["Session"] = range(1, len(history_df) + 1)

            fig2 = px.line(
                history_df,
                x="Session",
                y="primary_confidence",
                color="primary_emotion",
                markers=True,
                title="Confidence Timeline"
            )

            st.plotly_chart(fig2, width="stretch")

    with tab2:
        field_counts = (
            history_df
            .groupby(["field", "primary_emotion"])
            .size()
            .reset_index(name="Count")
        )

        fig3 = px.bar(
            field_counts,
            x="field",
            y="Count",
            color="primary_emotion",
            title="Emotion by Study Field"
        )

        st.plotly_chart(fig3, width="stretch")

    with tab3:
        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Sessions",
            len(history_df)
        )

        c2.metric(
            "Average Confidence",
            f"{history_df['primary_confidence'].mean():.1%}"
        )

        c3.metric(
            "Most Common Emotion",
            history_df["primary_emotion"].mode()[0]
        )

        st.dataframe(
            history_df,
            width="stretch"
        )


