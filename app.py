import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🕵️‍♀️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS (POSH + CUTE)
# =====================================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3 {
    color: #ff7aa2;
}

.card {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 6px 16px rgba(0,0,0,0.2);
    color: #333333;
    margin-bottom: 20px;
}

.metric-card {
    background: #1f3b4d;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.footer {
    text-align: center;
    margin-top: 30px;
    font-size: 14px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("fake_job_postings.csv")
    df["title"] = df["title"].astype(str).fillna("")
    df["description"] = df["description"].astype(str).fillna("")
    return df

@st.cache_resource
def load_model():
    model = pickle.load(open("final_best_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, tfidf

df = load_data()
model, tfidf = load_model()

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌸 Navigation")
page = st.sidebar.radio(
    "Go to section",
    [
        "🏠 Project Overview",
        "📊 Dataset Overview",
        "🧹 Data Cleaning",
        "📈 Exploratory Data Analysis",
        "🤖 Model Comparison",
        "🏆 Final Best Model",
        "📈 PR & ROC Curves",
        "🔮 Live Prediction"
    ]
)

# =====================================================
# PROJECT OVERVIEW
# =====================================================
if page == "🏠 Project Overview":
    st.title("🕵️ Fake Job Posting Detection")
    st.markdown("""
    <div class="card">
    <h3>✨ About the Project</h3>
    <p>
    This project detects <b>fraudulent job postings</b> using
    <b>Machine Learning</b> and <b>Natural Language Processing (NLP)</b>.
    </p>
    <ul>
        <li>Text preprocessing using TF-IDF</li>
        <li>Multiple classifiers trained & compared</li>
        <li>Best model selected using F1-score</li>
        <li>Interactive Streamlit deployment</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DATASET OVERVIEW
# =====================================================
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Features", df.shape[1])
    col3.metric("Fake Jobs", int(df["fraudulent"].sum()))

    st.markdown('<div class="card">Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head())

# =====================================================
# DATA CLEANING
# =====================================================
elif page == "🧹 Data Cleaning":
    st.title("🧹 Data Cleaning")

    st.markdown("""
    <div class="card">
    <ul>
        <li>Missing text values filled with empty strings</li>
        <li>All text converted to string type</li>
        <li>High-null and irrelevant columns dropped</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Remaining Null Values")
    st.dataframe(df.isnull().sum())

# =====================================================
# EDA
# =====================================================
elif page == "📈 Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis")

    fig, ax = plt.subplots()
    sns.countplot(x="fraudulent", data=df, ax=ax)
    ax.set_xticklabels(["Real", "Fake"])
    ax.set_xlabel("Job Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.info("Dataset is imbalanced → F1-score is used for evaluation")

# =====================================================
# MODEL COMPARISON
# =====================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")

    model_results = {
        "Logistic Regression": 0.82,
        "Linear SVM": 0.86,
        "SGD Classifier": 0.84,
        "Naive Bayes": 0.78
    }

    results_df = pd.DataFrame(
        model_results.items(),
        columns=["Model", "F1 Score"]
    ).sort_values(by="F1 Score", ascending=False)

    st.dataframe(results_df)

    fig, ax = plt.subplots()
    ax.bar(results_df["Model"], results_df["F1 Score"])
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=30)
    st.pyplot(fig)

# =====================================================
# FINAL MODEL
# =====================================================
elif page == "🏆 Final Best Model":
    st.title("🏆 Final Best Model")

    st.success("✅ Final Selected Model: **Linear SVM**")

    text_data = (df["title"] + " " + df["description"]).tolist()
    X_all = tfidf.transform(text_data)
    y_true = df["fraudulent"].values
    y_pred = model.predict(X_all)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# =====================================================
# PR & ROC CURVES
# =====================================================
elif page == "📈 PR & ROC Curves":
    st.title("📈 Precision-Recall & ROC Curves")

    text_data = (df["title"] + " " + df["description"]).tolist()
    X_all = tfidf.transform(text_data)
    y_true = df["fraudulent"].values
    scores = model.decision_function(X_all)

    precision, recall, _ = precision_recall_curve(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig1, ax1 = plt.subplots()
    ax1.plot(recall, precision)
    ax1.set_title("Precision-Recall Curve")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0,1], [0,1], linestyle="--")
    ax2.legend()
    ax2.set_title("ROC Curve")
    st.pyplot(fig2)

# =====================================================
# LIVE PREDICTION
# =====================================================
elif page == "🔮 Live Prediction":
    st.title("🔮 Live Fake Job Prediction")

    job_text = st.text_area("Paste Job Description", height=220)

    if st.button("Predict"):
        if job_text.strip() == "":
            st.warning("Please enter job description")
        else:
            vec = tfidf.transform([str(job_text)])
            pred = model.predict(vec)[0]

            if pred == 1:
                st.error("🚨 This job is likely FAKE")
            else:
                st.success("✅ This job looks REAL")

st.markdown('<div class="footer">💖 End-to-End ML & NLP Project | Streamlit App</div>', unsafe_allow_html=True)
