# FeelFinder 🧠 - AI-Powered Text Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-~89%25-brightgreen.svg)]()

**FeelFinder** is a portfolio-grade, web application built for predicting human emotions expressed in written text or paragraphs. Powered by a pre-trained **Logistic Regression** model and **CountVectorizer** feature extraction pipeline, FeelFinder delivers real-time sentiment analysis across six distinct emotion classes with ~89% test accuracy.

---

## 🌟 Key Features

- **Instant Emotion Prediction**: High-accuracy prediction for any input sentence, paragraph, or phrase.
- **Full Confidence Breakdown**: Calculates and displays probability confidence percentages across all 6 emotion classes (`model.predict_proba()`).
- **Interactive Modern Dashboard**: Styled Streamlit interface featuring glassmorphic design cards, dynamic color accents per emotion, responsive progress distribution bars, and quick-test example buttons.
- **Robust Machine Learning Engine**: Completely decoupled ML logic (`emotion_model.py`) from frontend UI (`app.py`) ensuring clean, modular, and maintainable architecture.
- **Pre-trained Artifact Validation**: Validates existence of serialized model files (`LogisticRegression_model.pkl` and `count_vectorizer.pkl`) prior to inference. Zero re-training or fitting on user input.
- **Input Guardrails & Security**: Graceful error handling for empty inputs, special characters, and out-of-vocabulary terms. User input is never evaluated as code.

---

## 🏷️ Emotion Categories & Mappings

The underlying model predicts numeric class labels from `1` to `6`, mapped to standard emotion categories:

| Numeric Label | Emotion Name | Emoji | Visual Accent | Description |
| :---: | :---: | :---: | :---: | :--- |
| **1** | **Sadness** | 😢 | `#3B82F6` (Blue) | Feelings of sorrow, unhappiness, or despair |
| **2** | **Anger** | 😠 | `#EF4444` (Red) | Expressions of annoyance, rage, or hostility |
| **3** | **Love** | ❤️ | `#EC4899` (Pink) | Deep affection, warmth, or fondness |
| **4** | **Surprise** | 😲 | `#F59E0B` (Amber) | Astonishment, wonder, or unexpected events |
| **5** | **Fear** | 😨 | `#8B5CF6` (Purple) | Apprehension, anxiety, or terror |
| **6** | **Joy** | 😊 | `#10B981` (Emerald) | Happiness, delight, or positive satisfaction |

---

## 🤖 How the Machine Learning Model Works

The FeelFinder classification pipeline utilizes classical Natural Language Processing (NLP) techniques:

```
User Input Text
      ↓
CountVectorizer.transform() (Bag-of-Words Feature Matrix)
      ↓
LogisticRegression.predict() & predict_proba()
      ↓
Numerical Label Mapping (1..6 → Emotion & Emoji)
      ↓
Streamlit Dashboard Render
```

### 1. NLP Feature Extraction (`CountVectorizer`)
- **Concept**: Converts raw textual data into a numerical token frequency matrix based on the vocabulary learned during model training.
- **Inference Requirement**: Raw text must undergo transformation via `count_vectorizer.pkl` using `.transform()` (never `.fit()` or `.fit_transform()`).

### 2. Classification Model (`LogisticRegression`)
- **Concept**: Multi-class Logistic Regression model trained using scikit-learn. It applies softmax multinomial probability estimations to compute confidence scores for each class.
- **Accuracy**: Achieved **~89% test accuracy** on the emotion dataset benchmark.

---

## 📂 Project Structure

```
FeelFinder/
│
├── app.py                         # Streamlit dashboard interface & interactive UI
├── emotion_model.py               # Core ML module (model loading, prediction, probability calculation)
├── LogisticRegression_model.pkl   # Serialized pre-trained Logistic Regression classifier
├── count_vectorizer.pkl           # Serialized pre-trained CountVectorizer vocabulary transformer
├── requirements.txt               # Project dependencies
├── Emotion_dataset.ipynb          # Notebook for model training & evaluation
└── README.md                      # Comprehensive documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python `3.9` or higher
- `pip` or `conda` package manager

### 1. Clone or Open Project Directory
```bash
cd FeelFinder-NLP-Based-Emotion-Classification-System
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Web Application

Launch the Streamlit web server:

```bash
streamlit run app.py
```

Or run via Python module:
```bash
python -m streamlit run app.py
```

Once executed, open your web browser at:
`http://localhost:8501`

---

## 🧪 Example Predictions

| Input Text | Predicted Emotion | Emoji | Expected Confidence |
| :--- | :---: | :---: | :---: |
| *"I got my dream job today!"* | **Joy** | 😊 | ~91% - 96% |
| *"I am really scared about tomorrow."* | **Fear** | 😨 | ~85% - 94% |
| *"I am so angry right now."* | **Anger** | 😠 | ~90% - 97% |
| *"I love spending time with my family."* | **Love** | ❤️ | ~88% - 95% |
| *"I feel so lonely and miserable today."* | **Sadness** | 😢 | ~89% - 96% |
| *"I can't believe what just happened, wow!"* | **Surprise** | 😲 | ~80% - 92% |

---

## 🛡️ Robustness & Security Standards

- **Static Model Files**: The application reads existing serialized binary files (`.pkl`) and does **NOT** attempt to retrain or modify the model weights at runtime.
- **Input Sanitization**: User input is strictly treated as text data passed to `.transform()`. No code execution (`eval`, `exec`) is permitted.
- **Fail-Safe Loading**: Includes automatic verification checks (`check_model_files`) that present user-friendly notification UI in case `.pkl` files are misplaced.

---

## 👤 Author & Acknowledgments

Developed as an end-to-end NLP Machine Learning Portfolio Project demonstrating model deployment, prediction probability interpretation, and custom frontend integration with Streamlit.