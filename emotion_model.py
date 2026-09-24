"""
FeelFinder - Emotion Prediction Engine Module
Handles ML model loading, text vectorization, prediction, and confidence probability mapping.
"""

import os
import warnings
import joblib
import numpy as np

# Suppress version mismatch warnings when unpickling scikit-learn estimators
warnings.filterwarnings("ignore", category=UserWarning)

# Label Mapping based on model output:
# 1 = Sadness, 2 = Anger, 3 = Love, 4 = Surprise, 5 = Fear, 6 = Joy
EMOTION_CONFIG = {
    1: {
        "name": "Sadness",
        "emoji": "😢",
        "color": "#3B82F6",
        "gradient": "linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%)",
        "light_bg": "rgba(59, 130, 246, 0.12)",
        "border_color": "rgba(59, 130, 246, 0.3)",
        "description": "Expresses feelings of sorrow, grief, or unhappiness."
    },
    2: {
        "name": "Anger",
        "emoji": "😠",
        "color": "#EF4444",
        "gradient": "linear-gradient(135deg, #991B1B 0%, #EF4444 100%)",
        "light_bg": "rgba(239, 68, 68, 0.12)",
        "border_color": "rgba(239, 68, 68, 0.3)",
        "description": "Expresses annoyance, frustration, or hostility."
    },
    3: {
        "name": "Love",
        "emoji": "❤️",
        "color": "#EC4899",
        "gradient": "linear-gradient(135deg, #9D174D 0%, #EC4899 100%)",
        "light_bg": "rgba(236, 72, 153, 0.12)",
        "border_color": "rgba(236, 72, 153, 0.3)",
        "description": "Expresses affection, deep fondness, or warmth."
    },
    4: {
        "name": "Surprise",
        "emoji": "😲",
        "color": "#F59E0B",
        "gradient": "linear-gradient(135deg, #B45309 0%, #F59E0B 100%)",
        "light_bg": "rgba(245, 158, 11, 0.12)",
        "border_color": "rgba(245, 158, 11, 0.3)",
        "description": "Expresses astonishment, wonder, or unexpectedness."
    },
    5: {
        "name": "Fear",
        "emoji": "😨",
        "color": "#8B5CF6",
        "gradient": "linear-gradient(135deg, #5B21B6 0%, #8B5CF6 100%)",
        "light_bg": "rgba(139, 92, 246, 0.12)",
        "border_color": "rgba(139, 92, 246, 0.3)",
        "description": "Expresses apprehension, anxiety, or fright."
    },
    6: {
        "name": "Joy",
        "emoji": "😊",
        "color": "#10B981",
        "gradient": "linear-gradient(135deg, #065F46 0%, #10B981 100%)",
        "light_bg": "rgba(16, 185, 129, 0.12)",
        "border_color": "rgba(16, 185, 129, 0.3)",
        "description": "Expresses happiness, delight, or positive satisfaction."
    }
}

DEFAULT_MODEL_PATH = "LogisticRegression_model.pkl"
DEFAULT_VEC_PATH = "count_vectorizer.pkl"


def check_model_files(model_path=DEFAULT_MODEL_PATH, vec_path=DEFAULT_VEC_PATH):
    """
    Validates whether required model artifact files exist.
    """
    missing_files = []
    if not os.path.exists(model_path):
        missing_files.append(model_path)
    if not os.path.exists(vec_path):
        missing_files.append(vec_path)
    
    return len(missing_files) == 0, missing_files


def load_model_artifacts(model_path=DEFAULT_MODEL_PATH, vec_path=DEFAULT_VEC_PATH):
    """
    Loads trained Logistic Regression model and CountVectorizer artifacts.
    """
    exists, missing = check_model_files(model_path, vec_path)
    if not exists:
        raise FileNotFoundError(f"Missing required model artifact files: {', '.join(missing)}")
    
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    except Exception as e:
        raise RuntimeError(f"Error loading model artifacts: {str(e)}")


def predict_emotion(text, model, vectorizer):
    """
    Core Emotion Prediction Logic:
    1. Validates text input.
    2. Vectorizes input text using loaded CountVectorizer instance.
    3. Predicts numeric label using trained Logistic Regression model.
    4. Computes probability scores across all emotion classes.
    5. Returns formatted dictionary containing prediction summary & class distribution.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "success": False,
            "error": "Input text is empty or invalid. Please enter meaningful text."
        }
    
    clean_text = text.strip()
    
    try:
        # Step 1: Transform text using existing CountVectorizer (no re-fitting)
        vectorized_text = vectorizer.transform([clean_text])
        
        # Step 2: Predict numeric class label using model.predict()
        prediction = model.predict(vectorized_text)[0]
        numeric_label = int(prediction)
        
        # Step 3: Compute class probabilities using model.predict_proba()
        probabilities = model.predict_proba(vectorized_text)[0]
        
        # Retrieve classes list from model if available, else standard 1..6
        model_classes = getattr(model, 'classes_', [1, 2, 3, 4, 5, 6])
        
        # Map class probabilities
        class_proba_map = {}
        for idx, cls_id in enumerate(model_classes):
            cls_int = int(cls_id)
            cls_info = EMOTION_CONFIG.get(cls_int, {"name": f"Emotion {cls_int}", "emoji": "❓"})
            class_proba_map[cls_info["name"]] = {
                "label_id": cls_int,
                "emoji": cls_info["emoji"],
                "probability": float(probabilities[idx]),
                "percentage": round(float(probabilities[idx]) * 100, 2),
                "color": cls_info.get("color", "#6B7280")
            }
        
        # Selected emotion metadata
        selected_info = EMOTION_CONFIG.get(numeric_label, {
            "name": "Unknown",
            "emoji": "❓",
            "color": "#6B7280",
            "gradient": "linear-gradient(135deg, #374151 0%, #6B7280 100%)",
            "light_bg": "rgba(107, 114, 128, 0.12)",
            "border_color": "rgba(107, 114, 128, 0.3)",
            "description": "Unrecognized emotion label."
        })
        
        # Confidence score for predicted class
        pred_idx = list(model_classes).index(numeric_label) if numeric_label in model_classes else 0
        confidence_pct = round(float(probabilities[pred_idx]) * 100, 2)
        
        # Explanation sentence
        explanation = f"The model detected **{selected_info['name']}** in your text with **{confidence_pct}%** confidence."
        
        return {
            "success": True,
            "numeric_label": numeric_label,
            "emotion": selected_info["name"],
            "emoji": selected_info["emoji"],
            "confidence": confidence_pct,
            "explanation": explanation,
            "description": selected_info["description"],
            "color": selected_info["color"],
            "gradient": selected_info["gradient"],
            "light_bg": selected_info["light_bg"],
            "border_color": selected_info["border_color"],
            "probabilities": class_proba_map,
            "input_text": clean_text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"An error occurred during emotion prediction: {str(e)}"
        }
