import json
from pathlib import Path

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_model"

MAX_LENGTH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


st.set_page_config(
    page_title="LLM Fake News Detector",
    page_icon="📰",
    layout="wide"
)


@st.cache_resource
def load_model():
    if not MODEL_DIR.exists():
        st.error("Saved model folder not found.")
        st.write("Please train the model first:")
        st.code("python train_llm.py", language="bash")
        st.stop()

    required_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json"
    ]

    existing_files = [file.name for file in MODEL_DIR.iterdir()]

    if "pytorch_model.bin" in existing_files:
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json"
        ]

    missing_files = [
        file for file in required_files
        if file not in existing_files
    ]

    if missing_files:
        st.error("Trained model files are missing.")
        st.write("Please run:")
        st.code("python train_llm.py", language="bash")
        st.write("Missing files:")
        for file in missing_files:
            st.write("-", file)
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model = model.to(DEVICE)
    model.eval()

    label_path = MODEL_DIR / "label_mapping.json"

    if label_path.exists():
        with open(label_path, "r") as file:
            label_mapping = json.load(file)
    else:
        label_mapping = {
            "0": "Fake",
            "1": "Real"
        }

    return tokenizer, model, label_mapping


def predict_news(title, text, tokenizer, model, label_mapping):
    combined_text = str(title) + " " + str(text)

    encoded = tokenizer(
        combined_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    label = label_mapping.get(str(predicted_class), str(predicted_class))

    return label, confidence, probabilities.cpu().numpy()


def generate_llm_style_explanation(label, confidence):
    if label.lower() == "fake":
        return (
            f"The model predicts that this news article is likely Fake with confidence {confidence:.2f}. "
            "This means the language pattern, title, and body text are closer to fake-news examples learned during training. "
            "The result should be verified with reliable sources before making any conclusion."
        )

    return (
        f"The model predicts that this news article is likely Real with confidence {confidence:.2f}. "
        "This means the language pattern, title, and body text are closer to real-news examples learned during training. "
        "However, users should still verify important news from trusted sources."
    )


tokenizer, model, label_mapping = load_model()


st.title("📰 LLM-Based Fake News Detection System")
st.write(
    "This application uses a fine-tuned DistilBERT transformer model to classify news articles as Fake or Real."
)

st.warning(
    "This system is for research and educational use only. It should not be used as the only source for verifying news."
)

tab1, tab2 = st.tabs(["Prediction", "About Model"])


with tab1:
    st.header("Enter News Article")

    title = st.text_input("News Title")

    text = st.text_area(
        "News Body Text",
        height=250,
        placeholder="Paste the full news article text here..."
    )

    if st.button("Detect Fake News"):
        if title.strip() == "" and text.strip() == "":
            st.error("Please enter a title or news text.")
        else:
            label, confidence, probabilities = predict_news(
                title=title,
                text=text,
                tokenizer=tokenizer,
                model=model,
                label_mapping=label_mapping
            )

            if label.lower() == "fake":
                st.error(f"Prediction: {label}")
            else:
                st.success(f"Prediction: {label}")

            st.metric("Confidence", f"{confidence:.2f}")

            st.write("Class Probabilities:")
            st.write({
                "Fake": float(probabilities[0]),
                "Real": float(probabilities[1])
            })

            explanation = generate_llm_style_explanation(
                label=label,
                confidence=confidence
            )

            st.info(explanation)


with tab2:
    st.header("Model Information")

    st.write(
        """
        This project fine-tunes a transformer-based language model for binary fake news classification.
        
        The pipeline includes:
        
        1. Dataset loading  
        2. Title and text merging  
        3. Label encoding  
        4. Train-validation-test split  
        5. DistilBERT fine-tuning  
        6. Evaluation using accuracy, precision, recall, and F1-score  
        7. Streamlit deployment  
        """
    )

    st.write("Device used:", str(DEVICE))

st.markdown("---")
st.caption("LLM Fake News Detector | Research Prototype")