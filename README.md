
# LLM-Based Fake News Detection System

This project is an LLM/Transformer-based fake news detection system built using **DistilBERT** and **Streamlit**. The system classifies news articles as **Fake** or **Real** based on the article title and body text.

The project includes a full training pipeline, model saving system, and a web-based prediction interface where users can enter a news headline and article text to get a prediction with confidence score.

---

## Project Overview

The main goal of this project is to build a transformer-based fake news detection model using Natural Language Processing. Instead of using only traditional machine learning models such as SVM, Random Forest, or Logistic Regression, this project fine-tunes a pre-trained transformer model called **DistilBERT** for binary text classification.

The system takes two inputs:

- News title
- News body text

Then it predicts whether the article is:

- Fake
- Real

---

## Features

- Fine-tuned DistilBERT model
- Fake/Real news classification
- Title and body text merging
- Robust CSV loading system
- Handles corrupted CSV rows
- Train-validation-test split
- Model evaluation using accuracy, precision, recall, and F1-score
- Saves trained model automatically
- Streamlit web application
- Confidence score display
- LLM-style prediction explanation
- Simple and clean user interface

---

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- TQDM

---

## Project Structure

```text
fake_news_llm_project/
│
├── data/
│   └── fake_news_merged.csv
│
├── saved_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── label_mapping.json
│   └── training_info.pkl
│
├── train_llm.py
├── app.py
├── requirements.txt
└── README.md
````

---

## Dataset

The dataset should contain at least the following columns:

```text
title
text
label
```

The `title` column contains the news headline.

The `text` column contains the full news article.

The `label` column contains the class label.

In this project, the label format is:

```text
0 = Fake
1 = Real
```

Before running the project, place the dataset inside the `data` folder and rename it exactly as:

```text
fake_news_merged.csv
```

Final dataset path:

```text
data/fake_news_merged.csv
```

---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/fake_news_llm_project.git
cd fake_news_llm_project
```

Or, if the project is already downloaded:

```bash
cd /c/Users/Kamrul/Downloads/fake_news_llm_project
```

---

### Step 2: Create a Virtual Environment

For Windows Git Bash:

```bash
py -3.11 -m venv venv
source venv/Scripts/activate
```

For Windows CMD:

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, the terminal should show:

```text
(venv)
```

---

### Step 3: Install Required Libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## requirements.txt

The `requirements.txt` file should contain:

```text
pandas
numpy
scikit-learn
torch
transformers
streamlit
joblib
tqdm
```

---

## How to Train the Model

Before running the web app, train the DistilBERT model first.

```bash
python train_llm.py
```

During training, the system will:

1. Load the dataset
2. Clean column names
3. Merge title and text
4. Encode labels
5. Split data into train, validation, and test sets
6. Load DistilBERT tokenizer
7. Fine-tune DistilBERT
8. Evaluate the model
9. Save the best model inside the `saved_model` folder

After successful training, the `saved_model` folder should contain files such as:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.txt
label_mapping.json
training_info.pkl
```

---

## How to Run the Web App

After training is complete, run:

```bash
streamlit run app.py
```

Then open the local URL in your browser:

```text
http://localhost:8501
```

---

## Correct Running Order

Always follow this order:

```bash
python train_llm.py
streamlit run app.py
```

Do not run the Streamlit app before training the model, because the app needs the saved model files from the `saved_model` folder.

---

## Example Input

### News Title

```text
Health Officials Launch Awareness Program to Promote Early Screening
```

### News Body Text

```text
DHAKA - Health officials launched a public awareness campaign on Monday to encourage early screening, regular medical checkups, and healthier lifestyle choices among adults. According to the health department, the program will be conducted through community health centers, schools, and local awareness sessions. Officials said the campaign aims to improve early detection of chronic diseases and reduce long-term health risks.
```

The system will analyze the language pattern and classify the news as Fake or Real.

---

## Model Workflow

The workflow of the system is:

```text
Dataset Loading
        ↓
Data Cleaning
        ↓
Title + Text Merging
        ↓
Label Encoding
        ↓
Train / Validation / Test Split
        ↓
Tokenization using DistilBERT Tokenizer
        ↓
DistilBERT Fine-Tuning
        ↓
Model Evaluation
        ↓
Model Saving
        ↓
Streamlit Web App Deployment
        ↓
Fake/Real News Prediction
```

---

## Model Used

This project uses:

```text
distilbert-base-uncased
```

DistilBERT is a lighter and faster version of BERT. It is suitable for text classification tasks such as fake news detection, sentiment analysis, and misinformation detection.

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

---

## Important Notes

If the model always predicts **Fake**, retrain the model using balanced Fake and Real samples.

To retrain from scratch, delete the old saved model:

```bash
rm -rf saved_model
mkdir saved_model
python train_llm.py
```

Then run the app again:

```bash
streamlit run app.py
```

---

## CPU Training Settings

If you are training on CPU, use smaller settings inside `train_llm.py`:

```python
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 1
MAX_ROWS = 5000
```

For better results, use GPU or Google Colab and increase:

```python
EPOCHS = 2
MAX_ROWS = 10000
```

---

## Limitations

* The model performance depends on the quality of the dataset.
* If the dataset has biased or imbalanced labels, predictions may be biased.
* The model may misclassify short or unclear articles.
* It should not be used as the only source for verifying news.
* Real-world fake news detection requires fact-checking from trusted sources.

---

## Future Improvements

Possible future improvements include:

* Add SHAP or LIME explainability
* Add confusion matrix visualization
* Add ROC curve and performance dashboard
* Train on a larger and cleaner dataset
* Use larger transformer models such as BERT, RoBERTa, or DeBERTa
* Add multilingual fake news detection
* Deploy the app online using Streamlit Cloud or Hugging Face Spaces
* Add source credibility analysis
* Add URL-based news verification

---

## Research Title

**LLM-Based Fake News Detection Using Fine-Tuned DistilBERT and Streamlit Deployment**

---

## Disclaimer

This project is developed for research and educational purposes only. The system predicts whether a news article appears Fake or Real based on learned language patterns from the dataset. It does not perform real-time fact-checking and should not be used as the only tool for verifying news.

---

## Author

**Md. Kamrul Hasan**
Department of Computer Science
American International University-Bangladesh

```
```
