import streamlit as st
import re
import torchvision

torchvision.disable_beta_transforms_warning()

from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from joblib import load
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "/Project 9 - Mental-Health_Hugging-Face-system/model_bert_mental2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
label_encoder = load("label_encoder.joblib")
nltk.download('punkt_tab')
nltk.download('stopwords')

def cleaned_text(text):
    text = text.lower()
    token = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [word for word in token if word not in stop_words and string.punctuation and word.isalnum()]
    return " ".join(filtered_sentence)

# Streamlit App
st.title("Mentality Sentence Detector App")

st.write("""
This app helps detect if a sentence relates to any of the following mental health conditions:

- Anxiety  
- Depression  
- Stress  
- Bipolar Disorder  
- Suicidal tendencies  
- Personality Disorders  
- Normal mental state  

Simply type your sentence below to get started!
""")


# User Input
sentence_input = st.text_input("Enter a sentence:", "")
# Clear button to reset the input

def predict_sentiment(text):
    text_cleaned = cleaned_text(text)
    inputs = tokenizer(text_cleaned, padding=True, truncation=True, max_length=128, return_tensors="pt")
    # Move input tensors to the same device as the model
    # inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]
# Prediction
if st.button("Detect"):
    text_cleaned = cleaned_text(sentence_input)
    get_prediction = predict_sentiment(text_cleaned)
    st.success(get_prediction)