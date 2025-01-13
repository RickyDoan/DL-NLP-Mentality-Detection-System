# Mentality Detection from Text using BERTForSequenceClassification
## Project Overview
Detect mental health-related sentiments from text data using deep learning with the pre-trained `bert-base-uncased` model. The model is trained on Google Colab and deployed as a web app using Streamlit for simplicity and efficiency.

* ![Uploading mentality-detection.gif…](https://github.com/RickyDoan/DL-NLP-Mentality-Detection-System/blob/main/mentality-detection.gif)
---
## Key Features
- **Pre-trained Model**: Fine-tuned `BertForSequenceClassification` from Hugging Face.
- **Custom Tokenizer**: Used `BertTokenizer` for text preprocessing.
- **Efficient Training**: GPU-enabled training on Google Colab.
- **Streamlit Deployment**: Web app for real-time sentiment prediction.
---
## Workflow
1. **Dataset Preparation**: Preprocessed and resampled labeled text data.
2. **Model Training**: Fine-tuned BERT with two sentiment labels using `Trainer` API.
3. **Deployment**: Streamlit app provides real-time predictions.
---
## Installation & Usage
### Training
1. Open `Mentality_text_detection.ipynb` on Google Colab.
2. Upload the dataset.
3. Follow instructions to preprocess, train, and evaluate the model.
### Deployment
1. Clone the repo:
   ```bash
   git clone <repository_url>
