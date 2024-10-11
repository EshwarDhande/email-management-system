import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_finetuned_model(model_path):
    """
    Load a fine-tuned DistilBERT model and its tokenizer from the specified model path.
    
    Args:
    - model_path (str): The path to the directory containing the model files.

    Returns:
    - model (AutoModelForSequenceClassification): The loaded fine-tuned model.
    - tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
    """
    # Load the tokenizer using AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model using AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()

    return model, tokenizer

def classify_email(model, tokenizer, email_text):
    """
    Classify the email text into predefined categories using the fine-tuned model.

    Args:
    - model (DistilBertForSequenceClassification): The fine-tuned model.
    - tokenizer (DistilBertTokenizer): The tokenizer for the model.
    - email_text (str): The email text to classify.

    Returns:
    - predicted_label (int): The predicted label for the email.
    """
    # Tokenize the email text
    inputs = tokenizer(email_text, return_tensors='pt', truncation=True, padding=True)
    
    # Predict the label
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    
    return predicted_label

# Response templates
response_templates = {
    0: "Dear Student, thank you for your inquiry. Here is the syllabus you requested.",
    1: "Dear Corporate Partner, thank you for your email. We will escalate your inquiry to the appropriate department.",
    2: "Dear Researcher, thank you for your interest. We will get back to you shortly with the requested information."
}

def process_incoming_email(email_text):
    """
    Process incoming email and classify it to provide an appropriate response.

    Args:
    - email_text (str): The text of the incoming email.

    Returns:
    - response (str): The response message based on the predicted category.
    """
    # Use a relative path to the model directory
    model_path = os.path.join(os.getcwd(), '..', 'data', 'distilbert_email_model')


    # Load the trained DistilBERT model and tokenizer
    model, tokenizer = load_finetuned_model(model_path)
    
    # Classify the email
    predicted_label = classify_email(model, tokenizer, email_text)
    
    # Handle email response based on predicted label
    response = response_templates.get(predicted_label, "Response: Unknown category. Please contact support.")
    return response

