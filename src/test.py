import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_finetuned_model(model_path):
    """
    Load a fine-tuned DistilBERT model and its tokenizer from the specified model path.
    
    Args:
    - model_path (str): The path to the directory containing the model files.

    Returns:
    - model (DistilBertForSequenceClassification): The loaded fine-tuned model.
    - tokenizer (DistilBertTokenizer): The tokenizer corresponding to the model.
    """
    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # Load the state dict (weights)
    model.load_state_dict(torch.load(os.path.join(model_path, 'finetuned_model.pth')))

    # Set the model to evaluation mode
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

# Function to handle incoming emails (mock function for this example)
def process_incoming_email(email_text):
    """
    Process incoming email and classify it to provide an appropriate response.

    Args:
    - email_text (str): The text of the incoming email.

    Returns:
    - response (str): The response message based on the predicted category.
    """
    # Use a relative path to the model directory
    model_path = os.path.join(os.path.dirname(__file__), '../data/distilbert_email_model/')
    
    # Load the trained DistilBERT model and tokenizer
    model, tokenizer = load_finetuned_model(model_path)
    
    # Classify the email
    predicted_label = classify_email(model, tokenizer, email_text)
    
    # Handle email response based on predicted label
    if predicted_label == 0:
        return "Response: This is a response to a student inquiry."
    elif predicted_label == 1:
        return "Response: This is an escalation for a corporate inquiry."
    elif predicted_label == 2:
        return "Response: This is a response to a research inquiry."
    else:
        return "Response: Unknown category."

if __name__ == "__main__":
    # Example of processing an email
    example_email = "Dear Professor, I would like to inquire about the syllabus for the upcoming semester."
    response = process_incoming_email(example_email)
    print(response)
