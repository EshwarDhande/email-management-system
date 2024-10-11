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
    - model (AutoModelForSequenceClassification): The fine-tuned model.
    - tokenizer (AutoTokenizer): The tokenizer for the model.
    - email_text (str): The email text to classify.

    Returns:
    - predicted_label (int): The predicted label for the email.
    """
    # Check for empty email text
    if not email_text.strip():
        print("Error: The provided email text is empty.")
        return None
    
    # Tokenize the email text
    inputs = tokenizer(email_text, return_tensors='pt', truncation=True, padding=True)
    
    # Predict the label
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()
    
    print(f"Predicted label: {predicted_label}")
    return predicted_label

# Response templates
response_templates = {
    0: "Dear Student, thank you for reaching out. I appreciate your interest and will review your inquiry thoroughly. You can expect to hear back from me soon with the details you requested.",
    
    1: "Dear Researcher, thank you for your inquiry. I value your interest and am currently reviewing your request. I will respond shortly with the information you seek.",
    
    2: "Dear Corporate Partner, thank you for your email. I appreciate your inquiry and will forward it to the relevant department for further action. You can expect a response from me shortly."
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
    model_path = os.path.join(os.getcwd(),'..', 'data', 'distilbert_email_model')

    # Load the trained DistilBERT model and tokenizer
    try:
        model, tokenizer = load_finetuned_model(model_path)
    except Exception as e:
        print(f"An error occurred while loading the model: {str(e)}")
        return "Response: Error loading model."

    # Classify the email
    try:
        predicted_label = classify_email(model, tokenizer, email_text)
        if predicted_label is None:
            return "Response: The email text was empty."
        
        # Handle email response based on predicted label
        response = response_templates.get(predicted_label, "Response: Unknown category. Please contact support.")
        print(f"Response generated: {response}\n")
        return response
    except Exception as e:
        print(f"An error occurred during classification: {str(e)}")
        return "Response: An error occurred while processing the email."

# Main guard to execute the script
if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Classify an email text.")
    
    # Add an argument for the email text
    parser.add_argument("email_text", type=str, help="The text of the email to classify.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Process the incoming email
    response = process_incoming_email(args.email_text)
    print(response)  # Print the response to the console