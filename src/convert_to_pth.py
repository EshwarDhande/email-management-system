import torch
from transformers import AutoModel, AutoTokenizer

# Define the path to your model files
model_path = "C:/Users/ishwa/Python_code/email-management-system/data/distilbert_email_model"

# Load the model and tokenizer from the original files
model = AutoModel.from_pretrained(model_path)  # Use from_flax=True if applicable
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Save the model's state dict as a .pth file
torch.save(model.state_dict(), "C:/Users/ishwa/Python_code/email-management-system/data/distilbert_email_model/model.pth")

print("Model converted and saved as model.pth")
