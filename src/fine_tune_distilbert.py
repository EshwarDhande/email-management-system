from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

# Load the dataset
import pandas as pd
data = pd.read_csv("student_emails.csv")

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize and encode the emails
tokens = tokenizer(
    data['Content'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# Split the dataset into training and testing sets
labels = data['Label'].apply(lambda x: 0 if x == 'student' else 1 if x == 'corporate' else 2).tolist()
X_train, X_test, y_train, y_test = train_test_split(tokens, labels, test_size=0.2, random_state=42)

import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load the model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=(X_train, y_train),
    eval_dataset=(X_test, y_test)
)

# Train the model
trainer.train()
