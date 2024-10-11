import os  # Importing os to handle file paths
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader

# Step 1: Load data from specified path
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock_emails', 'parsed_emails.csv')
data = pd.read_csv(data_path)  # Load your CSV file

# Step 2: Define categories based on 'from' column
def categorize_email(email_sender):
    # Check for specific keywords in the sender's email address
    if 'student' in email_sender:
        return 'student'
    elif 'researcher' in email_sender:
        return 'researcher'
    elif any(keyword in email_sender for keyword in ['hr', 'marketing', 'partnerships', 'events', 'recruitment']):
        return 'corporate'
    else:
        return 'corporate'  # Default category for any other corporate emails

# Apply categorization
data['category'] = data['from'].apply(categorize_email)

# Step 3: Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['category'])

# Step 4: Prepare data for training
X = data['body']  # You can also use 'subject' if needed
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Tokenization
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Tokenization
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 6: Create DataLoader
train_dataset = EmailDataset(X_train.tolist(), y_train.tolist())
test_dataset = EmailDataset(X_test.tolist(), y_test.tolist())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Step 7: Model training
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Training Loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

# Step 8: Evaluate the model
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(batch['labels'].numpy())

# Step 9: Generate classification report
print(classification_report(true_labels, predictions, target_names=label_encoder.classes_))


# Step 8: Save the model and tokenizer
model_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'distilbert_email_model')
os.makedirs(model_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the model and tokenizer
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f'Model and tokenizer saved to {model_dir}')
