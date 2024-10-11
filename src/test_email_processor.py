from email_processor import process_incoming_email

# List of test emails
test_emails = [
    "Dear Professor, I would like to inquire about the syllabus for the upcoming semester.",
    "Dear Sir/Madam, We would like to discuss a potential partnership opportunity with your institution.",
    "Dear Research Team, I am interested in collaborating on your recent research project. Please provide more details.",
    "Hello, can you tell me about the weather?"
]

# Run the test
for i, email in enumerate(test_emails):
    print(f"Test Email {i + 1}: {email}\n")
    response = process_incoming_email(email)
    
