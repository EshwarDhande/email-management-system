import os
import pandas as pd

def parse_emails(file_path):
    """
    Parses emails from a given text file.

    Args:
    - file_path (str): Path to the email file.

    Returns:
    - list of dict: A list of parsed email dictionaries.
    """
    emails = []
    with open(file_path, 'r') as file:
        content = file.read()
        raw_emails = content.split('\n---\n')  # Split emails by separator (---)
        for raw_email in raw_emails:
            lines = raw_email.strip().split('\n')
            if len(lines) < 4:
                continue  # Skip incomplete emails
            
            email = {
                'subject': lines[0].replace('Subject: ', '').strip(),
                'from': lines[1].replace('From: ', '').strip(),
                'to': lines[2].replace('To: ', '').strip(),
                'body': '\n'.join(lines[3:]).strip()  # Joining body lines
            }
            emails.append(email)
    
    return emails

def save_emails_to_csv(emails, output_file):
    """
    Saves a list of email dictionaries to a CSV file.

    Args:
    - emails (list of dict): List of parsed email dictionaries.
    - output_file (str): The output CSV file path.
    """
    df = pd.DataFrame(emails)  # Create a DataFrame from the email list
    df.to_csv(output_file, index=False)  # Export DataFrame to CSV

if __name__ == "__main__":
    email_types = ['generated_students_emails.txt', 'generated_corporate_emails.txt', 'generated_researchers_emails.txt']
    all_emails = []
    
    for email_file in email_types:
        file_path = os.path.join('..', 'data', email_file)  # Go one level up to access the data folder
        all_emails.extend(parse_emails(file_path))

    # Save the parsed emails to a CSV file
    output_csv_file = 'emails_dataset.csv'
    save_emails_to_csv(all_emails, output_csv_file)
    print(f'Successfully saved emails to {output_csv_file}')
