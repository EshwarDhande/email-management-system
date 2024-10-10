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
    email_types = ['student_emails.txt', 'corporate_emails.txt', 'researcher_emails.txt']
    all_emails = []
    
    for email_file in email_types:
        file_path = os.path.join('data', 'mock_emails', email_file)
        all_emails.extend(parse_emails(file_path))

    # Save the parsed emails to a CSV file
    output_csv_file = 'parsed_emails.csv'
    save_emails_to_csv(all_emails, output_csv_file)
    print(f'Successfully saved parsed emails to {output_csv_file}')
