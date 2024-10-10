import os

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
        # Split emails by separator (---)
        raw_emails = content.split('\n---\n')
        for raw_email in raw_emails:
            # Further split to extract subject, from, to, and body
            lines = raw_email.strip().split('\n')
            if len(lines) < 4:
                continue  # Skip incomplete emails
            email = {
                'subject': lines[0].replace('Subject: ', '').strip(),
                'from': lines[1].replace('From: ', '').strip(),
                'to': lines[2].replace('To: ', '').strip(),
                'body': '\n'.join(lines[3:]).strip()  # Joining body lines
            }
            print(f"Parsed email: {email}")
            emails.append(email)
    return emails

if __name__ == "__main__":
    # Parse all email files
    email_types = ['student_emails.txt', 'corporate_emails.txt', 'researcher_emails.txt']
    all_emails = []
    
    for email_file in email_types:
        file_path = os.path.join('data', 'mock_emails', email_file)  # Updated line here
        all_emails.extend(parse_emails(file_path))
    
    # Print parsed emails for verification
    for email in all_emails:
        print(email)
