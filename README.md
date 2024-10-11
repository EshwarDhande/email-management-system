# Email Management System

## Overview
This project is an Email Management System designed to classify and respond to incoming emails automatically. It utilizes a machine learning model to categorize emails into three types: students, corporates, and researchers.

## Features
- Email classification based on content
- Automated responses for common inquiries
- Dockerized for easy deployment


## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>

2. **Navigate to the project directory**
   ```bash
   cd email-management-system

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the application:**
   ```bash
   python src/email_processor.py

## Project Experience

In this project, I created an email dataset from scratch, comprising 50 emails for each of the following categories: students, researchers, and corporate professionals. I chose to train a DistilBERT model (66 million parameters), as I believed the task at hand did not require a larger model with billions of parameters. My aim was to optimize memory usage, trading off some performance for efficiency.

Additionally, I intended to develop a dataset that would allow the model to generate responses to queries from students and researchers. However, due to time constraints, I opted for pre-made templates, ultimately settling on just three templates instead of dynamic generation.

Initially, I planned to deploy a sophisticated email management system, but time limitations forced me to focus on a simpler solution that involved training the model for classification and generating emails based on that classification.

Upon reflection, I realized that I spent considerable time strategizing and perfecting my plan instead of taking action earlier. In hindsight, I should have started executing the project sooner and planned concurrently.

Nevertheless, it was an enjoyable experience overall. I am grateful to the team for giving me the opportunity, and I was delighted to be shortlisted from the exam. Thank you!

