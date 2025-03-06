# Magic Conches


# Scholarly Peer Review App

## Overview
This streamlit application enables the analysis of scientific papers and generates feedback along with a final recommendation for acceptance or rejection.
It uses Large Language Models to simulate a simplified peer reviewing process for research papers.

The application is split into three modes:

- Simple Conversation Mode: a simple LLM based chatbot to handle conversion prior to document upload
- Question Answer Mode: after uploading a research paper in pdf format, the user can ask any question about the document or choose from a set of premade questions
- Final Feedback Mode: the user can ask for a final feedback in the form of a recommendation for acceptance or rejection of the document. The user can ask follow up questions about this feedback

## Setup Guide

### 1. Prerequisites
- Python 3.9+
- `pip` package manager
- API key for OpenAI (to be stored in a `.env` file)

### 2. Installation
```bash
# Clone the repository
git clone <GITLAB_REPOSITORY_URL>
cd scholarly-peer-review-app

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

### 3. Create a .env File
Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 4. Start the Application
```bash
streamlit run app.py
```

### 5. Usage
- Upload a scientific paper in PDF format.
- Ask questions about its content.
- Generate a final recommendation based on the analysis.


# Davyd's Contribution

Group Coordinator – Managed team coordination and project planning.

Main Architecture Design – Defined and implemented the core structure and components of the application.

User Interface Development – Built an intuitive and user-friendly front end.

LLM Integration – Connected multiple LLMs (GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet) to the application.

Retrieval Development – Implemented a retrieval mechanism to fetch relevant document information.

# Jannek's Contribution

Final Feedback - Final usage mode for accept/reject recommendation and follow up questions.

Meeting Transcriptions - created bullet points from various meetings to form clear guidelines.

Application Structure Planning - design of the application structure and modes.

Prompt Engineering - Prompts for final feedback mode to give the role of an assistant reviewer. Also with emphasis on giving a recommendation instead of making a decision.