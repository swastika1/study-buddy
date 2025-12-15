# 📚 Study Buddy

AI-powered study assistant using AWS Bedrock. Ask questions and generate quizzes from your documents.

## Features

- Ask questions about your documents with source citations
- Generate practice quizzes from all files or specific documents
- Supports PDF, DOCX, PPTX, XLSX, TXT, MD, HTML, CSV, JSON
- Powered by Amazon Nova Lite and Titan Embeddings

## Quick Setup

1. **Install**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Configure AWS**
Create `.env` file:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1
```

3. **Enable Bedrock Models**
Enable these at [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/home#/modelaccess):
- Amazon Titan Embeddings G1 - Text
- Amazon Nova Lite

4. **Add Documents & Run**
```bash
# Add files to files/ folder
cp your_notes.pdf files/

# Run
python app.py
```

## Commands

- `[question]` - Ask about your documents
- `quiz` - 5 questions from all files
- `quiz 10` - 10 questions
- `quiz file` - Select file for quiz
- `quiz file name.pdf` - Quiz from specific file
- `files` - List documents
- `exit` - Quit

## Optional Formats

Install for DOCX, PPTX, XLSX, etc:
```bash
pip install docx2txt python-pptx openpyxl unstructured pandas pillow
```

## Troubleshooting

- **No documents found**: Add files to `files/` folder
- **AWS errors**: Check `.env` credentials and enable Bedrock models
- **Import errors**: Activate venv and run `pip install -r requirements.txt`


---

**Happy Studying! 📚**
