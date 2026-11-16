# Website Knowledge Assistant Using RAG

## 1. Project Overview

The Website Knowledge Assistant is an AI-powered application that allows users to ask questions about any website and receive answers based on its content.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a website and generate answers using a locally running AI model.

## 2. Problem Statement

Reading and searching through large websites to find specific information can take a lot of time.

This project solves that problem by allowing users to enter a website URL and ask questions directly. The application retrieves relevant information and generates simple answers.

## 3. Technologies Used

- Python – Main programming language
- Streamlit – Web application interface
- BeautifulSoup – Website text extraction
- Sentence Transformers – Text embeddings
- ChromaDB – Vector database
- Ollama – Local AI model execution
- Qwen2.5-Coder:3B – Language model

## 4. Understanding RAG

RAG stands for Retrieval-Augmented Generation.

It combines two main processes:

**Retrieval:** Finds relevant information from stored website content.

**Generation:** Uses the retrieved information to generate an answer.

## 5. How the Project Works

1. User enters a website URL.
2. BeautifulSoup extracts the website text.
3. The extracted text is divided into smaller chunks.
4. Sentence Transformers converts chunks into embeddings.
5. ChromaDB stores the embeddings and text.
6. User asks a question.
7. The question is converted into an embedding.
8. ChromaDB retrieves the top 3 relevant chunks.
9. Ollama sends the retrieved information to the Qwen2.5-Coder model.
10. The generated answer is displayed in Streamlit.

## 6. Project Features

- Website URL input
- Automatic website text extraction
- Text chunking
- Text embedding generation
- Vector database storage
- Semantic search and retrieval
- AI-generated answers
- Chat interface
- Clear Chat option
- Database information display

## 7. Project Architecture

User enters Website URL  
↓  
Streamlit Application  
↓  
BeautifulSoup Text Extraction  
↓  
Text Chunking  
↓  
Sentence Transformer Embeddings  
↓  
ChromaDB Vector Database  
↓  
User Question  
↓  
Retrieve Relevant Chunks  
↓  
Ollama – Qwen2.5-Coder  
↓  
Generate Answer  
↓  
Display Answer in Streamlit

## 8. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/dornadulasunandini-png/Website-Knowledge-Assistant.git
```

### Step 2: Open the Project Folder

```bash
cd Website-Knowledge-Assistant
```

### Step 3: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 4: Activate the Environment

```bash
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Install and Run Ollama

Download Ollama