# Week 13 – Building a RAG Question-Answering Pipeline

## Overview
This experiment demonstrates a complete Retrieval-Augmented Generation (RAG) workflow using LangChain, FAISS, Hugging Face embeddings, and Google Gemini. It retrieves relevant information from a climate change document and uses the retrieved context to generate answers.

## Objective
To understand how document retrieval and LLM-based answer generation work together to produce context-grounded responses.

## Technologies Used
- Python
- LangChain
- LangChain Community
- LangChain Google GenAI
- LangChain Hugging Face
- FAISS
- Sentence Transformers
- Google Gemini

## Experiment Description

### 1. Document Preparation
The experiment uses sample text describing the causes and effects of climate change.

### 2. Document Chunking
`RecursiveCharacterTextSplitter` divides the text into smaller documents using:
- Chunk size: `150`
- Chunk overlap: `20`

### 3. Embedding Generation
`HuggingFaceEmbeddings` uses the `sentence-transformers/all-MiniLM-L6-v2` model to convert document chunks into vector embeddings.

### 4. FAISS Vector Store
`FAISS.from_documents()` creates a vector store containing the document chunks and their embeddings.

### 5. Retriever
The vector store is converted into a retriever configured to return the top two relevant chunks for a query.

### 6. Prompt Template
`ChatPromptTemplate` instructs the LLM to:
- Answer using only the provided context.
- Say "I don't know" if the answer is not available in the context.

### 7. Gemini Integration
Google Gemini 3.7 Flash is used to generate answers based on the retrieved context.

### 8. RAG Chain
The experiment builds a LangChain Expression Language (LCEL) pipeline using:
- Retriever
- `RunnablePassthrough`
- Prompt template
- Gemini LLM
- `StrOutputParser`

### 9. Test Questions
The experiment tests:
1. What causes climate change?
2. What are the effects of climate change?

It displays the generated answer and retrieved chunks for each question.

## Model Configuration
- LLM: `gemini-3.7-flash`
- Temperature: `0`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Retrieved chunks: `2`
- Chunk size: `150`
- Chunk overlap: `20`

## API Key Configuration
The experiment retrieves `GOOGLE_API_KEY` from Google Colab Secrets.

Never upload API keys or secret credentials to GitHub.

## How to Run
1. Open the Python file or notebook in Google Colab.
2. Add your Google API key to Colab Secrets using the name `GOOGLE_API_KEY`.
3. Enable notebook access to the secret.
4. Install the required dependencies.
5. Execute the experiment.
6. Review the generated answers and retrieved document chunks.

## Expected Learning Outcome
- Understand a complete RAG workflow.
- Learn how document chunking and embeddings support retrieval.
- Understand vector database indexing using FAISS.
- Learn how retrievers provide relevant context to an LLM.
- Understand how prompt instructions can restrict answers to supplied context.
- Learn how LangChain components connect through LCEL.

## Output
The experiment displays:
1. Answers to both climate change questions.
2. Retrieved document chunks for each question.

Save an output screenshot in the `outputs/` directory.

## Conclusion
This experiment demonstrates how a RAG pipeline combines semantic retrieval with LLM-based generation. By grounding responses in retrieved document content, the system can answer questions using a specific knowledge source rather than relying only on the model's pretrained knowledge.