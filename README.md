# Multi-Source Document QA System with RAG and Fine-Tuned Embeddings

## Overview

This project implements an end-to-end multi-source document question-answering (QA) system that:
- Processes documents in multiple formats (PDF, DOCX, TXT)
- Splits the text into chunks using a recursive character text splitter
- Indexes document chunks using FAISS with embeddings from a SentenceTransformer
- Uses a Retrieval-Augmented Generation (RAG) pipeline (integrated via LangChain and OpenAI's GPT-3.5-turbo) to answer user queries
- Evaluates performance using metrics like Recall@K and BLEU

## Project Structure

