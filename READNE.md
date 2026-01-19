# Semantic Book Recommendation System

A semantic book recommendation system powered by LLM-based text embeddings, emotion analysis, and vector similarity search.  
Users can describe a book and receive recommendations filtered by category and emotional tone.

The user interface is built using Gradio, and the backend uses Ollama embeddings, ChromaDB, and Python-based preprocessing.

## Features

- Uses semantic similarity to suggest books based on description
- Emotion-based filtering (anger, disgust, fear, joy, sadness, surprise, neutral)
- Category filtering (Fiction, Nonfiction, Children's Fiction, etc.)
- Clean and simple Gradio interface


## Models Used

### Embedding Model
- `nomic-embed-text:latest` (Ollama)

### Text Classification
- `facebook/bart-large-mnli`
- `j-hartmann/emotion-english-distilroberta-base`


## Tech Stack

- Python
- Pandas, NumPy
- LangChain
- Ollama
- ChromaDB
- Hugging Face Transformers
- Gradio
