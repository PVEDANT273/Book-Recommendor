A semantic book recommendation system powered by LLM-based text embeddings, emotion analysis, and vector similarity search. Users can describe a book and receive recommendations filtered by category and tone (happy / sad / suspenseful etc.).
The UI is built using Gradio, and the backend uses Ollama embeddings, ChromaDB, and dataset preprocessing in Python.

Features:
Uses Semantic similarity to suggest books using subject and tone 
the tones are classified in 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral
Category Filtering(Fiction, Nonfiction, Children's Fiction etc)
Clean Gradio Interface

Models:
Embedding Models - nomic-embed-text:latest
Text Classification - facebook/bart-large-mnli, j-hartmann/emotion-english-distilroberta-base