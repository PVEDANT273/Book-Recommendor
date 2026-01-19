import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document   

import gradio as gr

books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.png",
    books["large_thumbnail"]
)

# raw_documents = TextLoader("tagged_descriptions.txt", encoding="utf-8").load()
# text_spliiter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = []

with open("tagged_descriptions.txt", "r", encoding="utf-8") as f:
    for line in f:
        text = line.strip()
        if text:
            documents.append(Document(page_content=text))

retriever = Chroma.from_documents(documents=documents, 
                                  embedding=OllamaEmbeddings(
                                      model="nomic-embed-text:latest"
                                ))

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    
    recs = retriever.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category][:final_top_k]
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        books_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        books_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        books_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        books_recs.sort_values(by="sadness", ascending=False, inplace=True)
    
    return books_recs


def retrieve_books(
        query: str,
        category: str,
        tone: str
):
    
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()  #if the description is too long
        truncated_description = " ".join(truncated_desc_split[:30] + ["..."])

        authors_split = row["authors"].split(";")

        if(len(authors_split)) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"

        elif(len(authors_split)) > 2:
            authors_str = f"...,{authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic Book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter the description of a book",
                                placeholder="e.g.. A book about revenge")
        
        category_dropdown = gr.Dropdown(choices= categories, label= "Select a cateogry:", value="All")
        tone_dropwdown = gr.Dropdown(choices= tones, label= "Select a tone:", value="All")
        submit_button = gr.Button("Find recommendations")


    gr.Markdown("## Recommendations")
    output = gr.Gallery(label= "Recommended books", columns= 8, rows=2)

    submit_button.click(fn= retrieve_books,
                        inputs= [user_query, category_dropdown, tone_dropwdown],
                        outputs= output)

if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Ocean())