import pandas as pd
import chromadb
from chromadb.config import Settings
from openai import AzureOpenAI


df = pd.read_csv("assignment2dataset.csv")


embedding_client = AzureOpenAI(
    api_key="DfzssXXXXXXXXXXXXX",
    api_version="2024-02-15-preview",
    azure_endpoint="https://user2-mb0ggj8z-northcentralus.cognitiveservices.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2023-05-15"
)
EMBEDDING_MODEL = "text-embedding-3-small"

summarization_client = AzureOpenAI(
    api_key="DfzssXXXXXXXXXXXXX",
    api_version="2024-02-15-preview",
    azure_endpoint="https://user2-mb0ggj8z-northcentralus.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview"
)
SUMMARIZER_MODEL = "gpt-4o-mini"


chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection_name = "course_collection"

if collection_name in [c.name for c in chroma_client.list_collections()]:
    chroma_client.delete_collection(collection_name)

collection = chroma_client.create_collection(name=collection_name)


def embed_and_add_courses(df: pd.DataFrame):
    texts = (df['title'] + ". " + df['description']).tolist()
    embeddings = embedding_client.embeddings.create(input=texts, model=EMBEDDING_MODEL).data
    vector_data = [e.embedding for e in embeddings]

    collection.add(
        documents=texts,
        embeddings=vector_data,
        ids=df['course_id'].astype(str).tolist(),
        metadatas=df[['title', 'description']].to_dict(orient='records')
    )


def recommend_courses(query: str, top_k: int = 5):
    response = embedding_client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    query_vector = response.data[0].embedding

    results = collection.query(
    query_embeddings=[query_vector],
    n_results=top_k,
    include=['metadatas', 'distances', 'documents']  
)

    matches = []
    for i in range(top_k):
        matches.append({
            "course_id": results['ids'][0][i],
            "title": results['metadatas'][0][i]['title'],
            "description": results['metadatas'][0][i]['description'],
            "score": results['distances'][0][i]
        })
    return matches


def summarize_results(query: str, courses: list) -> str:
    context = "\n\n".join([f"Title: {c['title']}\nDescription: {c['description']}" for c in courses])
    prompt = f"""You are a helpful assistant. A user asked: "{query}"
Below are some courses retrieved:

{context}

Summarize these results and explain which courses are most relevant to the query."""

    response = summarization_client.chat.completions.create(
        model=SUMMARIZER_MODEL,
        messages=[
            {"role": "system", "content": "You are an educational advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    embed_and_add_courses(df)

    query = "Looking to study programming and AI basics"
    top_courses = recommend_courses(query)
    print("Top Matches:\n", top_courses)

    summary = summarize_results(query, top_courses)
    print("\nSummary:\n", summary)
