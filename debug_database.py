import chromadb
from chromadb.config import Settings

# Connect to your persistent database
client = chromadb.PersistentClient(path="./data/vector_db", settings=Settings(anonymized_telemetry=False))
collection = client.get_collection("enterprise_docs")

# Get the first 20 items
results = collection.get(include=["documents", "metadatas"], limit=20)
print(f"Total chunks in DB: {collection.count()}\n")
print("="*50)

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
    print(f"Chunk ID: {i}")
    print(f"From File: {meta.get('filename', 'N/A')}")
    print(f"Text Preview: {doc[:150]}...")  # First 150 chars
    print("-"*50)