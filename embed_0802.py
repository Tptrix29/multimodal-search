from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from chromadb import PersistentClient
from chromadb.config import Settings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import shutil
import pandas as pd
from config import persist_dir

import os
persist_dir = "/Users/christinecym/Desktop/multimodal-search/chroma_storage"
os.makedirs(persist_dir, exist_ok=True)


# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Load image and text
image = Image.open("/Users/christinecym/Desktop/multimodal-search/image/1.jpg")  # replace with your image path
text = "A Die DIY Kit"

# Preprocess and encode
inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Get embeddings
image_embedding = outputs.image_embeds[0].detach().numpy()
text_embedding = outputs.text_embeds[0].detach().numpy()

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(np.float32).tolist()

def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(np.float32).tolist()

# # Initialize the vector data embedding
# chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
# collection = chroma_client.create_collection(name = 'product_images')

# Connect to persistent ChromaDB instance
from chromadb.config import Settings
chroma_client = PersistentClient(path=persist_dir)

# Or just delete the whole collection and recreate
collection = chroma_client.get_or_create_collection("product_images", embedding_function=None)

# Load metadata and ensure consistent ordering
df = pd.read_csv("/Users/christinecym/Desktop/multimodal-search/archive/amazon_products.csv").reset_index(drop=True)

# Build mapping: "1.jpg" -> row 0, "2.jpg" -> row 1, etc.
metadata_lookup = {
    f"{i+1}.jpg": {
        "file_name": f"{i+1}.jpg",
        "product_name": row["title"],
        "product_url": row["productURL"]
    }
    for i, row in df.iterrows()
}



# Load and Embed Product Images
image_path = "/Users/christinecym/Desktop/multimodal-search/image"
image_num = 0
for file in os.listdir(image_path):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(image_path, file)
        image_id = os.path.splitext(file)[0]
        embedding = embed_image(path)

        meta = metadata_lookup.get(file, {"file_name": file, "amazon_url": "#", "product_name": "Unknown Product"})

        collection.add(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[meta]
        )

        image_num += 1
        print(f"Embedding stored for: {file}")
        
print(f"\n Total images processed: {image_num}")

# Retriever
def retrieve_similar_products(query_text, top_k = 5):
    
    print(f"\n Encoding query: '{query_text}")
    
    query_vector = embed_text(query_text)
    print(f"\n Query encoded. Searching database...")
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["embeddings", "metadatas"] 
    )
    
    print("Top results retreived.")
    return results

# Calculate cosine similarity
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(
        torch.tensor(a), torch.tensor(b), dim=0
    ).item()

# Run text query
query = "dye kit"
results = retrieve_similar_products(query)
query_vec = embed_text(query)


# Filter out only image-type results
image_results = [
    (meta, emb) for meta, emb in zip(results["metadatas"][0], results["embeddings"][0])
    if meta and (
        meta.get("type") == "image" or meta.get("file_name", "").lower().endswith(('.jpg', '.jpeg', '.png'))
    )
]


if not image_results:
    print(" No image results found.")
else:
    meta, top_result_embedding = image_results[0]
    file_name = meta.get("file_name", "unknown.jpg")

    # Create IDs
    text_id = f"text_{query.replace(' ', '_')}"
    image_id = f"image_{os.path.splitext(file_name)[0]}"

    # Save to ChromaDB
    assert isinstance(query_vec, list), "Text embedding must be a list"
    collection.add(
        ids=[text_id],
        embeddings=[query_vec],
        metadatas=[{"type": "text", "query": query}]
    )


all_items = collection.get(include=["embeddings", "metadatas"])

print("IDs:", all_items["ids"])
print("Metadata:", all_items["metadatas"])
print("Embeddings:", all_items["embeddings"])




