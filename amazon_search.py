import packages 
import streamlit as st
from PIL import Image
import os
import torch
from transformers import CLIPModel, CLIPProcessor
import chromadb
from chromadb.config import Settings
import numpy as np


# Load model
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

model, processor = load_model()

# ChromaDB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_storage", anonymized_telemetry=False))
collection = chroma_client.get_or_create_collection("product_images", embedding_function=None)

# Embedding helpers
def embed_image(image):
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

# Interface
st.markdown(
    """
    <div style="display:inline-flex; align-items:center; gap: 12px; font-size: 2.2rem; font-weight: 700;">
        <span>🔍</span>
        <span>Multimodal Search: Text & Image</span>
    </div>
    """,
    unsafe_allow_html=True
)



search_type = st.radio("Choose search type:", ["Text", "Image"])

if search_type == "Text":
    query = st.text_input("Enter your text query:")
    if query:
        query_vector = embed_text(query)
        results = collection.query(query_embeddings=[query_vector], n_results=3, include=["metadatas"])
        st.subheader("Results:")
        for r in results["metadatas"][0]:
            st.image(f"./image/{r['file_name']}", caption=r['file_name'], width=250)

elif search_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", width=250)
        image_vector = embed_image(img)
        results = collection.query(query_embeddings=[image_vector], n_results=3, include=["metadatas"])
        st.subheader("Results:")
        for r in results["metadatas"][0]:
            st.image(f"./image/{r['file_name']}", caption=r['file_name'], width=250)

