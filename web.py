import streamlit as st
import os
import numpy as np
from PIL import Image

# Utils
from multimodal import MultimodalEmbedding
from ChromaDB import ChromaDBVector

# Initialize Multimodal Processor
processor = MultimodalEmbedding()

# Initialize ChromaDB Database
db = ChromaDBVector()

db.stats()

def show_img_from_url(url):
    try:
        img = processor.get_online_image(url)
        st.image(img, caption = "Image from URL", use_column_width = True)
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        

# Start to write the website
## Title
st.title("🔎 Multimodal Search (Image + Text)")
st.write("Upload an image and/or enter a text query. It will return matching images with captions.")

## Input
uploaded_image = st.file_uploader("Upload an image", type = ["png", "jpg", "jpeg"])
text_query = st.text_input("Enter text query")

if st.button("Search"):
    if not uploaded_image and not text_query:
        st.warning("Please provide at least an image or a text query.")
    else:
        queries = []
        
        # Process image input
        if uploaded_image:
            query_image = Image.open(uploaded_image).convert("RGB")
            st.image(query_image, caption="Your uploaded image", use_column_width=True)
        else:
            query_image = None

        # Process text input
        if text_query:
            query_text = text_query
        else:
            query_text = None
        
        # Generate embeddings
        if query_image:
            img_query_embedding = processor.embed_image_path(uploaded_image)
            img_query, txt_query = db.query_return(img_query_embedding)
        if query_text:
            txt_query_embedding = processor.embed_text(query_text)
            img_query, txt_query = db.query_return(txt_query_embedding)
        
        # Display results
        st.subheader("Search Results")
        if img_query:
            st.write("Matching Images:")
            for i, metadata in enumerate(img_query["metadatas"][0]):
                st.text(f"Image {i+1}: {metadata["imgUrl"], metadata["title"]}")
                show_img_from_url(metadata["imgUrl"])
            st.write("No matching images found.")
        if txt_query:
            st.write("Matching Texts:")
            for i, metadata in enumerate(txt_query["metadatas"][0]):
                st.text(f"Text {i+1}: {metadata["imgUrl"], metadata["title"]}")
                show_img_from_url(metadata["imgUrl"])
        else:
            st.write("No matching texts found.")