import streamlit as st
import os
import numpy as np
from PIL import Image

# Utils
from multimodal import MultimodalEmbedProcessor
from db_utils import VectorDBManager

# Initialize the multimodal processor and database manager
processor = MultimodalEmbedProcessor()
db_manager = VectorDBManager()

db_manager.report_stats()

def show_img_from_url(url):
    """
    Display an image from a URL.
    """
    try:
        img = processor.get_online_image(url)
        st.image(img, caption="Image from URL", use_column_width=True)
    except Exception as e:
        st.error(f"Error fetching image: {e}")

# Title
st.title("🔎 Multimodal Search (Image + Text)")
st.write("Upload an image and/or enter a text query. It will return matching images with captions.")

# Inputs
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
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
            img_query_embedding = processor.embed_image_from_path(uploaded_image)
            img_query, txt_query = db_manager.query_by_embedding(img_query_embedding)
        if query_text:
            txt_query_embedding = processor.embed_text(query_text)
            img_query, txt_query = db_manager.query_by_embedding(txt_query_embedding)
        
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