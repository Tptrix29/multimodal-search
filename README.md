# multimodal-search

## Introduction

This project implements a multi-modal search system that allows users to query using either text or images to retrieve relevant product results based on semantic similarity. Using **CLIP**, a model that maps both text and images into a shared embedding space, we compute cosine similarity between query and dataset embeddings for retrieval.

A subset of the **Amazon Products Dataset** is used for testing, with embeddings stored in a vector database like **Qdrant** or **ChromaDB**. The user interface is built with **Gradio** or **Streamlit**, demonstrating an end-to-end pipeline for cross-modal product search.

## Tech Stack

- **Data Processing:** pandas  
- **Encoding Model:** [CLIP](https://github.com/openai/CLIP) for unified text/image embedding  
- **Vector Database:** [Qdrant](https://qdrant.tech/documentation/) / [ChromaDB](https://docs.trychroma.com/docs/overview/introduction)  
- **Retrieval:** [LangChain](https://python.langchain.com/docs/integrations/retrievers/) / [LlamaIndex](https://docs.llamaindex.ai/en/stable/)  
- **Web UI:** [Gradio](https://www.gradio.app/docs/gradio/interface) / [Streamlit](https://docs.streamlit.io/)

## Dataset

[Amazon product dataset](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products) on Kaggle
> Use subset sample to ensure testing efficiency

## Deliverable

A webpage with search functionality, deployed on GitHub Pages:

- **Input:** text / image  
- **Output:** results retrieved by embedding cosine similarity

## Timeline

- **Day 1:** Data Processing, Embedding Generation & Storage  
- **Day 2:** Retrieval Pipeline Setup  
- **Day 3:** UI Implementation  
- **Day 4:** Evaluation based on ranking (Optional)
