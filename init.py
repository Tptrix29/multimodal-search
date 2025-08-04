import pandas as pd
from multimodal import MultimodalEmbedding
from ChromaDB import ChromaDBVector

if __name__ == "__main__":
    embedder = MultimodalEmbedding()
    
    data = pd.read_csv("./archive/amazon_products.csv")
    data.head()
    
    storage = ChromaDBVector()
    storage.stats()
    
    # Uncomment to initialize the collections
    storage.clear_cache()
    
    # Add product embedding to the collections
    pd_count = 20
    for i in range(pd_count):
        imgUrl = data['imgUrl'][i]
        title = data['title'][i]
        image_embedding, text_embedding = embedder.multimodal_embed(imgUrl, title)
        
        metadata = {
            "id":str(data.index[i]),
            "title":title,
            "imgUrl": imgUrl
        }
        
        storage.product_embedding(image_embedding, text_embedding, metadata)