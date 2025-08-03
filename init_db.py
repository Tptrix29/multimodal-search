import pandas as pd
from multimodal import MultimodalEmbedProcessor
from db_utils import VectorDBManager

if __name__ == "__main__":
    embedder = MultimodalEmbedProcessor()

    data = pd.read_csv("./data/amazon_products.csv")
    data.head()

    manager = VectorDBManager()
    manager.report_stats()

    # Uncomment to initialize the collections
    manager.clear_cache()

    # Add product embedding to the collections
    data_count = 20
    for i in range(data_count):
        imgUrl = data['imgUrl'][i]
        title = data["title"][i]
        image_embedding, text_embedding = embedder.multimodal_embed(imgUrl, title)
        
        metadata = {
            "id": str(data.index[i]),
            "title": title,
            "imgUrl": imgUrl
        }
        
        manager.add_product_embedding(image_embedding, text_embedding, metadata)