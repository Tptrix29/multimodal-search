import chromadb

class VectorDBManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.image_collection_name = "images"
        self.text_collection_name = "texts"
        self.init_collections()
        
    def init_collections(self):
        """
        Initialize the image and text collections.
        """
        self.image_collection = self.client.get_or_create_collection(self.image_collection_name)
        self.text_collection = self.client.get_or_create_collection(self.text_collection_name)
        print(f"Initialized collections: {self.image_collection_name}, {self.text_collection_name}")

    def add_product_embedding(self, image_embedding, text_embedding, metadata):
        """
        Add product embeddings and metadata to the collections.
        """
        self.image_collection.add(
            embeddings=[image_embedding],
            metadatas=[metadata],
            ids=[metadata['id']]
        )
        
        self.text_collection.add(
            embeddings=[text_embedding],
            metadatas=[metadata],
            ids=[metadata['id']]
        )

        print(f"Added embeddings for product ID: {metadata['id']}")
    
    def query_by_embedding(self, embedding, k=5):
        """
        Query the image collection by embedding.
        """
        img_query = self.image_collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["embeddings", "metadatas"]
        )
        txt_query = self.text_collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["embeddings", "metadatas"]
        )
        return img_query, txt_query
    
    def clear_cache(self):
        """
        Clear the cache of the collections.
        """
        self.client.delete_collection(self.image_collection_name)
        self.client.delete_collection(self.text_collection_name)
        self.init_collections()
        print("Cache cleared for both collections.")
    
    def report_stats(self):
        """
        Report the number of items in each collection.
        """
        img_count = self.image_collection.count()
        txt_count = self.text_collection.count()
        print(f"Image Collection Count: {img_count}")
        print(f"Text Collection Count: {txt_count}")
    
    def rearrange_results(self):
        pass