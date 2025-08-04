from chromadb import PersistentClient

class ChromaDBVector:
    def __init__(self):
        self.client = PersistentClient(path="./chroma_db")
        self.image_collection_name = "images"
        self.text_collection_name = "texts"
        self.init_collections()
        
    def init_collections(self):
        """
        Create/Retreive collections of images & texts from ChromaDB
        Ensures both collections are available before later operations
        """
        self.image_collection = self.client.get_or_create_collection(self.image_collection_name)
        self.text_collection = self.client.get_or_create_collection(self.text_collection_name)
        print(f"Initialized collections: {self.image_collection_name}, {self.text_collection_name}")
        
    def product_embedding(self, image_embedding, text_embedding, metadata):
        """
        Adds both image and text embeddings into respective ChromaDB collections using metadata["id"] as shared identifiers
        Enables cross-modal retrieval and allows product to be searched via text/image query
        """
        self.image_collection.add(
            embeddings =[image_embedding],
            metadatas = [metadata],
            ids = [metadata["id"]]
        )
        
        self.text_collection.add(
            embeddings = [text_embedding],
            metadatas = [metadata],
            ids = [metadata["id"]]
        )
        print(f"Add embeddings of Product ID: {metadata['id']} to collections")

    
    def query_return(self, embedding, k=5):
        """
        Performs a query using a vector against both collections and returns top K similar items 
        Evaluate how close a new item to the previously stored items from image & text modalities
        """
        image_query = self.image_collection.query(
            query_embeddings = [embedding],
            n_results = k,
            include = ["embeddings", "metadatas"]
        )
        
        text_query = self.text_collection.query(
            query_embeddings = [embedding],
            n_results = k,
            include = ["embeddings", "metadatas"]
        )
        return image_query, text_query

    
    def clear_cache(self):
        """
        Deletes image and text collections and reinitializes them
        Used when starting over with new dataset
        """
        self.client.delete_collection(self.image_collection_name)
        self.client.delete_collection(self.text_collection_name)
        self.init_collections()
        print("Cache cleared for both collections")
        
    def stats(self):
        """
        Prints out the number of items in each collection
        Used for monitoring the size of database
        """
        image_number = self.image_collection.count()
        text_number = self.text_collection.count()
        print(f"Image Collection Total Number: {image_number}")
        print(f"Text Collection Total Number: {text_number}")
        
    def rearrange(self):
        "Placeholder function"
        pass