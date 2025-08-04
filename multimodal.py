# Multi-Modal Searching Packages
import requests
from io import BytesIO
# CLIP Model based Processor Packages
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class MultimodalEmbedding:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("Device:", self.model.device)
        
    def get_online_image(self, url):
        """
        Downloads an image from a given URL using requests
        Converts it to a PIL Image object for processing
        """
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    
    def multimodal_embed(self, imgUrl, text):
        """
        Takes an image URL and a text string
        Encodes both simultaneously using processor
        Return both image vector embedded and text vector embedded
        """
        image = self.get_online_image(imgUrl)
        inputs = self.processor(text=[text], images = image, return_tensors = "pt", padding = True)
        outputs = self.model(**inputs)
        
        image_embedding = outputs.image_embeds.squeeze(0).detach().numpy()
        text_embedding = outputs.text_embeds[0].squeeze(0).detach().numpy()
        
        return image_embedding, text_embedding 
    
    def embed_text(self, text):
        """
        Encodes a text string using model.get_text_features()
        Returns a NumPy vector of the text
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding = True)
        outputs = self.model.get_text_features(**inputs).to("cpu")
        return outputs.squeeze().detach().numpy()
    
    def embed_image_url(self, Url):
        """
        Downloads an image from a URL link
        Returns its CLIP embedding vector
        """
        image = self.get_online_image(Url)
        inputs = self.processor(images = image, return_tensors = "pt", padding = True)
        outputs = self.model.get_image_features(**inputs).to("cpu")
        return outputs.squeeze().detach().numpy()
    
    def embed_image_path(self, path):
        """
        Loads an image from a local file path
        Returns its CLIP embedding vector
        """
        image = Image.open(path)
        inputs = self.processor(images = image, return_tensors = "pt", padding = True)
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().numpy()[0]