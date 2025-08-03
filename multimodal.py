from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch


class MultimodalEmbedProcessor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("Device:", self.model.device)
    
    def get_online_image(self, url):
        """
        Fetch an image from a URL and return it as a PIL Image object.
        """
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    def multimodal_embed(self, imgUrl, text):
        """
        Generate multimodal embeddings for a given image URL and text.
        """
        image = self.get_online_image(imgUrl)
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        
        image_embedding = outputs.image_embeds.squeeze(0).detach().numpy()
        text_embedding = outputs.text_embeds[0].squeeze(0).detach().numpy()
        
        return image_embedding, text_embedding
    
    def embed_text(self, text):
        """
        Generate text embeddings.
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        outputs = self.model.get_text_features(**inputs).to("cpu")
        return outputs.squeeze().detach().numpy()

    def embed_image_from_url(self, imgUrl):
        """
        Generate image embeddings for a given image URL.
        """
        image = self.get_online_image(imgUrl)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        outputs = self.model.get_image_features(**inputs).to("cpu")
        return outputs.squeeze().detach().numpy()

    def embed_image_from_path(self, imgPath):
        """
        Generate image embeddings for a local image file.
        """
        image = Image.open(imgPath)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().numpy()[0]
