import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Load metadata and ensure consistent ordering
df = pd.read_csv("./archive/amazon_products.csv").reset_index(drop=True)
df.head()


df.query("category_id == 1").head()

# Retrieve image url links
url = df["imgUrl"][0]

# Request image url links
response = requests.get(url)

# Check if request successful
if response.status_code == 200:
    # Load the image
    image = Image.open(BytesIO(response.content ))
    image.show()
    image.save("amazon_image.webp")
else:
    print("Failed to request the image.")


