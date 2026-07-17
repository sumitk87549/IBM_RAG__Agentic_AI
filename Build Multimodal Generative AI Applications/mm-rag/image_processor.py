import torch
import os
import  requests, base64
import numpy as np
from io import BytesIO
from PIL import Image
from torchvision.transforms import transforms
from torchvision.models import resnet56
from sklearn.metrics.pairwise import cosine_similarity

class ImageProcessor:
    def __init__(self, image_size=(224,244),norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet56(pretrained=True).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    
    def encode_image(self, image_input, is_url=True):
        try:
            if is_url:
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")

            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            base64_string = base64.b64decode(buffered.getvalue()).decode("utf-8")

            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(input_tensor)

            feature_vectors = features.cpu().numpy().flatten()

            return {"base64": base64_string , "vector":feature_vectors}
        except Exception as e:
            print(f"Error:\n{e}")

    def find_closest_match(self, user_vector, dataset):
        try:
            vectors = np.vstack(dataset["Embedding"].dropna().values)
            similarities = cosine_similarity(user_vector.reshape(1,-1), vectors)

            closest_index = np.argmax(similarities)
            similarity_score = similarities[0][closest_index]

            closest_row = dataset.iloc[closest_index]
            return closest_row, similarity_score

        except Exception as e:
            print(f"Error in finding_closest_match:\n{e}")
        