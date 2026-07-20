import requests
import re
import os
import base64
from PIL import Image
import ollama
from flask import Flask, render_template, request, redirect, flash, url_for
from langchain_ollama import ChatOllama
import random
import string

app = Flask(__name__)
app.secret_key = ''.join(random.choices(string.ascii_letters + string.digits, k=24))

model = ChatOllama(
    model="llava:7b",
    base_url="http://localhost:11434",
    temperature=0.7
)

def encode_input_image(uploaded_image):
    if uploaded_image is not None:
        image_bytes = uploaded_image.read()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        return encoded_image
    else:
        raise FileNotFoundError("No file uploaded")

def format_response(response_text):
    response_text = re.sub(r"\*\*(.*?)\*\*", r"<p><strong>\1</strong></p>", response_text) # Replace section headers that are bolded with '**' to HTML paragraph tags with bold text
    response_text = re.sub(r"(?m)^\s*\*\s(.*)", r"<li>\1</li>", response_text)     # Convert bullet points denoted by "*" to HTML list items
    response_text = re.sub(r"(<li>.*?</li>)+", lambda match: f"<ul>{match.group(0)}</ul>", response_text, flags=re.DOTALL)     # Wrap list items within <ul> tags for proper HTML structure and indentation
    response_text = re.sub(r"</p>(?=<p>)", r"</p><br>", response_text)     # Ensure that all paragraphs have a line break after them for better separation
    response_text = re.sub(r"(\n|\\n)+", r"<br>", response_text)     # Ensure the disclaimer and other distinct paragraphs have proper line breaks
    return response_text


def generate_model_response(encoded_image, query, assistant_prompt ):
    messages = [{"role":"user",
            "content": [
                {"type": "text", "text":assistant_prompt + "\n\n" + query},
                {"type":"image_url", "image_url": {"url" : "data:image/jpeg;base64," + encoded_image }}]
            }]
    try:
        response = model.chat(messages = messages)
        return format_response(response['message']['content'])
    except Exception as e:
        print(f" Error:\n {e}")

@app.route('/', methods=['GET'])
def STARTER():
    return 'Hello, World!'

@app.route("/generate", methods=["GET",'POST'])
def index():
    assistant_prompt = """
            You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

        1. **Identification**: List each identified food item clearly, one per line.
        2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
        - **[Food Item]**: [Portion Size], [Number of Calories] calories

        Example:
        *   **Salmon**: 6 ounces, 210 calories
        *   **Asparagus**: 3 spears, 25 calories

        3. **Total Calories**: Provide the total number of calories for all food items.

        Example:
        Total Calories: [Number of Calories]

        4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

        Example:
        *   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

        5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

        6. **Disclaimer**: Include the following exact text as a disclaimer:

        The nutritional information and calorie estimates provided are approximate and are based on general food data. 
        Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations. 
        For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

        Format your response exactly like the template above to ensure consistency.

        """
    
    if request.method == 'POST':
        query = request.form.get("user_query")
        uploaded_file = request.files.get("file")
        if uploaded_file:
            encoded_image = encode_input_image(uploaded_image=uploaded_file)
        else:
            print("failed to encode image")
            return
        response = generate_model_response(encoded_image=encoded_image, query=query, assistant_prompt=assistant_prompt)
        return render_template("index.html", user_query=query, response=response)
    else:
        flash("upload Image", "warning")
        return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)