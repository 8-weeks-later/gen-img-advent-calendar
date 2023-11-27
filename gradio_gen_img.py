# code reference from https://learn.deeplearning.ai/huggingface-gradio/lesson/4/image-generation-app

import gradio as gr 
import os
from io import BytesIO
from PIL import Image
import base64 
import requests, json
import openai

with open('api_key.txt') as f:
    private_key = f.read()
    
openai.api_key = private_key

#Text-to-image endpoint
def get_completion(inputs):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=inputs,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

# A helper function to convert the PIL image to base64
def base64_to_pil(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def generate(prompt):
    output = get_completion(prompt)
    result_image = base64_to_pil(output)
    return result_image

gr.close_all()
demo = gr.Interface(fn=generate,
                    inputs=[gr.Textbox(label="Your prompt")],
                    outputs=[gr.Image(label="Result")],
                    title="🎄Image Advent Calendar🎄",
                    description="Generate image for the end of the year",
                    allow_flagging="never",
                    examples=["christmas tree in a cozy room","Twinkling lights on a winter street"])

demo.launch(share=True)