# code reference from https://learn.deeplearning.ai/huggingface-gradio/lesson/4/image-generation-app
# code reference from https://learn.deeplearning.ai/huggingface-gradio/lesson/6/chat-with-any-llm

import gradio as gr 
import os
from io import BytesIO
from PIL import Image
import base64 
import requests, json
from openai import OpenAI

with open('api_key.txt') as f:
    private_key = f.read()

client = OpenAI(
    api_key=private_key,
)

# Convert turn to prompt
def format_chat_prompt(message, chat_history):
    prompt = []
    for turn in chat_history:
        user_message, bot_message = turn
        prompt.append({"role":"user", "content":user_message})
        prompt.append({"role":"assistant", "content":bot_message})
    prompt.append({"role":"user", "content": message})
    return prompt

# Basic chatbot endpoint
def chatbot_response(message):
    formatted_prompt = format_chat_prompt(message, [])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=formatted_prompt
    )
    chat_message = response.choices[0].message.content
    return [(message, chat_message)]

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎄Image Advent Calendar🎄
        Generate image for the end of the year.
        """
    )
    chatbot = gr.Chatbot() #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(chatbot_response, inputs=msg, outputs=chatbot)
    msg.submit(chatbot_response, inputs=msg, outputs=chatbot) #Press enter to submit


gr.close_all()
demo.launch(share=True)

# # Text-to-image endpoint
# def run_dall_e(message):
#     response = client.images.generate(
#         model="dall-e-3",
#         prompt=message,
#         size="1024x1024",
#         quality="standard",
#         n=1,
#     )
#     return response.data[0].url

# # A helper function to convert the PIL image to base64
# def base64_to_pil(image_url):
#     response = requests.get(image_url)
#     img = Image.open(BytesIO(response.content))
#     return img

# # From prompt to base64 image output 
# def gen_img(message):
#     output = run_dall_e(message)
#     result_image = base64_to_pil(output)
#     return result_image


# demo = gr.Interface(fn=gen_img,
#                     inputs=[gr.Textbox(label="Your prompt")],
#                     outputs=[gr.Image(label="Result")],
#                     title="🎄Image Advent Calendar🎄",
#                     description="Generate image for the end of the year",
#                     allow_flagging="never",
#                     examples=["christmas tree in a cozy room","Twinkling lights on a winter street"])
