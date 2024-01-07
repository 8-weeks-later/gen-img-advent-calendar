# code reference from https://learn.deeplearning.ai/huggingface-gradio/lesson/4/image-generation-app
# code reference from https://learn.deeplearning.ai/huggingface-gradio/lesson/6/chat-with-any-llm

import gradio as gr 
from io import BytesIO
from PIL import Image
import requests, json
from openai import OpenAI
from datetime import datetime
import base64

with open('api_key.txt') as f:
    private_key = f.read()

client = OpenAI(
    api_key=private_key,
)

# Text-to-image endpoint
def run_dall_e(message):
    response = client.images.generate(
        model="dall-e-3",
        prompt=message,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response.data[0].url

# convert response to PIL image
def response_to_pil(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# convert the PIL image to base64
def image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# generate image
def gen_img(date, input_prompt):
    try:
        prompt = f"A winter picture matching {date} and mood of this conversation: {input_prompt}"
        output = run_dall_e(prompt)
        return output
    except Exception as e:
        print(f"Error generating image: {e}")
        return None
    
# Convert turn to prompt
def convert_history_to_prompt(message, chat_history):
    guide = """
        You're a bot that needs to ask questions to guess the mood of the user's input message. 
        Please answer of the user's input message in one or two sentences. 
        In response to user's third message, please don't ask questions.
    """
    prompt = [{"role": "system", "content": guide}]
    prompt.append({"role":"assistant", "content":"How was your day?"})
    for turn in chat_history:
        user_message, bot_message = turn
        prompt.append({"role":"user", "content":user_message})
        prompt.append({"role":"assistant", "content":bot_message})
    prompt.append({"role":"user", "content": message})
    return prompt

# generate chatbot response
def chatbot_response(message, chat_history):    
    input_prompt = convert_history_to_prompt(message, chat_history)
    if len(chat_history) < 3:
        response = client.chat.completions.create(model="gpt-4", messages=input_prompt)
        bot_message = response.choices[0].message.content
        if len(chat_history) == 2:
            question = "Okay, I will make an image for you. Is there any specific object you want to see in the image?"
            bot_message += f"<br><br>{question}"
        chat_history.append((message, bot_message))
    else:
        image_url = gen_img(datetime.today().strftime("%Y-%m-%d"), input_prompt)
        image_html = f"<img src='{image_url}'/>"
        bot_message = f"This is the image I made for you:)"
        image_html += f"<br><br>{bot_message}"
        chat_history.append((message, image_html))
    return "", chat_history

with gr.Blocks() as demo:
    date = datetime.today().strftime("%Y-%m-%d")
    gr.Markdown(
        f"""
        # 🎄Image Advent Calendar🎄
        Hello! I am image advent calendar. <br>
        I can generate an image for you. <br>
        Today is {date}. <br>
        How was your day?
        """
    )
    
    chatbot = gr.Chatbot(label="Conversation")
    msg = gr.Textbox(label="User Input")
    btn = gr.Button("Submit")

    btn.click(chatbot_response, [msg, chatbot], [msg, chatbot])
    msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])
    
gr.close_all()
demo.launch(share=True)
