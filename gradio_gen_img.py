# code reference from https://learn.deeplearning.ai/huggingface-gradio/lesson/4/image-generation-app
# code reference from https://learn.deeplearning.ai/huggingface-gradio/lesson/6/chat-with-any-llm

import gradio as gr 
from io import BytesIO
from PIL import Image
import requests, json
from openai import OpenAI
from datetime import datetime

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

# A helper function to convert the PIL image to base64
def base64_to_pil(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

# From prompt to base64 image output 
def gen_img(date, input_prompt):
    try:
        # 이 부분 테스트해보고 보완하기
        prompt = f"A winter picture matching {date} and mood of this conversation : {input_prompt}"
        output = run_dall_e(prompt)
        result_image = base64_to_pil(output)
        return result_image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

# Convert turn to prompt
def convert_history_to_prompt(message, chat_history):
    prompt = []
    for turn in chat_history:
        user_message, bot_message = turn
        prompt.append({"role":"user", "content":user_message})
        prompt.append({"role":"assistant", "content":bot_message})
    prompt.append({"role":"user", "content": message})
    return prompt

# Basic chatbot endpoint
def chatbot_response(message, chat_history):
    input_prompt = convert_history_to_prompt(message, chat_history)
    print(len(chat_history))
    if len(chat_history) < 2:
        response = client.chat.completions.create(model="gpt-3.5-turbo", messages=input_prompt)
        bot_message = response.choices[0].message.content
        chat_history.append((message, bot_message))
        return "", chat_history
    else:
        # Extract mood from the message
        image = gen_img(datetime.today().strftime("%Y-%m-%d"), input_prompt)
        return "", image


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎄Image Advent Calendar🎄
        Generate image for the end of the year.
        """
    )
    # count 변수로 output 포맷 구분하기
    # turn_count = gr.State([])
    chatbot = gr.Chatbot() #just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(chatbot_response, [msg, chatbot], [msg, chatbot])
    msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot]) #Press enter to submit


gr.close_all()
demo.launch(share=True)


# demo = gr.Interface(fn=gen_img,
#                     inputs=[gr.Textbox(label="Your prompt")],
#                     outputs=[gr.Image(label="Result")],
#                     title="🎄Image Advent Calendar🎄",
#                     description="Generate image for the end of the year",
#                     allow_flagging="never",
#                     examples=["christmas tree in a cozy room","Twinkling lights on a winter street"])
