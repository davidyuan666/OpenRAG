# -*- coding = utf-8 -*-
# @time:2024/7/23 17:59
# Author:david yuan
# @File:audio_util.py
# @Software:VeSync

from datetime import datetime
import os

'''
pip install python-dotenv
'''
from openai import OpenAI
import openai
from configs.config import VESYNC_OPENAI_KEY


os.environ["OPENAI_API_KEY"]=VESYNC_OPENAI_KEY
openai.api_key=os.environ["OPENAI_API_KEY"]
client = OpenAI()


def text_to_speech(input_text):
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(os.path.join(os.getcwd(),'text2speech_temp')):
            os.makedirs(os.path.join(os.getcwd(),'text2speech_temp'))
        speech_file_path = os.path.join(os.getcwd(), 'text2speech_temp', f'{timestamp}.m4a')
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=input_text
        )

        response.stream_to_file(speech_file_path)
        return speech_file_path
    except Exception as e:
        print(f"An error occurred in text_to_speech: {e}")
        return None


def speech_to_text(speech_file_path):
    try:
        audio_file = open(speech_file_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
        return transcription.text
    except Exception as e:
        print(f"An error occurred in speech_to_text: {e}")
        return None

'''
use flutter speech to text
'''
def local_speech_to_text():
    pass

