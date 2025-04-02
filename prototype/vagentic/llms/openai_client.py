# -*- coding = utf-8 -*-
# @time:2024/7/29 10:54
# Author:david yuan
# @File:openai_client.py
# @Software:VeSync


import logging
import os
import requests
import sys
import time
import traceback
import openai
from config.Config import OpenAIkey
def make_gpt_messages(query, system, history):
    msgs = list()
    if system:
        msgs.append({
            "role": "system",
            "content": system
        })
    for q, a in history:
        msgs.append({
            "role": "user",
            "content": str(q)
        })
        msgs.append({
            "role": "assistant",
            "content": str(a)
        })
    msgs.append({
        "role": "user",
        "content": query
    })
    return msgs


class OpenAIClient(object):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        openai.api_type = 'open_ai'
        openai.api_key = OpenAIkey

    def chat(self, query, history=list(), system="", temperature=0.0, stop="", *args, **kwargs):
        msgs = make_gpt_messages(query, system, history)

        try:
            if openai.api_type == "open_ai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=msgs,
                    temperature = temperature,
                    stop=stop
                    )

            response_text = response['choices'][0]['message']['content']
        except:
            print(traceback.format_exc())
            response_text = ""

        new_history = history[:] + [[query, response_text]]
        return response_text, new_history

