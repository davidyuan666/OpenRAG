# -*- coding = utf-8 -*-
# @time:2024/7/29 10:55
# Author:david yuan
# @File:fastchat_client.py
# @Software:VeSync


class FastChatClient(object):
    def __init__(self, model="localmodel", host="localhost", port=8888):
        self.model = model
        self.host = host
        self.port = port

    def chat(self, query, history=list(), system="", temperature=0.0, stop="", *args, **kwargs):
        url = f'http://{self.host}:{self.port}/v1/completions/'

        headers = {"Content-Type": "application/json"}
        if "baichuan" in self.model:
            prompt = self.make_baichuan_prompt(query, system, history)
        elif "qwen" in self.model:
            prompt = self.make_qwen_prompt(query, system, history)
        else:
            prompt = self.make_prompt(query, system, history)
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.75,
            "top_k": 40,
            "max_tokens": 512
        }
        resp = requests.post(url=url, json=data, headers=headers)
        response = resp.json() # Check the JSON Response Content documentation below
        response_text = response['choices'][0]['text']

        new_history = history[:] + [[query, response_text]]
        return response_text, new_history

    @staticmethod
    def make_prompt(query, system, history):
        if not history:
            history = list()
        if system:
            prompt = system + "\n"
        else:
            prompt = ''
        for q, r in history:
            prompt += 'User:' + q + '\nAssistant' + r + "\n"
        prompt += query
        return prompt

    @staticmethod
    def make_baichuan_prompt(query, system, history):
        if not history:
            history = list()
        if system:
            prompt = system + "\n"
        else:
            prompt = ''
        for q, r in history:
            prompt += '<reserved_106>' + q + '<reserved_107>' + r
        prompt += query
        return prompt

    @staticmethod
    def make_qwen_prompt(query, system, history):
        if not history:
            history = list()
        if system:
            prompt = '<|im_start|>' + system + '<|im_end|>\n'
        else:
            prompt = ''
        for q, r in history:
            response = r if r else ''
            prompt += '<|im_start|>user\n' + q + '<|im_end|>\n<|im_start|>assistant\n' + response + "<|im_end|>\n"
        prompt += query
        return prompt
