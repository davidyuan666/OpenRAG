# -*- coding = utf-8 -*-
# @time:2024/7/25 11:52
# Author:david yuan
# @File:message_type.py
# @Software:VeSync


class HumanMessage:
    def __init__(self, message):
        self.message = {"role": "user", "content": message}

    @property
    def content(self):
        return self.message



class SystemMessage:
    def __init__(self, message):
        self.message = {"role": "system", "content": message}

    @property
    def content(self):
        return self.message


class AssistantMessage:
    def __init__(self, message):
        self.message = {"role": "assistant", "content": message}

    @property
    def content(self):
        return self.message

