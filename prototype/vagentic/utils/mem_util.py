# -*- coding = utf-8 -*-
# @time:2024/7/17 19:46
# Author:david yuan
# @File:mem_util.py
# @Software:VeSync


import os


'''
function call
'''
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

mem_path = os.path.join(log_directory, "MemFile.txt")

def write_to_memory(memory):
    mode = 'w' if not os.path.exists(mem_path) else 'a'  # Use 'a' for append if file exists, 'w' to write if not
    try:
        with open(mem_path, mode) as f:
            f.write(memory + '|')
        return "Memory written successfully."
    except IOError as e:
        print(f"Error writing to memory: {e}")
        return "Failed to write memory."

def read_from_memory():
    try:
        if os.path.exists(mem_path):
            with open(mem_path, 'r') as f:
                data = f.read()
            return data if data else "MEMORY FILE IS EMPTY"
        else:
            return "NO LONG TERM MEMORY FOUND"
    except IOError as e:
        print(f"Error reading from memory: {e}")
        return "ERROR ACCESSING MEMORY"



def reset_memory():
    try:
        if os.path.exists(mem_path):
            # Open the file in write mode to clear its contents
            with open(mem_path, 'w') as f:
                f.truncate(0)  # Clear the file if you prefer explicitly clearing
            return "Memory cleared successfully"
        else:
            return "NO LONG TERM MEMORY FOUND"
    except IOError as e:
        print(f"Error accessing memory: {e}")
        return "ERROR ACCESSING MEMORY"