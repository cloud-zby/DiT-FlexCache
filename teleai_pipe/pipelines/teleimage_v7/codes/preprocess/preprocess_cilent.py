from os.path import join, dirname
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

import torch
import re
from openai import OpenAI
from .preprocess import TeleReasoning 
# Set OpenAI's API key and API base to use vLLM's API server.
import os
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:11451/v1"


class TeleReasoningCilent(TeleReasoning):
    def __init__(self, args, device):
        
        super().__init__(args, device=None)
        assert args.cilent_base is not None, "Please set the client base url"
        
        

    def init_telellm(self):
        #print("--------加载LLM---------")
        self.openai_api_base = self.args.cilent_base
        # print("模型路径: ", self.model_path, "到", self.device)

        # Initialize the vLLM engine
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=self.openai_api_base,
        )
        #print("--------------------------\n\n\n")


    
    def telechat(self, operation, prompt):
        messages = self.generate_message(operation, prompt)
        # Generate outputs

        chat_response = self.client.chat.completions.create(
            model=self.args.served_model_name,
            messages=messages,
            max_tokens=712,
            temperature=0.6,
            top_p=0.95,
            extra_body={
                "top_k": 20,
            },
        )
        chat_response_content = chat_response.choices[0].message.content
        #print(chat_response_content)
        return chat_response_content
