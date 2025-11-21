from os.path import join, dirname
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import re
import numpy as np

class TeleReasoning():
    def __init__(self, args):
        self.args = args.preprocess
        self.device = args.device.trans_config
        self.current_dir = dirname(dirname(__file__))
        self.model_path = join(self.current_dir, self.args.model_path)
        self.max_len = self.args.resize_length

        self.init_telellm()
        self.init_message()


    def init_telellm(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        torch.cuda.empty_cache()


    def init_message(self):
        self.translate_message = "Translate the following text into English: {}."
        self.noun_phrase_message = """
                Classify the following text as either a 'noun phrase' (describing a thing/concept) or an 'instruction' (directing an action). Respond ONLY with 'noun phrase' or 'instruction'.

                Examples:

                'the blue car' → noun phrase
                'open the door' → instruction

                Text to classify: {}
                """
        self.instruction_parse_message = """
                Analyze the given text and extract the following details for each detected operation:

                Operation Type: Identify if the action is 'add', 'replace', or 'remove' based on:
                Increase: Phrases like "increase A", "add A", "raise A".
                Replace: Phrases like "replace A with B", "substitute A for B".
                Delete: Phrases like "delete A", "remove A".
                Associated Object: Return the object(s) involved in the operation:
                For increase/delete, return the object A.
                For replace, return the target object B (the new value replacing A).

                Output Format:

                Return results as a structured pair: [operation_type, object].
                If no operation is detected, return None.

                Examples:

                Input: "Increase the budget." → Output: ["increase", "the budget"]
                Input: "Replace old code with new code." → Output: ["replace", "new code"]
                Input: "Delete outdated records." → Output: ["delete", "outdated records"]
                Input: "Review the document." → Output: None

                Now process the following text: {}
                """
    

    def generate_message(self, operation, prompt):
        if operation == 'translate':
            return [{"role": "user", "content": self.translate_message.format(prompt)}]
        elif operation == 'noun_phrase':
            return [{"role": "user", "content": self.noun_phrase_message.format(prompt)}]
        elif operation == 'instruction':
            return [{"role": "user", "content": self.instruction_parse_message.format(prompt)}]
        
    
    def telechat(self, operation, prompt):
        messages = self.generate_message(operation, prompt)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
            )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=75
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        torch.cuda.empty_cache()
        return content


    def expand_image_and_mask(self, input, mask):
        input = input.convert('RGB')
        max_side = max(input.width, input.height)

        ratio = self.max_len / max_side
        new_width, new_height = ratio * input.width, ratio * input.height

        W = int(new_width - new_width % 16)
        H = int(new_height - new_height % 16)
        input_resized = input.resize((W,H))
        mask_resized = mask.resize((W,H))
        
        return input_resized, mask_resized


    def preprocess_mask(self, img):         # 根据图像的不同类型，做相应处理
        if img.mode == 'RGBA':
            bw_img = Image.new('L', img.size, color=255)  # 默认填充白色
            r, g, b, a = img.split()
            bw_img.paste(0, mask=a)  # 0=黑色，mask=a 表示用 Alpha 通道作为遮罩
            return bw_img
        elif img.mode == 'L':
            return img
        elif img.mode == 'RGB':
            mask = img.convert("L")
            return mask
    

    def is_chinese(self, text):
        if text:
            pattern_cn = re.compile(r'[\u4e00-\u9fa5]')
            return pattern_cn.search(text)
        else:               # 如果为空
            return False


    def run(self, prompt, image, mask):
        
        if prompt: # 如果提示词不为空
            ## 转换为英文
            if self.is_chinese(prompt):
                en_text = self.telechat('translate', prompt)
            else:
                en_text = prompt
            ## 判断是否为名词性短语
            noun_text = self.telechat('noun_phrase', en_text)
            if noun_text == 'noun phrase':
                operation, text = 'add', en_text
            elif noun_text == 'instruction':
                instruction_parse = self.telechat('instruction', en_text)
                operation, text = [eval(j) for j in instruction_parse.split('\n')][0]       #  仅处理第一个操作
        else:
            operation, text = 'remove', prompt
        # 处理图像和蒙版
        mask = self.preprocess_mask(mask)
        new_img, new_mask = self.expand_image_and_mask(image, mask)

        return operation.lower(), text, new_img, new_mask