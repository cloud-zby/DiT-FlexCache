import math
import random
import sys
from dataclasses import dataclass
from typing import List, Union
from PIL import Image
from openai import OpenAI
from teleai_pipe.utils.video import images_to_base_64
from teleai_pipe.prompters.system_prompt import SYSTEM_PROMPT_TYPES
from loguru import logger
import math
import random
import sys
from dataclasses import dataclass
from typing import List, Union
from PIL import Image
from openai import OpenAI
from teleai_pipe.utils.video import images_to_base_64
from teleai_pipe.prompters.system_prompt import SYSTEM_PROMPT_TYPES,S2V_ZH_STD_SYS_PROMPT,S2V_EN_STD_SYS_PROMPT
from loguru import logger

# 可选：定义动作规则
# 动作规则优化版（短描述，突出动作与结束状态）
ACTION_RULES = {
    # 跳水 / Diving
    "跳水": "完整表现动作，最后向下跳入水中消失，镜头跟随下移，描写人物姿势与服装。",
    "diving": "Show full dive; diver jumps downward into the pool and disappears at the end, camera follows the motion, include pose and clothing details.",
    "diver": "Show full dive; diver jumps downward into the pool and disappears at the end, camera follows the motion, include pose and clothing details.",

    # # 跑步 / Running
    # "跑步": "突出前倾姿势和双腿交替，展现运动感。",
    # "running": "Emphasize forward-leaning posture and alternating legs for motion.",
    # "runner": "Emphasize forward-leaning posture and alternating legs for motion.",

    # # 跳舞 / Dancing
    # "跳舞": "突出身体协调与优雅，姿态自然流畅。",
    # "dancing": "Highlight coordination and elegance; ensure smooth poses.",
    # "dancer": "Highlight coordination and elegance; ensure smooth poses.",
}

def build_messages(system_prompt, prompt, image=None, action_rule=None, max_words=200):
    """
    简洁版消息模板：
    - 突出动作
    - 融合原prompt和图片
    - 限制输出字数（英文或中文）
    """
    word_limit_text = f"请将扩展描述压缩到最多 {max_words} 词。" if max_words else ""

    role_content = [
        {
            "type": "text",
            "text": (
                "请生成扩展描述，要求：\n"
                "1. 动作为主线，完整合理，突出最终动作状态。\n"
                "2. 保留原始prompt的语义。\n"
                "3. 如果有图像，请结合画面人物细节。\n"
                f"{action_rule if action_rule else ''}\n"
                f"{word_limit_text}\n"
                f"{prompt}"
            )
        }
    ]

    if image is not None:
        role_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"}
        })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": role_content},
    ]

@dataclass
class PromptOutput(object):
    status: bool
    prompt: str
    seed: int
    system_prompt: str
    message: str

    def add_custom_field(self, key: str, value) -> None:
        self.__setattr__(key, value)


class PromptExpander:
    def __init__(self, model_name, is_vl=False, device=0):
        self.model_name = model_name
        self.is_vl = is_vl
        self.device = device

    def extend_with_img(self, prompt, system_prompt, image=None, seed=-1):
        pass

    def extend(self, prompt, system_prompt, seed=-1):
        pass
    def decide_system_prompt(
        self,
        tar_lang="zh",
        multi_images_input=False,
        movie_style: bool = False,
        empty_prompt: bool = False,
        force_s2v: bool = True,   # 新增
    ):
        if force_s2v:
            if tar_lang.lower() == "zh":
                return S2V_ZH_STD_SYS_PROMPT
            else:
                return S2V_EN_STD_SYS_PROMPT

        zh = tar_lang == "zh"
        if multi_images_input and movie_style:
            logger.warning("no movie style for multi images")
            movie_style = False
        self.is_vl |= multi_images_input
        a = zh
        b = self.is_vl << 1
        c = multi_images_input << 2
        d = movie_style << 3
        e = empty_prompt << 4
        task_type = a + b + c + d + e
        return SYSTEM_PROMPT_TYPES[task_type]
    # def decide_system_prompt(
    #     self,
    #     tar_lang="zh",
    #     multi_images_input=False,
    #     movie_style: bool = False,
    #     empty_prompt: bool = False,
    # ):
    #     zh = tar_lang == "zh"
    #     if multi_images_input and movie_style:
    #         logger.warning("no movie style for multi images")
    #         movie_style = False
    #     self.is_vl |= multi_images_input
    #     a = zh
    #     b = self.is_vl << 1
    #     c = multi_images_input << 2
    #     d = movie_style << 3
    #     e = empty_prompt << 4
    #     task_type = a + b + c + d + e
    #     return SYSTEM_PROMPT_TYPES[task_type]

    def __call__(
        self,
        prompt,
        system_prompt=None,
        tar_lang="zh",
        image=None,
        seed=-1,
        movie_style: bool = False,
    ):
        if system_prompt is None:
            system_prompt = self.decide_system_prompt(
                tar_lang=tar_lang,
                multi_images_input=isinstance(image, (list, tuple)) and len(image) > 1,
                movie_style=movie_style,
                empty_prompt=(prompt == ""),
            )
        if seed < 0:
            seed = random.randint(0, sys.maxsize)
        if image is not None and self.is_vl:
            return self.extend_with_img(prompt, system_prompt, image=image, seed=seed)
        elif not self.is_vl:
            return self.extend(prompt, system_prompt, seed)
        else:
            raise NotImplementedError


class OpenAIPromptExpander(PromptExpander):
    def __init__(
        self,
        address,
        api_key="EMPTY",
        model_name="Qwen2.5-VL-72B-Instruct",
        max_image_size=512 * 512,
        retry_times=4,
        is_vl=False,
        **kwargs,
    ):
        super().__init__(model_name, is_vl, **kwargs)
        self.api_key = api_key
        self.max_image_size = max_image_size
        self.model = model_name
        self.retry_times = retry_times
        self.client = OpenAI(api_key=self.api_key, base_url=address)

    def _get_action_rule(self, prompt: str) -> str:
        for action, rule in ACTION_RULES.items():
            if action in prompt:
                return rule,True
        return None,False

    def extend(self, prompt, system_prompt, seed=-1):
        is_diving = False
        action_rule,is_diving = self._get_action_rule(prompt)
        messages = build_messages(system_prompt, prompt, action_rule=action_rule)

        exception = None
        for _ in range(self.retry_times):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    seed=seed,
                )
                expanded_prompt = response.choices[0].message.content
                return PromptOutput(
                    status=True,
                    prompt=expanded_prompt,
                    seed=seed,
                    system_prompt=system_prompt,
                    message=response.choices[0].message,
                ),is_diving
            except Exception as e:
                logger.warning(f"failed to expand prompt for {e}")
                exception = e
        return PromptOutput(
            status=False,
            prompt=prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception),
        ),is_diving

    def extend_with_img(
        self,
        prompt,
        system_prompt,
        image: Union[List[Image.Image], List[str], Image.Image, str] = None,
        seed=-1,
    ):
        is_diving = False
        def ensure_image(_image):
            if isinstance(_image, str):
                _image = Image.open(_image).convert("RGB")
            w = _image.width
            h = _image.height
            area = min(w * h, self.max_image_size)
            aspect_ratio = h / w
            resized_h = round(math.sqrt(area * aspect_ratio))
            resized_w = round(math.sqrt(area / aspect_ratio))
            _image = _image.resize((resized_w, resized_h))
            return _image

        if not isinstance(image, (list, tuple)):
            image = [image]
        image_list = [ensure_image(_image) for _image in image]
        base64_images = images_to_base_64(images=image_list)

        action_rule,is_diving = self._get_action_rule(prompt)
        role_content = [
            {"type": "text", "text": prompt},
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                }
                for image_data in base64_images
            ],
        ]
        prompt = f"{prompt}"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": role_content,  # [{"type": "text", "text": prompt}]
            },
        ]
        # 这里仅取第一张图（如需多图可扩展）
        if is_diving:
            print("跳水 prompt优化=====>")
            messages = build_messages(
                system_prompt, prompt, image=base64_images[0], action_rule=action_rule
            )
        else:
            prompt_with_image = f"动作参考原 prompt: {prompt}。场景以及人物性别、发型、体态、姿态服装和外貌请以图像为准。"

            messages = build_messages(
                system_prompt,
                prompt_with_image,
                image=base64_images[0],
                action_rule=action_rule
            )


        exception = None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                seed=seed,
            )
            expanded_prompt = response.choices[0].message.content
            return PromptOutput(
                status=True,
                prompt=expanded_prompt,
                seed=seed,
                system_prompt=system_prompt,
                message=response.choices[0].message,
            ),is_diving
        except Exception as e:
            logger.warning(f"failed to expand prompt for {e}")
            exception = e

        return PromptOutput(
            status=False,
            prompt=prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception),
        ),is_diving

if __name__ == "__main__":
    seed = 100
    url = "http://10.127.16.79:8999/v1/"
    image = "/nvfile-heatstorage/zhoulx/shared_data/first_frame_img_for_t2v_wxr_0625/s0000091ymh_2-90_0_frame5.jpg"
    en_prompt = "The video captures a dynamic moment at a diving competition held in Paris 2024. The main subject, a diver dressed in a yellow swimsuit, executes a complex dive from a high platform. The diver's body is arched gracefully as they perform a mid-air flip before entering the water with minimal splash. The background features a crowded stadium filled with spectators, adding to the event's lively atmosphere. The camera follows the diver's movement smoothly, emphasizing the athleticism and precision of the dive. The bright lighting highlights the clarity of the water and the diver's form, creating a visually striking scene"
    model_name = "Qwen2.5-VL-7B-Instruct"  # 或者你要的模型名
    openai_prompt_expander = OpenAIPromptExpander(
        model_name=model_name, address=url, is_vl=True
    )
    openai_result,_ = openai_prompt_expander(en_prompt, tar_lang="zh", image=image)
    print(
        "LM  openai result -> zh", openai_result.prompt
    )  # dashscope_result.system_prompt)
