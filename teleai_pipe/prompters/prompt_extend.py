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
    def __init__(self, model_name, is_vl=False, timeout=60):
        self.model_name = model_name
        self.is_vl = is_vl
        self.timeout = timeout

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
    ):
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
        elif image is None:
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
        timeout=60,
    ):
        """
        Args:
            api_key: The API key for Dash Scope authentication and access to related services.
            model_name: Model name, 'qwen-plus' for extending prompts, 'qwen-vl-max' for extending prompt-images.
            max_image_size: The maximum size of the image; unit unspecified (e.g., pixels, KB). Please specify the unit based on actual usage.
            retry_times: Number of retry attempts in case of request failure.
            is_vl: A flag indicating whether the task involves visual-language processing.
            timeout: timeout in seconds
        """
        super().__init__(model_name, is_vl, timeout=timeout)
        self.api_key = api_key
        self.max_image_size = max_image_size
        self.model = model_name
        self.retry_times = retry_times
        self.client = OpenAI(api_key=self.api_key, base_url=address)

    def extend(self, prompt, system_prompt, seed=-1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

        exception = None
        for _ in range(self.retry_times):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, seed=seed, timeout=self.timeout
                )
                expanded_prompt = response.choices[0].message.content
                return PromptOutput(
                    status=True,
                    prompt=expanded_prompt,
                    seed=seed,
                    system_prompt=system_prompt,
                    message=response.choices[0].message,
                )
            except Exception as e:
                logger.warning(f"failed to expand prompt for {e}")
                exception = e
        return PromptOutput(
            status=False,
            prompt=prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception),
        )

    def extend_with_img(
        self,
        prompt,
        system_prompt,
        image: Union[List[Image.Image], List[str], Image.Image, str] = None,
        seed=-1,
    ):
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
        response = None
        result_prompt = prompt
        exception = None
        status = False
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                seed=seed,
                timeout=self.timeout,
            )
            expanded_prompt = response.choices[0].message.content
            return PromptOutput(
                status=True,
                prompt=expanded_prompt,
                seed=seed,
                system_prompt=system_prompt,
                message=response.choices[0].message,
            )
        except Exception as e:
            logger.warning(f"failed to expand prompt for {e}")
            exception = e
        result_prompt = result_prompt.replace("\n", "\\n")

        return PromptOutput(
            status=status,
            prompt=result_prompt,
            seed=seed,
            system_prompt=system_prompt,
            message=str(exception) if not status else response,
        )


if __name__ == "__main__":
    seed = 100
    url = "http://10.127.16.79:8999/v1/"
    prompt = "夏日海滩度假风格，一只戴着墨镜的白色猫咪坐在冲浪板上。猫咪毛发蓬松，表情悠闲，直视镜头。背景是模糊的海滩景色，海水清澈，远处有绿色的山丘和蓝天白云。猫咪的姿态自然放松，仿佛在享受海风和阳光。近景特写，强调猫咪的细节和海滩的清新氛围。"
    en_prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
    # test openai api
    model_name = "Qwen2.5-VL-7B-Instruct"
    openai_prompt_expander = OpenAIPromptExpander(model_name=model_name, address=url)
    openai_result = openai_prompt_expander(prompt, tar_lang="zh")
    print(
        "LM  openai result -> zh", openai_result.prompt
    )  # dashscope_result.system_prompt)
    dashscope_result = openai_prompt_expander(prompt, tar_lang="en")
    print(
        "LM dashscope result -> en", dashscope_result.prompt
    )  # dashscope_result.system_prompt)
    dashscope_result = openai_prompt_expander(en_prompt, tar_lang="zh")
    print(
        "LM dashscope en result -> zh", dashscope_result.prompt
    )  # dashscope_result.system_prompt)
    image = "/nvfile-heatstorage/AIGC_H100/zuoxin/workspace/teleai_pipe/examples/data/101235-video-720_0.png"
    en_prompt = "A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets."
    openai_prompt_expander = OpenAIPromptExpander(
        model_name=model_name, address=url, is_vl=True
    )
    openai_result = openai_prompt_expander(
        en_prompt, tar_lang="zh", image=image, movie_style=True
    )
    print(
        "LM  openai result -> zh", openai_result.prompt
    )  # dashscope_result.system_prompt)
