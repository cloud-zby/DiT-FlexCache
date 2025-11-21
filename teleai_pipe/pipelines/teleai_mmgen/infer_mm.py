import json
import re
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import partial

from diffusers.utils import export_to_video
from PIL import Image
from torchvision.transforms import InterpolationMode, functional as F
from tqdm import tqdm
# from vast.pipelines import MMHunyuanVideoPipeline
from teleai_pipe.pipelines.teleai_mmgen.pipeline_diffusers_mm import MMHunyuanVideoPipeline
import cv2
import os
# import vast.utils.utils_io as IOUtils
from copy import copy

# Constants
MODEL_TYPE_MAPPING = {
    16: "t2v",
    32: "i2v",
    33: "i2vhy",

    49: "i2vhy_cond_dn" # rgb,depth, normal
}
DEFAULT_CONFIG = {
    # "dst_size": (1280, 720),
    "dst_size": (640, 352),
    "num_frames": 49,
    # "dst_size": (640, 480),
    "guidance_scale": 6.0,

    "embedded_guidance_scale": 1.0,
    "save_fps": 15,
    "inference_steps": 50,
    "base_model": "/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b"
}


class InferenceConfig:
    """集中管理推理配置参数"""
    def __init__(
        self,
        prompt: str,
        ref_images: Dict[int, str],
        dst_size: Tuple[int, int] = DEFAULT_CONFIG["dst_size"],
        guidance_scale: float = DEFAULT_CONFIG["guidance_scale"],
        embedded_guidance_scale: float = DEFAULT_CONFIG["embedded_guidance_scale"],
        save_fps: int = DEFAULT_CONFIG["save_fps"],
        num_frames: int = DEFAULT_CONFIG["num_frames"],
        inference_steps: int = DEFAULT_CONFIG["inference_steps"],
        base_model: str = DEFAULT_CONFIG["base_model"],
        dict_mm_info: Optional[Dict] = {}
    ):
        self.prompt = prompt
        self.ref_images = ref_images
        self.dst_size = dst_size
        self.guidance_scale = guidance_scale
        self.embedded_guidance_scale = embedded_guidance_scale
        self.save_fps = save_fps
        self.num_frames = num_frames
        self.inference_steps = inference_steps
        self.base_model = base_model
        self.dict_mm_info = dict_mm_info


def get_parent_path(path: str, level: int = 3) -> Path:
    """获取指定层级的父目录"""
    return Path(path).resolve().parents[level-1]


def get_checkpoints(model_dir: str, interval: int = 1) -> List[Tuple[str, str]]:
    """扫描目录获取模型检查点"""
    checkpoints = []
    pattern = re.compile(r"checkpoint_epoch_(\d+)_step_(\d+)")
    models_dir = Path(model_dir) / "models"
    
    for dir_path in models_dir.glob("**/checkpoint_*"):
        if match := pattern.search(dir_path.name):
            epoch, step = match.groups()
            if int(step) % interval == 0:
                name = f"{Path(model_dir).name}_ep{epoch}iter{step}"
                trans_path = dir_path / "transformer"
                if trans_path.exists():
                    checkpoints.append((name, str(trans_path)))
    return checkpoints


def generate_video_filename(result_name: str, model_name: str, config: InferenceConfig) -> str:
    """生成视频文件名"""
    width, height = config.dst_size
    base_name = f"{result_name}_{model_name}_{height}p_f{config.num_frames}_s{config.inference_steps}_fps{config.save_fps}_g{config.guidance_scale}_eg{config.embedded_guidance_scale}"
    return f"{base_name}.mp4"


def prepare_reference_images(config: InferenceConfig) -> List[Image.Image]:
    """准备参考图像"""
    width, height = config.dst_size
    ref_images = [Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(config.num_frames)]
    
    for frame_idx, img_path in config.ref_images.items():
        if frame_idx >= config.num_frames:
            raise ValueError(f"参考帧索引 {frame_idx} 超过总帧数 {config.num_frames}")
        
        original_img = Image.open(img_path)
        resized_img = F.resize(original_img, (height, width), InterpolationMode.BILINEAR)
        ref_images[frame_idx] = resized_img
    
    return ref_images


def prepare_reference_video(config: InferenceConfig, video_task) -> List[Image.Image]:
    width, height = config.dst_size
    cap = cv2.VideoCapture(config.dict_mm_info[video_task])
    ref_video = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        resized_img = F.resize(pil_image, (height, width), InterpolationMode.BILINEAR)
        ref_video.append(resized_img)
    
    if video_task == 'normal':
        ref_video.append(ref_video[-1])
    # print(f'video_task: {video_task}. raw video length: {len(ref_video)}')
    assert len(ref_video) >= config.num_frames, f'video length: {len(ref_video)} is less than num_frames: {config.num_frames}'
    return ref_video[:config.num_frames]


def prepare_reference_videos(config):
    dict_ref_videos = {}
    lis_task_keys = config.dict_mm_info.keys()
    for video_task in lis_task_keys:
        if video_task in config.dict_mm_info:
            # TODO: add fps sampling to align frames
            # print(f"video task: {video_task}")
            # print('add fps sampling to align frames *******\n\n')
            dict_ref_videos[video_task] = prepare_reference_video(config, video_task)
        
        if video_task == 'rgb':
            width, height = config.dst_size
            ref_images = [Image.new("RGB", (width, height), (0, 0, 0)) for _ in range(config.num_frames)]

            ref_images[0] = copy(dict_ref_videos['rgb'][0])
    
    return ref_images, dict_ref_videos


def load_pipeline(model_path: str, config: InferenceConfig, device: torch.device) -> MMHunyuanVideoPipeline:
    """加载模型管道"""
    config_path = get_parent_path(model_path) / "config.json"
    if not os.path.exists(config_path):
        config_path = f'{model_path}/config.json'

        with open(config_path) as f:
            in_channels = json.load(f)["in_channels"]
    
    model_type = MODEL_TYPE_MAPPING.get(in_channels)
    if not model_type:
        raise ValueError(f"不支持的 in_channels: {in_channels}")
    
    pipe = MMHunyuanVideoPipeline.from_pretrained(
        config.base_model,
        transformer_model_path=model_path,
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)
    # print(f'Not send device to GPU ****************************\n\n')
    pipe.vae.enable_tiling()
    return pipe


def inference_worker(
    rank: int,
    world_size: int,
    model_list: List[Tuple[str, str]],
    prompt_configs,
    gpu_ids: List[int],
    # dir_result: str
    dict_args={}
):
    """推理工作进程"""
    device_id = gpu_ids[rank]
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # save_dir = Path(__file__).parent / "results" / result_name
    
    dir_result_root = dict_args['dir_result_root']
    n_skip = dict_args['n_skip'] if 'n_skip' in dict_args else -1

    # 分配模型处理任务
    models_to_process = model_list[rank::world_size]
    if not models_to_process:
        print(f"[Rank {rank}] No models to process")
        return
    
    # ref_images = prepare_reference_images(config)
    from tqdm import tqdm
    for model_name, model_path in models_to_process:
        # try:
        # if True:
        config0 = InferenceConfig(
                prompt="",
                ref_images={},
                dict_mm_info={},
                **{k: v for k, v in DEFAULT_CONFIG.items() if k != "base_model"}
            )
        pipe = load_pipeline(model_path, config0, device)

        for test_name, prompt_config in tqdm(prompt_configs.items(), desc='Process prompts:'):
            print(f"\n{'='*40}\n开始处理测试用例: {test_name}\n{'='*40}")
            # result_name = Path(config["ref_images"][0]).stem
            prompt_short = prompt_config['prompt'][:25]

            cam_type = prompt_config['dict_mm_info']['rgb'].split('/')[-1].split('-')[0]

            # clip_id = prompt_config['clip_id']
            # result_name = f'{cam_type}-{prompt_short}_clip_{int(clip_id):03d}'
            result_name = f'{cam_type}-{prompt_short}'
            dir_result_curr = os.path.join(dir_result_root, result_name)
            # if not os.path.exists(dir_result_curr): 
            #     assert False
            os.makedirs(dir_result_curr, exist_ok=True)
            print(f'dir_result_curr: {dir_result_curr}. result_name: {result_name}')
            # exit()
            # continue
            save_dir = Path(dir_result_curr)
            save_dir.mkdir(parents=True, exist_ok=True)

            # 构建配置对象
            config = InferenceConfig(
                prompt=prompt_config["prompt"],
                ref_images=prompt_config["ref_images"],
                dict_mm_info=prompt_config["dict_mm_info"],
                **{k: v for k, v in DEFAULT_CONFIG.items() if k != "base_model"}
            )

            dict_conf_args = {
                'n_skip': n_skip # 7
            }

            ref_images, ref_mm_videos = prepare_reference_videos(config)

            save_name = generate_video_filename('', model_name, config)
            save_path = save_dir / save_name
            
            if save_path.exists():
                print(f"[Rank {rank}] 跳过已存在文件: {save_path}")
                continue
                
            
            
            output = pipe(
                prompt=config.prompt,
                height=config.dst_size[1],
                width=config.dst_size[0],
                num_frames=config.num_frames,
                num_inference_steps=config.inference_steps,
                seed=42,
                model_type=MODEL_TYPE_MAPPING[pipe.transformer.config.in_channels],
                ref_images=ref_images,
                ref_mm_videos = ref_mm_videos,
                guidance_scale=config.guidance_scale,
                embedded_guidance_scale=config.embedded_guidance_scale,
                padding=False,
                dict_conf=dict_args
            ).frames[0]
            
            export_to_video(output, str(save_path), fps=config.save_fps)
            print(f"[Rank {rank}] 保存成功: {save_path}")
            
            # 显存优化
            # del pipe
            torch.cuda.empty_cache()
        del pipe
        torch.cuda.empty_cache()
            
        # except Exception as e:
        #     print(f"[Rank {rank}] 处理模型 {model_name} 时出错: {str(e)}")


def run_inference_pipeline(
    checkpoint_dirs: List[str],
    prompt_configs: Dict[str, Dict[str, str]],
    gpu_ids: List[int] = None,
    checkpoint_interval: int = 100,
    all_models = [],
    dict_args = {}
):
    """运行完整推理流程"""
    # GPU配置
    available_gpus = torch.cuda.device_count()
    gpu_ids = gpu_ids or list(range(available_gpus))
    world_size = len(gpu_ids)
    
    # 加载所有检查点

    if len(all_models)==0:
        all_models = []
        for dir_path in checkpoint_dirs:
            all_models.extend(get_checkpoints(dir_path, checkpoint_interval))
        
    # all_models = [all_models[-1]]
    all_models = all_models[-8:]
    print(f'process model: {all_models}')

    if not all_models:
        raise ValueError("未找到任何有效检查点")
    
    # dir_result_root = './test/results'
    print(f'Number of items: {len(prompt_configs)}')
    
    # 处理所有测试用例
    inference_worker(rank=0, 
                        world_size=world_size, 
                        model_list=all_models, 
                        prompt_configs=prompt_configs, 
                        gpu_ids=gpu_ids,
                        dict_args=dict_args)

    # continue
        # # inference all temporay models
        # mp.spawn(
        #     partial(inference_worker,
        #            world_size=world_size,
        #            model_list=all_models,
        #            config=inference_config,
        #            gpu_ids=gpu_ids,
        #            dir_result=dir_result_curr),
        #     nprocs=world_size,
        #     join=True
        # )

def prepare_prompt_config_i2v():
    PROMPT_CONFIGS = {
        # "oven":{
        #     "prompt": """A woman is crouching in front of the oven in the kitchen, holding the oven door handle with both hands and opening the oven door.""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/oven.jpg"},
        # },
        # "kids2": {
        #     "prompt":  """Two young boys, one with blonde hair in a yellow shirt and blue jeans, the other with light brown hair in a red shirt and dark shorts, are seated on a grassy field, sipping water from clear plastic bottles. They are surrounded by the tranquility of nature, with a yellow bicycle and its black frame lying on its side nearby, suggesting a recent ride. The boys appear relaxed and engaged in conversation, enjoying a peaceful moment of rest in the late afternoon sun.""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/kids2.jpg"},
        # },
        # "talk": {
        #     "prompt":  """A woman with blonde hair styled in a high bun, wearing a white sweater and black pants, is seated on a wooden chair in front of a microphone on a stage. She is engaged in a conversation with a man in a black suit, who is seated opposite her. The stage is illuminated by warm, ambient lighting, and the background features a cityscape with tall buildings and colorful lights. The woman appears to be speaking and gesturing animatedly, while the man listens and occasionally responds. The main subjects are the woman and the man. The woman has blonde hair styled in a high bun, wears a white sweater, and black pants. She is seated on a wooden chair and is positioned in front of a microphone. The man, dressed in a black suit, is seated opposite her, facing her. He is holding a notepad and pen, and he occasionally gestures with his hands while speaking. The woman is actively speaking and gesturing, moving her hands and head as she talks. The man occasionally responds, nodding and gesturing with his hands. The background remains static, with no visible movement. The camera is stationary, capturing a medium shot of the two subjects from a frontal view.""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/talk.jpg"},
        # },
        # "work": {
        #     "prompt":  """Three professionals, two men in suits and a woman in a black blazer, are engaged in a serious discussion at an outdoor construction site. They are focused on a clipboard and documents, suggesting a meeting about a construction project. The setting includes towering cranes and an overcast sky, indicating a professional atmosphere. Throughout the video, the woman points to specific details on the clipboard, actively participating in the conversation. The men, one in a dark suit and the other in a navy blue suit, listen attentively, reflecting a collaborative and professional exchange of ideas.""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/work.jpg"},
        # },
        "kids": {
            "prompt":  """Two young boys, one with blonde hair in a yellow shirt and blue jeans, the other with light brown hair in a red shirt and dark shorts, are seated on a grassy field, sipping water from clear plastic bottles. They are surrounded by the tranquility of nature, with a yellow bicycle and its black frame lying on its side nearby, suggesting a recent ride. The boys appear relaxed and engaged in conversation, enjoying a peaceful moment of rest in the late afternoon sun.""",
            "ref_images": {0: "/nvfile-heatstorage/wjp/code/vast/test/assets/kids.jpg"},
        },
        # "naobaijin_cn": {
        #     "prompt":  """有两个卡通人物形象在原地跳舞。左边的是一位戴着帽子、留着小胡子的男性，身着带有花纹装饰的传统服饰；右边的是一位头戴头巾、佩戴着耳环等饰品的女性，穿着具有民族风格的服装，两人正在跳舞舞蹈姿势，男性跳着跳着摔倒了""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/naobaijin.jpg"},
        # },
        # "hula": {
        #     "prompt":  """the woman is hula hooping outdoors.""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/hula.jpg"},
        # },
        # "hamburger":{
        #     "prompt":  """A woman is dining in a fast-food restaurant. She is wearing transparent disposable gloves and holding a hamburger with both hands, bringing it to her mouth. There are other foods placed on the dining table in front of her. In the background, the ordering screens of the fast - food restaurant can be seen. The whole scene presents a relaxed dining atmosphere.。""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/hamburger.jpg"},
        # },
        # "hamburger_cn":{
        #     "prompt":  """一位女性正在快餐店用餐的场景。她戴着透明的一次性手套，双手拿着汉堡往嘴里送，面前的餐桌上还摆放着其他食物，背景中能看到快餐店的点餐屏幕。整个画面呈现出一种轻松的用餐氛围。""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/hamburger.jpg"},
        # },

        # "waterfall":{
        #     "prompt": """This is a natural scenery, showcasing a magnificent waterfall landscape. The water of the waterfall cascades down from a height, creating white splashes. Surrounding it are dense forests and continuous mountains, with a clear - blue sky, presenting a beautiful scene.""",
        #     "ref_images": {0: "/nvfile-heatstorage/ljq/repos/vast/test/assets/waterfall.jpg"},
        # },

        # the water from the waterfall hits the rocks, creating splashes of water.
    }
    return PROMPT_CONFIGS

def prepare_prompt_config_i2v_mm(path_meta, dir_videos, n_process = 20):
    # path_meta = '/nvfile-heatstorage/wjp/code/vast/test/assets/videos/meta.json'
    # path_meta = '/nvfile-heatstorage/wjp/data/video/vast/koala/meta_rdns/1-20.json'
    # path_meta = '/nvfile-heatstorage/wjp/code/teleai_data_tool/test/dataset_cam_test_v2.json'
    lis_clips = IOUtils.read_json_to_dict(path_meta)['clips']
    # convert clip to prompt format
    dict_config_infer = {}

    count_clip = 0
    for clip in lis_clips:
        clip_id = clip['id']

        count_clip += 1
        # if count_clip < 3:
        #     continue

        dict_mm_info = clip['meta']['mm_info']
        # copy video to current temp dir
        dir_cp = f'{dir_videos}/clip_{clip_id:03d}'
        os.makedirs(dir_cp, exist_ok=True)

        for video_type in dict_mm_info:
            path_src = dict_mm_info[video_type]
            path_tgt = f'{dir_cp}/{video_type}.mp4'
            os.system(f'cp {path_src} {path_tgt}')

        prompt = clip['caption']['dense_caption']

        dict_curr = {
            'prompt': prompt,
            'dict_mm_info': dict_mm_info,
            "ref_images": dict_mm_info['rgb']
        }
        dict_config_infer[clip_id] = dict_curr

        if count_clip >= n_process:
            break

    return dict_config_infer


def prepare_prompt_config_i2v_ours():

    path_vid = '/nvfile-heatstorage/wjp/code/vast/test/save_vid.mp4'
    path_vid_center = '/nvfile-heatstorage/wjp/code/vast/test/save_vid_center.mp4'
    prompt = "A couple is dancing in a luxurious ballroom filled with elegantly dressed guests. The man, dressed in a gray suit, and the woman, wearing a black dress, are dancing closely together, holding each other's hands and moving gracefully. The atmosphere is sophisticated, with a backdrop of elegant chandeliers, ornate decorations, and a crowd of well-dressed people enjoying the event. The couple's movements are smooth and rhythmic, with the woman's hair flowing as they twirl and sway. The background is a luxurious ballroom with ornate decorations, including chandeliers, potted plants, and elegant furniture. The room is filled with well-dressed guests, some of whom are seated at tables, while others are standing and mingling. The setting suggests a formal event, possibly a gala or a high-end party, with a focus on elegance and sophistication. The main subjects are a man and a woman dancing together. The man has dark hair and a mustache, wearing a gray suit, and the woman has her hair styled in an updo, wearing a black dress. They are holding hands and moving in close proximity, with the woman's back facing the camera at times. Their movements are fluid and synchronized, indicating a comfortable and intimate connection."

    prompt = "The video features a residential bathroom with a wooden cabinet and a white door, leading to a bathroom with a pedestal sink, toilet, and a window with a green towel. A framed anatomical poster is visible on the wall. As the video continues, the bathroom is shown with a white toilet, a green towel, and a blue bottle of soap, with a window providing natural light. The scene includes a wooden cabinet with brass handles and a poster of a giraffe. Finally, the bathroom is seen with a white toilet, a green towel, and a blue bottle of hand soap, with a guitar leaning against the wall, suggesting a musical interest."
    path_vid_center = '/nvfile-heatstorage/wjp/code/vast/test/data/vid_rot/t_f_90-The video features a resi-06709_merge_[2, 3]_seed=999999_center.mp4'


    dict_one_item = {
        'prompt': prompt,
        "ref_images": {},
        "dict_mm_info": {
            'rgb': path_vid
        }
    }

    dict_one_item_center = {
        'prompt': prompt,
        "ref_images": {},
        "dict_mm_info": {
            'rgb': path_vid_center
        }
    }

    dict_config_infer = { 
        # 0: dict_one_item, 
                            1: dict_one_item_center 
                            }
    return dict_config_infer


def read_dict2json(path_json):
    with open(path_json, 'r') as f:
        data = json.load(f)
        return data
    
def prepare_prompt_config_i2v_ours_test():
    path_json = "/nvfile-heatstorage/wjp/code/vast/test/data/vid_rot/config.json"
    dict_data = read_dict2json(path_json)


    # split dict to 4 chunks
    lis_dict = [{}, {}, {}, {}]
    n_item = len(dict_data)
    n_chunk = int(n_item / 4)
    
    count = 0
    for item in dict_data.keys():


        # if int(item) % 2 == 0:
        #     continue
        
        if 'center' not in dict_data[item]['dict_mm_info']['rgb']:
            continue
       

        index_c = count % 3

        dict_data[item]['clip_id'] = count
        lis_dict[index_c][int(item)] = dict_data[item]

        dict_mm_info = dict_data[item]['dict_mm_info']
        prompt_short = dict_data[item]['prompt'][:25]
        cam_type = dict_mm_info['rgb'].split('/')[-1].split('-')[0]
        # copy video to current temp dir
        dir_cp = f'{dir_videos}/{cam_type}-{prompt_short}'
        # dir_cp = f'{dir_videos}/{cam_type}-{prompt_short}_clip_{int(count):03d}'
        os.makedirs(dir_cp, exist_ok=True)

        for video_type in dict_mm_info:
            path_src = dict_mm_info[video_type]
            path_name = os.path.basename(path_src)
            path_tgt = f'{dir_cp}/{video_type}_{path_name}'
            os.system(f'cp "{path_src}" "{path_tgt}"')

        count += 1


    # for i in range(len(dict_data)):
    #     print('index: ', i, 'path: ',  dict_data[str(i)]['dict_mm_info']['rgb'].split('/')[-1])

    return lis_dict


def get_hunyuan_pipeline(device=torch.device('cuda')):
    model_path = "/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/transformer"
    # model_path = "/home/hunyuanvideo_13b/transformer"
    print('Load model from local home. put it at model zoo')
    config0 = InferenceConfig(
            prompt="",
            ref_images={},
            dict_mm_info={},
            **{k: v for k, v in DEFAULT_CONFIG.items() if k != "base_model"}
        )
    pipe = load_pipeline(model_path, config0, device)
    return pipe

def process_one_prompt(pipe, prompt, ref_vidsave_path, path_vid_save):
    # 构建配置对象
    config = InferenceConfig(
        prompt=prompt,
        ref_images=None,
        dict_mm_info={"rgb": ref_vidsave_path},
        **{k: v for k, v in DEFAULT_CONFIG.items() if k != "base_model"}
    )

    dict_conf_args = {
        'n_skip': 10 # 7
    }
    ref_images, ref_mm_videos = prepare_reference_videos(config)
    
    output = pipe(
        prompt=config.prompt,
        height=config.dst_size[1],
        width=config.dst_size[0],
        num_frames=config.num_frames,
        num_inference_steps=config.inference_steps,
        seed=42,
        model_type=MODEL_TYPE_MAPPING[pipe.transformer.config.in_channels],
        ref_images=ref_images,
        ref_mm_videos = ref_mm_videos,
        guidance_scale=config.guidance_scale,
        embedded_guidance_scale=config.embedded_guidance_scale,
        padding=False,
        dict_conf=dict_conf_args
    ).frames[0]
    
    export_to_video(output, path_vid_save, fps=config.save_fps)
    
    # 显存优化
    # del pipe
    torch.cuda.empty_cache()
    return 

if __name__ == "__main__":
    # 配置示例
    # depth normal finetune
    # dir_exp = '/nvfile-heatstorage/wjp/exps/vast/20250513_mm_rdn_noMMOut_v2_1000continue'
    # dir_videos = '/nvfile-heatstorage/wjp/code/vast/test/exps/depthnormal_1000continue'

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    index_chunk = args.index
    print(f'index chunk: {index_chunk}')



    exp_name = '20250521_hunyuan_orig_v3_rgbcond'
    path_meta = '/nvfile-heatstorage/wjp/data/video/vast/koala/meta_rdns/1-20.json'
    
    # camera depth gen
    # dir_exp = '/nvfile-heatstorage/wjp/exps/vast/20250514_mm_rCam_noMMOut_continue1400'
    # dir_videos = '/nvfile-heatstorage/wjp/code/vast/test/exps/camgen_1400continue'
    
    # exp_name = '20250519_mmcam_640_360_f49_depthcond'
    # path_meta = '/nvfile-heatstorage/wjp/code/teleai_data_tool/test/dataset_cam_test_v2.json'

    # # camera depth gen reso640480, f49
    # exp_name = '20250514_mm_rCam_noMMOut_reso640480_f49'
    n_skip = 10

    dir_exp = f'/nvfile-heatstorage/wjp/exps/vast/{exp_name}_skip1'
    dir_videos = f'/nvfile-heatstorage/wjp/code/vast/test/exps/{exp_name}_skip{n_skip}_ours_720p_injection_more-test'
    
    dir_videos = '/nvfile-heatstorage/wjp/code/teleai_pipe/temp/temp1'
    all_models = [
        (exp_name, '/home/hunyuanvideo_13b/transformer')
    ]

    n_process = 30

    CHECKPOINT_DIRS = [
        # "/nvfile-heatstorage/ljq/repos/distribute_vast/work_dirs/i2vhy_480p_81_dp320",
        # "/home/exps/test/mm/models",
        # '/nvfile-heatstorage/wjp/exps/vast/20250513_mm_rdn_noMMOut'
        # '/nvfile-heatstorage/wjp/exps/vast/20250514_mm_rCam_noMMOut'
        dir_exp
    ]
    checkpoint_interval = 200

    dir_result_root = dir_videos
    # PROMPT_CONFIGS = prepare_prompt_config_i2v()

    print(f'exp name: {exp_name}')
    print(f'evaluation resolution: {DEFAULT_CONFIG["dst_size"]}')
    print(f'evaluation frames: {DEFAULT_CONFIG["num_frames"]}')
    # input('Enter any key to contine ... ')


    # PROMPT_CONFIGS = prepare_prompt_config_i2v_mm(path_meta, dir_videos, n_process)
    PROMPT_CONFIGS = prepare_prompt_config_i2v_ours()
    # LIS_PROMPT_CONFIGS = prepare_prompt_config_i2v_ours_test()

    # PROMPT_CONFIGS = LIS_PROMPT_CONFIGS[index_chunk]

    GPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  
    # GPU_IDS = [0]  

    dict_args = {
        'dir_result_root': dir_result_root,
        'n_skip': n_skip,
        # 'alpha_injection': 0.0
    }

    run_inference_pipeline(
        checkpoint_dirs=CHECKPOINT_DIRS,
        prompt_configs=PROMPT_CONFIGS,
        gpu_ids=GPU_IDS,
        checkpoint_interval=checkpoint_interval,
        all_models=all_models,
        dict_args=dict_args
    )
    exit()
    try:
        run_inference_pipeline(
            checkpoint_dirs=CHECKPOINT_DIRS,
            prompt_configs=PROMPT_CONFIGS,
            gpu_ids=GPU_IDS,
            checkpoint_interval=checkpoint_interval,
            all_models=all_models,
            dict_args=dict_args
        )
    except Exception as e:
        print(f"主程序运行出错: {str(e)}")
        raise