import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import hashlib
import colorsys
from typing import Dict, Tuple, List
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict
import torch
import imageio


def image_compose_width(imag, imag_1):
    # read the size of image1
    rom_image = imag
    width, height = imag.size
    # read the size of image2
    rom_image_1 = imag_1
    
    width1 = rom_image_1.size[0]
    # create a new image
    to_image = Image.new('RGB', (width+width1, height))
    # paste old images
    to_image.paste(rom_image, (0, 0))
    to_image.paste(rom_image_1, (width, 0))
    return to_image

def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def create_4grid_video(videos, grid_size=(2, 2), frame_size=(1280, 720)):
    """
    将 1-4 个视频拼接成一个 4 宫格（2x2 布局），不足的部分用黑色图像填充。

    Args:
        videos (list): 包含视频帧列表的列表，例如 [video1_frames, video2_frames, ...]。
        grid_size (tuple): 宫格布局，默认为 (2, 2)。
        frame_size (tuple): 每帧的目标尺寸，默认为 (256, 256)。

    Returns:
        list: 拼接后的视频帧列表。
    """
    rows, cols = grid_size
    num_videos = len(videos)
    max_frames = max(len(video) for video in videos)  # 获取最大帧数

    # 创建黑色图像作为填充
    black_image = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

    # 处理每一帧
    combined_frames = []
    for frame_idx in range(max_frames):
        grid_frame = []  # 当前帧的宫格
        for video_idx in range(rows * cols):
            if video_idx < num_videos and frame_idx < len(videos[video_idx]):
                # 如果视频存在且当前帧存在，则使用该帧
                frame = videos[video_idx][frame_idx]
                if isinstance(
                    frame, Image.Image
                ):  # 如果是 PIL.Image.Image，转换为 NumPy 数组
                    frame = np.array(frame.resize(frame_size))
                else:  # 如果是 NumPy 数组，直接调整大小
                    frame = np.array(Image.fromarray(frame).resize(frame_size))
            else:
                # 否则使用黑色图像填充
                frame = black_image
            grid_frame.append(frame)

        # 将宫格中的图像拼接成完整帧
        row1 = np.hstack(grid_frame[:cols])  # 第一行
        row2 = np.hstack(grid_frame[cols:])  # 第二行
        combined_frame = np.vstack((row1, row2))  # 纵向拼接两行
        combined_frames.append(combined_frame)

    return combined_frames


def sample_video(video, indexes, method=2):
    if method == 1:
        frames = video.get_batch(indexes)
        frames = (
            frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        )
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        frames = video.get_batch(all_indexes)
        frames = (
            frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        )
        frames = frames[indexes]
    else:
        assert False
    return frames


class GeneratePoseControlImages:
    def __init__(self):
        self.if_valid = True

    def compute_iou(self, box1, box2):
        """
        计算两个检测框的 IOU
        box 格式为 (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)

        union_area = box1_area  #  + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def compute_bbox(self, points):
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)

        return [x_min, y_min, x_max, y_max]

    def draw_smpl_body(
        self, canvas, points, stickwidth=2, r=2, box=None, scores=None, score_thre=None
    ):
        # 过滤掉全0 或者 -1
        if box is not None:
            # 给box 扩 1/10 的padding
            x_pad, y_pad = (box[2] - box[0]) / 20.0, (box[3] - box[1]) / 20.0
            box[0], box[2] = box[0] - x_pad, box[2] + x_pad
            box[1], box[3] = box[1] - x_pad, box[3] + x_pad

        colors = [
            [255, 0, 0],  # 0
            [0, 255, 0],  # 1
            [0, 0, 255],  # 2
            [255, 0, 255],  # 3
            [255, 255, 0],  # 4
            [85, 255, 0],  # 5
            [0, 75, 255],  # 6
            [0, 255, 85],  # 7
            [0, 255, 170],  # 8
            [170, 0, 255],  # 9
            [85, 0, 255],  # 10
            [0, 85, 255],  # 11
            [0, 255, 255],  # 12
            [85, 0, 255],  # 13
            [170, 0, 255],  # 14
            [255, 0, 255],  # 15
            [255, 0, 170],  # 16
            [255, 0, 85],  # 17
        ]
        connetions = [
            [15, 12],
            [12, 16],
            [16, 18],
            [18, 20],
            [20, 22],
            [12, 17],
            [17, 19],
            [19, 21],
            [21, 23],
            [12, 9],
            [9, 6],
            [6, 3],
            [3, 0],
            [0, 1],
            [1, 4],
            [4, 7],
            [7, 10],
            [0, 2],
            [2, 5],
            [5, 8],
            [8, 11],
        ]
        connection_colors = [
            [255, 0, 0],  # 0
            [0, 255, 0],  # 1
            [0, 0, 255],  # 2
            [255, 255, 0],  # 3
            [255, 0, 255],  # 4
            [0, 255, 0],  # 5
            [0, 85, 255],  # 6
            [255, 175, 0],  # 7
            [0, 0, 255],  ## 8
            [255, 85, 0],  # 9
            [0, 255, 85],  # 10
            [255, 0, 255],  # 11
            [255, 0, 0],  # 12
            [0, 175, 255],  # 13
            [255, 255, 0],  # 14
            [0, 0, 255],  # 15
            [0, 255, 0],  # 16
        ]

        # draw point
        for i in range(len(points)):
            x, y = points[i][0:2]
            x, y = int(x), int(y)
            if i == 13 or i == 14:
                continue
            if box is not None and (
                x < box[0] or x > box[2] or y < box[1] or y > box[3]
            ):  # 在box外不绘制
                continue
            if score_thre is not None and scores is not None and scores[i] > score_thre:
                continue
            overlay = canvas.copy()
            cv2.circle(overlay, (x, y), r, colors[i % 17], thickness=-1)
            canvas = cv2.addWeighted(
                overlay, 0.5, canvas, 0.5, 0
            )  # 0.5为透明度系数，可调整

        # draw line
        for i in range(len(connetions)):
            point1_idx, point2_idx = connetions[i][0:2]
            point1 = points[point1_idx]
            point2 = points[point2_idx]
            Y = [point2[0], point1[0]]
            X = [point2[1], point1[1]]
            if box is not None and (
                Y[0] < box[0]
                or Y[0] > box[2]
                or Y[1] < box[0]
                or Y[1] > box[2]
                or X[0] < box[1]
                or X[0] > box[3]
                or X[1] < box[1]
                or X[1] > box[3]
            ):  # 在box外不绘制
                continue

            if (
                score_thre is not None
                and scores is not None
                and (scores[point1_idx] > score_thre or scores[point2_idx] > score_thre)
            ):
                continue

            mX = int(np.mean(X))
            mY = int(np.mean(Y))
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (mY, mX), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            overlay = canvas.copy()
            cv2.fillConvexPoly(overlay, polygon, connection_colors[i % 17])
            canvas = cv2.addWeighted(
                overlay, 0.5, canvas, 0.5, 0
            )  # 0.5为透明度系数，可调整

        return canvas

    def __call__(self, data_dict):
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        joints2ds = data_dict["joints2d_nonparam"]  # 有可能bbox无检出, 导致长度为0
        joints2ds_scores = data_dict["joint_uncertainties"]
        
        bbox = data_dict["bbox"]
        input_cn_images = []
        for idx in sample_indexes:
            filtered_bboxes = []
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            try:
                bboxDictofPerson = {
                    k: v for k, v in bbox[f"frame#{idx}"].items() if "person" in k
                }
                bboxDictofPerson = {
                    k: v for k, v in bboxDictofPerson.items() if v[0] != -1
                }
                sorted_bboxDictofPerson = sorted(
                    bboxDictofPerson.items(),
                    key=lambda item: (item[1][2] - item[1][0])
                    * (item[1][3] - item[1][1]),
                    reverse=True,
                )
                oriOrderofPerson, sortedOrderofPerson = (
                    [k for k in bboxDictofPerson.keys()],
                    [k[0] for k in sorted_bboxDictofPerson],
                )
                for sortedPersonBoxkey in sortedOrderofPerson:
                    ori_PersonBoxIdx = oriOrderofPerson.index(sortedPersonBoxkey)
                    joints2d = joints2ds[idx][ori_PersonBoxIdx]
                    joints2d_score = joints2ds_scores[idx][ori_PersonBoxIdx]
                    box = bboxDictofPerson[sortedPersonBoxkey]
                    if np.isnan(joints2d).any() or (joints2d == -1).all():
                        continue
                    # 对所有在框内，但分数阈值<80%都不合格的, 过滤整个框
                    inbox_mask = (
                        (joints2d[:, 0] >= box[0])
                        & (joints2d[:, 0] <= box[2])
                        & (joints2d[:, 1] >= box[1])
                        & (joints2d[:, 1] <= box[3])
                    )
                    if (
                        sum(inbox_mask) == 0
                        or joints2d_score[inbox_mask].mean() > 350
                        or np.sum(joints2d_score[inbox_mask] <= 350)
                        / max(sum(inbox_mask), 1)
                        < 0.8
                    ):
                        continue
                    if (box[3] - box[1]) < height / 5:
                        continue  # 高度不足H/5，直接过滤
                    if any(
                        self.compute_iou(box, kept_box) > 0.8
                        for kept_box in filtered_bboxes
                    ):
                        continue  # 存在 IOU 大于 0.8 的框，过滤掉该检测框
                    filtered_bboxes.append(box)  # 保留当前检测框
                    canvas = self.draw_smpl_body(
                        canvas,
                        joints2d,
                        stickwidth=int(height / 350),
                        box=box,
                        scores=joints2d_score,
                        score_thre=600,
                    )
                # Image.fromarray(canvas).save(f'results/{idx}_pose.jpg')
                if not filtered_bboxes:
                    self.if_valid = False
                input_cn_images.append(Image.fromarray(canvas))
            except Exception as e:
                input_cn_images.append(Image.fromarray(canvas))
        return input_cn_images
    
    def call_nobox_noscore(self, data_dict):
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        joints2ds = data_dict["joints2d_nonparam"]  # 有可能bbox无检出, 导致长度为0

        input_cn_images = []
        for idx in sample_indexes:
            canvas = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            try:
                for ori_PersonBoxIdx in range(len(joints2ds[idx])):
                    joints2d = joints2ds[idx][ori_PersonBoxIdx]
                    if np.isnan(joints2d).any() or (joints2d == -1).all():
                        continue
                    canvas = self.draw_smpl_body(
                        canvas,
                        joints2d,
                        stickwidth=int(height / 350),
                        box=None,
                        scores=None,
                        score_thre=None,
                    )
                # Image.fromarray(canvas).save(f'results/{idx}_pose.jpg')
                if np.sum(canvas) == 0:
                    self.if_valid = False
                input_cn_images.append(Image.fromarray(canvas))
            except Exception as e:
                input_cn_images.append(Image.fromarray(canvas))
        return input_cn_images


# 类型别名定义
BBox = Tuple[float, float, float, float]
BBoxDict = Dict[str, BBox]
# 框的类别定义
dino_class_dict = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}
motion_class_dict = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    # 9: "traffic light",
    # 10: "fire hydrant",
    # 11: "stop sign",
    # 12: "parking meter", #停车计时器
    # 13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    # 24: "backpack",
    # 25: "umbrella",
    # 26: "handbag",
    # 27: "tie",
    # 28: "suitcase",
    29: "frisbee",  # 飞盘
    30: "skis",  # 滑雪板，双板
    31: "snowboard",  # 滑雪板，单板
    32: "sports ball",
    33: "kite",
    34: "baseball bat",  # 棒球棒
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",  # 冲浪板
    38: "tennis racket",  # 网球拍
    # 39: "bottle",
    # 40: "wine glass",
    # 41: "cup",
    # 42: "fork",
    # 43: "knife",
    # 44: "spoon",
    # 45: "bowl",
    # 46: "banana",
    # 47: "apple",
    # 48: "sandwich",
    # 49: "orange",
    # 50: "broccoli", #花椰菜
    # 51: "carrot",
    # 52: "hot dog",
    # 53: "pizza",
    # 54: "donut",
    # 55: "cake",
    # 56: "chair",
    # 57: "couch",
    # 58: "potted plant",
    # 59: "bed",
    # 60: "dining table",
    # 61: "toilet",
    # 62: "tv",
    # 63: "laptop",
    # 64: "mouse",
    # 65: "remote", #遥控器
    # 66: "keyboard",
    # 67: "cell phone",
    # 68: "microwave",
    # 69: "oven",
    # 70: "toaster",
    # 71: "sink",
    # 72: "refrigerator",
    # 73: "book",
    # 74: "clock",
    # 75: "vase",
    # 76: "scissors",
    # 77: "teddy bear",
    # 78: "hair drier",
    # 79: "toothbrush",
}
dino_class_list = list(dino_class_dict.values())
motion_class_list = list(motion_class_dict.values())


class ColorSystem:
    """颜色管理系统"""

    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        self.class_hues = {}  # 类名到色相的映射
        self.cache = {}  # 颜色缓存

    def _get_class_hue(self, class_name: str) -> float:
        """为每个类别分配唯一的色相（均匀分布）"""
        if class_name not in self.class_hues:
            hash_int = int(
                hashlib.sha256(class_name.encode()).hexdigest(), 16
            )  # 使用哈希确定初始位置
            start = (hash_int % 1000) / 1000.0
            self.class_hues[class_name] = (
                start + (len(self.class_hues) / self.num_classes)  # 均匀分布色相
            )
        return self.class_hues[class_name] % 1.0

    def get_box_color(self, class_name: str) -> Tuple[int, int, int]:
        """获取类别的主颜色（固定饱和度/明度）"""
        if class_name not in self.cache:
            hue = self._get_class_hue(class_name)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.6)
            self.cache[class_name] = (int(r * 255), int(g * 255), int(b * 255), 128)
        return self.cache[class_name]

    def get_dot_color(self, class_name: str, track_id: str) -> Tuple[int, int, int]:
        """生成与框颜色对比的辅助颜色"""
        cache_key = f"{class_name}_.{track_id}"
        if cache_key not in self.cache:
            base_hue = self._get_class_hue(class_name)
            comp_hue = (base_hue + 0.5) % 1.0  # 生成互补色相
            hash_int = int(hashlib.sha256(track_id.encode()).hexdigest(), 16)
            value = 0.3 + (hash_int % 70) / 100  # 亮度范围0.3-1.0

            r, g, b = colorsys.hsv_to_rgb(comp_hue, 0.9, value)
            candidate = (int(r * 255), int(g * 255), int(b * 255), 128)

            box_color = self.get_box_color(class_name)
            max_attempts = 10
            attempts = 0
            best_candidate = candidate
            max_distance = self.color_distance(box_color, candidate)

            while (
                self.color_distance(box_color, candidate) < 128
                and attempts < max_attempts
            ):
                value = min(1.0, value + 0.2)
                r, g, b = colorsys.hsv_to_rgb(comp_hue, 0.9, value)
                new_candidate = (int(r * 255), int(g * 255), int(b * 255), 128)
                new_distance = self.color_distance(box_color, new_candidate)

                if new_distance > max_distance:
                    max_distance = new_distance
                    best_candidate = new_candidate

                candidate = new_candidate
                attempts += 1

            if self.color_distance(box_color, candidate) < 128:
                candidate = best_candidate

            self.cache[cache_key] = candidate
        return self.cache[cache_key]

    @staticmethod
    def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        """计算颜色差异(0-255范围)"""
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


class BBoxProcessor:
    """改进版目标框处理器，仅在高度重叠时过滤小框"""

    def __init__(
        self,
        height: int,
        width: int,
        max_boxes: int = 6,
        iou_threshold: float = 0.7,
        coverage_threshold: float = 0.8,
    ):
        self.height = height
        self.width = width
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.coverage_threshold = coverage_threshold

    def process_frame(self, boxes: BBoxDict) -> BBoxDict:
        pre_filtered_boxes = self.filter_bboxes(
            boxes,
            height=self.height,
            width=self.width,
            min_area_ratio=0.03,
            min_dim_ratio=0.15,
        )
        filtered = self._filter_overlaps(pre_filtered_boxes)
        return self._random_limit(filtered)

    def _filter_overlaps(self, boxes: Dict) -> Dict:
        class_groups = defaultdict(list)
        for name, box in boxes.items():
            cls = name.split("#")[0]
            class_groups[cls].append((name, box))

        result = {}
        for cls, group in class_groups.items():
            # 按面积降序排列（确保大框先处理）
            sorted_group = sorted(
                group, key=lambda x: self._box_area(x[1]), reverse=True
            )
            kept = []
            for name, box in sorted_group:
                # 仅当与已保留框高度重叠时才过滤
                if not self._is_high_overlap(box, kept):
                    kept.append((name, box))
            # 保留所有未被高度重叠的框
            result.update({name: box for name, box in kept})
        return result

    def _is_high_overlap(self, box: tuple, kept: list) -> bool:
        """判断是否与已保留框存在高度重叠"""
        for _, kept_box in kept:
            iou = self._iou(box, kept_box)
            # 条件1：IOU超过阈值
            if iou > self.iou_threshold:
                return True
            # 条件2：当前框被完全包含
            if self._is_contained(box, kept_box):
                return True
            # 条件3：当前框的覆盖率超过阈值（小框大部分在大框内）
            coverage = self._coverage(box, kept_box)
            if coverage > self.coverage_threshold:
                return True
        return False

    def _random_limit(self, boxes: Dict, class_list: list = motion_class_list) -> Dict:
        """随机筛选到最大数量"""
        if len(boxes) <= self.max_boxes:
            return boxes
        import random

        random.seed(42)
        keys = list(boxes.keys())
        # random.shuffle(keys)
        # return {k: boxes[k] for k in keys[:self.max_boxes]}
        # 先保留在class_list中的类别
        reserved_keys = [k for k in keys if k.split("#")[0].lower() in class_list]
        remaining_keys = [k for k in keys if k not in reserved_keys]
        random.shuffle(remaining_keys)
        num_remaining = self.max_boxes - len(reserved_keys)
        if num_remaining > 0:
            selected_remaining_keys = remaining_keys[:num_remaining]
        else:
            selected_remaining_keys = []
        selected_keys = reserved_keys + selected_remaining_keys
        return {k: boxes[k] for k in selected_keys}

    @staticmethod
    def _is_contained(inner_box: tuple, outer_box: tuple) -> bool:
        return (
            inner_box[0] >= outer_box[0]
            and inner_box[1] >= outer_box[1]
            and inner_box[2] <= outer_box[2]
            and inner_box[3] <= outer_box[3]
        )

    @staticmethod
    def _coverage(box_a: tuple, box_b: tuple) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        return inter_area / (area_a + 1e-6)

    @staticmethod
    def _iou(box_a: tuple, box_b: tuple) -> float:
        """计算交并比"""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _box_area(box: tuple) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    @staticmethod
    def filter_bboxes(
        bboxes: BBoxDict,
        height: int,
        width: int,
        exclude_classes: List[str] = ["person"],
        include_list: List[str] = dino_class_list,
        min_area_ratio: float = 0.04,
        min_dim_ratio: float = 0.2,
    ) -> BBoxDict:
        """
        过滤无效的边界框

        Args:
            bboxes: 原始边界框字典
            height: 图像高度
            width: 图像宽度
            exclude_classes: 需要排除的类别列表
            min_area_ratio: 最小区域比例（相对于整个画面）
            min_dim_ratio: 最小尺寸比例（相对于对应轴）

        Returns:
            过滤后的边界框字典
        """
        filtered = {}

        for cls_name, bbox in bboxes.items():
            base_cls = cls_name.split("#")[0].lower()
            if any(
                ex_cls in base_cls for ex_cls in exclude_classes
            ):  # 按照是否是子字符串来判断是否合理
                continue
            if include_list and base_cls not in include_list:
                continue
            if_valid = True  # 判断是否是validate_bbox
            x1, y1, x2, y2 = bbox
            if not (0 <= x1 < x2 < width) or not (0 <= y1 < y2 < height):
                if_valid = False
            area = (x2 - x1) * (y2 - y1)
            if area < (height * width * min_area_ratio):
                if_valid = False
            if (x2 - x1) < width * min_dim_ratio and (y2 - y1) < height * min_dim_ratio:
                if_valid = False

            if if_valid:
                filtered[cls_name] = bbox
        return filtered


class GenerateBoxControlImages:
    def __init__(self):
        self.if_valid = True

    def __call__(self, data_dict):
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        boxes_lst = data_dict["bbox"]
        color_system = ColorSystem()
        processor = BBoxProcessor(height=height, width=width)
        base_canvas = self.create_canvas(height, width)
        input_cn_images = []
        for idx in sample_indexes:
            key_name = f"frame#{idx}"
            frame_boxes = boxes_lst.get(key_name, {})
            if not frame_boxes or "empty" in frame_boxes:
                input_cn_images.append(base_canvas.copy())
            else:
                filtered = processor.process_frame(frame_boxes)
                canvas = base_canvas.copy()
                if filtered:
                    canvas = self.draw_enhanced_bboxes(
                        image=canvas, bboxes=filtered, color_system=color_system
                    )
                else:
                    self.if_valid = False
                input_cn_images.append(canvas)
        return input_cn_images

    @staticmethod
    def create_canvas(
        height: int, width: int, background: Tuple[int, int, int] = (0, 0, 0)
    ) -> Image.Image:
        """
        创建指定尺寸的画布
        """
        bg_array = np.array(background, dtype=np.uint8)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = bg_array
        return Image.fromarray(canvas, mode="RGB")

    @staticmethod
    def draw_enhanced_bboxes(
        image: Image.Image,
        bboxes: Dict[str, Tuple[float, float, float, float]],
        color_system: ColorSystem,
        line_width: int = 6,
        point_size: int = 20,
    ) -> Image.Image:
        """增强版绘制函数"""
        # draw = ImageDraw.Draw(image)
        width, height = image.size
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for full_name, bbox in bboxes.items():
            try:
                if "#" not in full_name:
                    continue
                class_name, track_id = full_name.split("#", 1)
                class_name = class_name.lower()

                box_color = color_system.get_box_color(class_name)
                dot_color = color_system.get_dot_color(class_name, track_id)

                draw.rectangle(bbox, outline=box_color, width=line_width)

                x_center = (bbox[0] + bbox[2]) / 2
                y_center = (bbox[1] + bbox[3]) / 2
                radius = point_size // 2

                draw.ellipse(
                    [
                        x_center - radius,
                        y_center - radius,
                        x_center + radius,
                        y_center + radius,
                    ],
                    fill=dot_color,
                )

            except Exception as e:
                print(f"Error drawing {full_name}: {str(e)}")

        result = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        return result


class GeneratePluckerControlImages:
    def __init__(self):
        self.if_valid = True
    
    def save_plucker_as_images(self, rays_world, cam_origin):
        H, W, _ = rays_world.shape
        rays_flat = rays_world.reshape(-1, 3)  # [H*W, 3]
        cam_origin = cam_origin.reshape(1, 3)
        cam_origin_expand = np.repeat(cam_origin, H * W, axis=0)
        moment = np.cross(cam_origin_expand, rays_flat).reshape(H, W, 3) 
        return moment

    def compute_ray_directions(self, intrinsic, H=480, W=640):
        """
        计算相机视角下每个像素点的方向向量（相对于相机原点）

        Args:
            intrinsic (np.ndarray): 3x3 相机内参矩阵
            H (int): 图像高度
            W (int): 图像宽度

        Returns:
            rays (np.ndarray): 方向向量 (H, W, 3)
        """
        # 生成像素网格 (H, W)
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        # 构造齐次坐标 (H, W, 3)，z=1
        pixels_homo = np.stack([u, v, np.ones_like(u)], axis=-1)  # (H, W, 3)

        # 计算方向向量 d = K^-1 * pixels_homo
        K_inv = np.linalg.inv(intrinsic)  # 计算内参逆矩阵
        rays = np.einsum('ij,hwj->hwi', K_inv, pixels_homo)  # 矩阵相乘

        # 归一化方向向量
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

        return rays


    def transform_rays_to_world(self, intrinsic, extrinsic, H=480, W=640):
        """
        计算世界坐标系下的光线方向（适用于 Camera-to-World 外参）

        Args:
            intrinsic (np.ndarray): 3x3 内参矩阵 K
            extrinsic (np.ndarray): 4x4 Camera-to-World 变换矩阵 [R | t]（齐次形式）
            H (int): 图像高度
            W (int): 图像宽度

        Returns:
            rays_world (np.ndarray): 世界坐标系下的光线方向 (H, W, 3)
            cam_origin (np.ndarray): 世界坐标系下的相机位置 (3,)
        """
        # 计算相机坐标系下的方向向量
        rays_cam = self.compute_ray_directions(intrinsic, H, W)  # (H, W, 3)

        # 提取旋转矩阵 R (前3x3) 和 平移 t (前3,4 的最后一列)
        R = extrinsic[:3, :3]  # 旋转矩阵
        t = extrinsic[:3, 3]   # 平移向量

        # 直接使用 t 作为相机原点
        cam_origin = t  # (3,)

        # 方向向量变换到世界坐标系
        rays_world = np.einsum('ij,hwj->hwi', R, rays_cam)  # (H, W, 3)

        return rays_world, cam_origin

    def __call__(self, data_dict):
        sample_indexes = data_dict["sample_indexes"]
        height = int(data_dict["video_height"])
        width = int(data_dict["video_width"])
        extrinsics = data_dict["camera_poses"]
        instrinsic = np.array([[900, 0, 360], [0, 900, 240], [0, 0, 1]]).astype(np.float32)

        ray_images, moments_norm_images = [], []
        for idx_sample_index, sample_index in enumerate(sample_indexes):
            if idx_sample_index == 0:
                extrinsic = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
            else:
                R1_inv = np.linalg.inv(extrinsics[sample_indexes[0]])
                extrinsic = np.dot(extrinsics[sample_index], R1_inv)

            ray_world, cam_origin = self.transform_rays_to_world(instrinsic, extrinsic, H=480, W=720)
            rays_norm = (((ray_world + 1) / 2.0) * 255).astype(np.uint8)
            moments = self.save_plucker_as_images(ray_world, cam_origin)
            moments_norm = (((moments + 1) / 2.0) * 255).astype(np.uint8)

            ray_images.append(Image.fromarray(rays_norm))
            moments_norm_images.append(Image.fromarray(moments_norm))

        return ray_images, moments_norm_images