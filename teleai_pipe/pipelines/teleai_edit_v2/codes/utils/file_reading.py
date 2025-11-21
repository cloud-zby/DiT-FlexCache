import pandas as pd
import json
import os
import cv2

def read_list_from_jsonl(path):
    data = pd.read_json(path, lines=True)
    data_list = data.to_dict(orient='records')
    return data_list

def read_pd_from_jsonl(path):
    data = pd.read_json(path, lines=True)
    return data

def read_list_from_json(path):
    with open(path, 'r') as f:
        data_list = json.load(f)
    return data_list

def read_list_from_txt(path):
    data = open(path).readlines()
    data = [i.strip() for i in data]
    return data

def read_list_from_csv(path):
    data = pd.read_csv(path).to_dict('records')
    return data

def save_list_to_txt(data, path):
    with open(path, 'w') as f:
        for i in data:
            f.write(i + '\n')
            
def save_caption_to_txt(data, path):
    with open(path, 'w') as f:
        f.write(data)
            
def save_list_to_jsonl(data, path):
    data_df = pd.DataFrame(data)
    data_df.to_json(path, orient='records', lines=True, force_ascii=False)
    
def save_list_to_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
        
def sorted_list(datas, rule):
    return sorted(datas, key=lambda x: eval(rule))


def extract_frame(video_path, frame_number, output_path="frame.jpg"):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # 设置要提取的帧号（从0开始）
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 读取帧
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 释放视频资源
    cap.release()
    
    if ret:
        # 保存帧为图片
        # cv2.imwrite(output_path, frame)
        print(f"Frame {frame_number} saved as {output_path}")
        return frame_rgb
    else:
        print("Error: Could not read frame.")
    

# 示例：提取第100帧
extract_frame("input.mp4", 100, "frame_100.jpg")

    
