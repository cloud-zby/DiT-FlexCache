import json
import os
import sys
import cv2
import fire
import torch
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger
from sklearn.preprocessing import normalize
from torchvision import transforms
from insightface.app import FaceAnalysis


def relax_bbox(bbox, H, W, top_relax=0.0, bottom_relax=0.0, horizontal_relax=0.0, return_integer=False):
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    x1 = max(x1 - w * horizontal_relax, 0)
    x2 = min(x2 + w * horizontal_relax, W)
    y1 = max(y1 - h * top_relax, 0)
    y2 = min(y2 + h * bottom_relax, H)
    if return_integer:
        return [int(np.round(x)) for x in (x1, y1, x2, y2)]
    else:
        return [x1, y1, x2, y2]

def relax_bbox_square(bbox, H, W, force=False):
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    if force:
        if h > W:
            y1 = int(np.round((y1 + y2) / 2 - W / 2))
            y2 = y1 + W
        elif w > H:
            x1 = int(np.round((x1 + x2) / 2 - H / 2))
            x2 = x1 + H
        elif h > w:
            if x1 - (h - w) / 2 < 0:
                x1 = 0
                x2 = h
            elif x2 + (h - w) / 2 > W:
                x2 = W
                x1 = W - h
            else:
                x1 = max(x1 - (h - w) / 2, 0)
                x2 = min(x2 + (h - w) / 2, W)
        elif w > h:
            if y1 - (w - h) / 2 < 0:
                y1 = 0
                y2 = w
            elif y2 + (w - h) / 2 > H:
                y2 = H
                y1 = H - w
            else:
                y1 = max(y1 - (w - h) / 2, 0)
                y2 = min(y2 + (w - h) / 2, H)
    else:
        if h > w:
            x1 = max(x1 - (h - w) / 2, 0)
            x2 = min(x2 + (h - w) / 2, W)
        else:
            y1 = max(y1 - (w - h) / 2, 0)
            y2 = min(y2 + (w - h) / 2, H)
    return x1, y1, x2, y2



class FaceFeatureProcessor:

    def __init__(self, model_dir='~/.insightface') -> None:
        self.face_analysis = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'], root=model_dir)
        self.face_analysis.prepare(0, det_size=(640, 640), det_thresh=0.5)

    @staticmethod
    def _area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def process_ref_image(self, image_face):
        faces = self.face_analysis.get(image_face)
        faces = sorted(faces, key=lambda x: -self._area(x['bbox']))
        if len(faces) > 0:
            face = faces[0]
            face_feature = face['embedding']
            return face_feature
        else:
            print("warning: no face detected")
            return None

    def process_ref_image_list(self, image_face_list):
        face_features = []
        for image_face in image_face_list:
            image_face = cv2.imread(image_face)
            ff = self.process_ref_image(image_face)
            if ff is not None:
                face_features.append(ff)
        if len(face_features) == 0:
            return None
        else:
            face_features = normalize(np.stack(face_features, axis=0))
            return torch.tensor(face_features).unsqueeze(0).cuda() * 20
        
    def detect_image(self, image):
        return self.face_analysis.get(image)
    
    def get_topk_faces(self, image, topk=3):
        faces = self.face_analysis.get(image)
        faces = sorted(faces, key=lambda x: -self._area(x['bbox']))[:topk]
        if len(faces) > 0:
            return faces
        else:
            return None
        
    def match_faces(self, faces, ref_faces):
        """match ref_faces with faces which is topk faces returned, give the final index of order of ref_faces"""
        target_face_features = [face['embedding'] for face in faces]
        ref_face_features = []
        for face in ref_faces:
            image_face = cv2.imread(face)
            ff = self.process_ref_image(image_face)
            if ff is not None:
                ref_face_features.append(ff)
        if len(target_face_features) == 0 or len(ref_face_features) == 0:
            return []
        target_face_features = normalize(np.stack(target_face_features, axis=0))
        ref_face_features = normalize(np.stack(ref_face_features, axis=0))
        sim_matrix = target_face_features @ ref_face_features.T
        matched_indices = sim_matrix.argmax(axis=1)
        return matched_indices

    def generate_mask_from_detection(self, image, target_face, padding=0.4, crop_size=768):
        H, W = image.height, image.width
        bbox = relax_bbox_square(target_face['bbox'], H, W)
        bbox = relax_bbox(bbox, H, W, 0.02, 0.02, 0.02, return_integer=True)
        x1, y1, x2, y2 = relax_bbox(bbox, H, W, padding, padding, padding, return_integer=True)
        cropped_image = image.crop((x1, y1, x2, y2))
        mask = Image.new('L', (W, H), 0)
        ImageDraw.Draw(mask).rectangle(bbox, fill=255)
        mask = mask.crop((x1, y1, x2, y2))
        h, w = mask.height, mask.width
        r = crop_size / min(h, w)
        h = int((h * r) // 64 * 64)
        w = int((w * r) // 64 * 64)
        bbox = x1, y1, x2, y2
        return cropped_image, mask, bbox, (h, w)