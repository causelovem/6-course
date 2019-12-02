import glob
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import torch
from PIL import Image
import cv2

def get_dicom_fps(data_dir):
    dicom_fps = glob.glob(os.path.join(data_dir, "*.dcm"))
    return list(set(dicom_fps))


def parse_dataset(data_dir, anns):
    image_fps = get_dicom_fps(data_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for _, row in anns.iterrows():
        fp = os.path.join(data_dir, row["patientId"] + ".dcm")
        image_annotations.get(fp, []).append(row)
    return image_fps, image_annotations


# def parse_dataset(data_dir, anns):
#     # image_fps = get_dicom_fps(data_dir)
#     # image_annotations = {fp: [] for fp in image_fps}
#     image_fps = []
#     image_annotations = {}
#     i = 1
#     for index, row in anns.iterrows():
#         fp = os.path.join(data_dir, row["patientId"] + ".dcm")
#         image_fps.append(fp)
#         # image_annotations.get(fp, []).append(row)
#         image_annotations[fp] = image_annotations.get(fp, []) + [row]
#         if i == 1000:
#           break
#         i += 1
#     return image_fps, image_annotations


def convertAnnotations(oldAnnotation, image):
    b = oldAnnotation['boxes'].tolist()
    return {'image': np.array(image.permute(1, 2, 0)), 'category_id': oldAnnotation['labels'].tolist(), 
        'bboxes': [] if oldAnnotation['labels'][0] == 0 else [l[0:2] + [l[2] - l[0], l[3] - l[1]] for l in b]} 


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=(255, 0, 0), thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,(255, 255, 255), lineType=cv2.LINE_AA)
    return img


def visualize(annotations):
    category_id_to_name = {1: 'Issue'}
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)


class RSNADataset(Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width, train=True):
        self.image_fps = image_fps
        self.image_annotations = image_annotations
        self.orig_height = orig_height
        self.orig_width = orig_width

        image_info = dict()
        for image_idx, file_path in enumerate(image_fps):
            annotations = image_annotations[file_path]
            image_info[image_idx] = {"path": file_path,
                                     "annotations": annotations}
        self.image_info = image_info


    def __len__(self):
        return len(self.image_fps)


    def show_image(self, image_id):
        info = self.image_info[image_id]
        fp = info["path"]
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image


    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info["path"]
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        image = np.array(Image.fromarray(image).resize((self.orig_width, self.orig_height)))

        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)

        image = np.rollaxis(image, 2, 0) / 255
        return image


    def load_bbox(self, image_id, scale_factor):
        info = self.image_info[image_id]
        annotations = info["annotations"]
        count = len(annotations)
        if count == 0 or all((ann["Target"] == 0 for ann in annotations)):
            # Пневмонии нет, считаем за объект все фото
            xmin = 0
            xmax = 1024 * scale_factor
            ymin = 0
            ymax = 1024 * scale_factor
            boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            boxes = []
            mask = np.zeros((self.orig_height, self.orig_width, count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for annotation_num, annotation in enumerate(annotations):
                if annotation["Target"] == 1:
                    x = int(annotation["x"])
                    y = int(annotation["y"])
                    w = int(annotation["width"])
                    h = int(annotation["height"])
                    xmin = int(max(x * scale_factor, 0))
                    xmax = int(min((x + w) * scale_factor, self.orig_width))
                    ymin = int(max(y * scale_factor, 0))
                    ymax = int(min((y + h) * scale_factor, self.orig_height))
                    box = [xmin, ymin, xmax, ymax]

                    boxes.append(box)
                    class_ids[annotation_num] = 1
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return boxes, class_ids.astype(np.int32)


    def mask_to_bbox(self, masks, scale_factor):
        boxes = []
        for bbox in masks:
            pos = np.where(bbox[:, :])
            xmin = int(max(np.min(pos[1]) * scale_factor, 0))
            xmax = int(min(np.max(pos[1]) * scale_factor, self.orig_width))
            ymin = int(max(np.min(pos[0]) * scale_factor, 0))
            ymax = int(min(np.max(pos[0]) * scale_factor, self.orig_height))
            box = [xmin, ymin, xmax, ymax]
            boxes.append(box)
            if xmin >= xmax:
                print(xmin, xmax)
            if ymin >= ymax:
                print(ymin, ymax)
            assert xmin < xmax
            assert ymin < ymax

        torch_boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return torch_boxes


    @staticmethod
    def get_area(boxes):
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return area


    def __getitem__(self, index):
        scale_factor = self.orig_width / 1024

        boxes, labels = self.load_bbox(index, scale_factor)
        img = torch.Tensor(self.load_image(index))

        if np.all(labels == 0):
            area = self.get_area(boxes)
            iscrowd = torch.ones((len(boxes),), dtype=torch.int64)
            labels = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            area = self.get_area(boxes)
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {"image_id": torch.tensor([index]),
              "boxes": boxes,
              "labels": labels,
              "area": area,
              "iscrowd": iscrowd
             }

        return img, target


class RSNAAlbumentationsDataset(RSNADataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width, transformations = None, train=True):
        self.transformations = transformations
        RSNADataset.__init__(self, image_fps, image_annotations, orig_height, orig_width, train)
    
    
    def __getitem__(self, index):
        img, target = RSNADataset.__getitem__(self, index)
        if self.transformations is not None:
            tmpAnn = convertAnnotations(target, img)
            augmented = self.transformations(**tmpAnn)
            if len(augmented['bboxes']) != 0:
                target['boxes'] = torch.tensor([list(l[0:2]) + [l[2] + l[0], l[3] + l[1]] for l in augmented['bboxes']])
            return torch.tensor(augmented['image']).permute(2, 0, 1), target
        return img, target
