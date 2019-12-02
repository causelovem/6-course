import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit, KFold
import numpy as np
import torch
import random
import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pydicom
from PIL import Image
import cv2

import sys
import math
import json
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision import transforms

# https://github.com/albumentations-team/albumentations/blob/master/notebooks/example_bboxes.ipynb
from albumentations import (
    BboxParams,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose,
    Flip,
    Rotate,
    RandomSizedBBoxSafeCrop,
    RandomSizedCrop,
    RandomCropNearBBox
)

import utils
from torchvision.ops import nms


def fix_seed(seed=23):
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# # частичная загрузка данных
# def parse_dataset(data_dir, anns):
#     image_fps = []
#     image_annotations = {}
#     i = 1
#     for index, row in anns.iterrows():
#         fp = os.path.join(data_dir, row["patientId"] + ".dcm")
#         image_fps.append(fp)
#         image_annotations[fp] = image_annotations.get(fp, []) + [row]
#         if i == 50:
#             break
#         i += 1
#     return image_fps, image_annotations


def makeNewAnnotations(oldAnnotation, image):
    boxes = oldAnnotation['boxes'].tolist()
    return {'image': np.array(image.permute(1, 2, 0)), 'category_id': oldAnnotation['labels'].tolist(),
            'bboxes': [] if oldAnnotation['labels'][0] == 0 else
            [cords[0:2] + [cords[2] - cords[0], cords[3] - cords[1]] for cords in boxes]}


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=(255, 0, 0), thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)
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


class RSNADatasetWithAlbument(RSNADataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width, transformations=None, train=True):
        self.transformations = transformations
        RSNADataset.__init__(self, image_fps, image_annotations, orig_height, orig_width, train)

    def __getitem__(self, index):
        img, target = RSNADataset.__getitem__(self, index)
        if self.transformations is not None:
            tmpAnn = makeNewAnnotations(target, img)
            augmented = self.transformations(**tmpAnn)
            if len(augmented['bboxes']) != 0:
                target['boxes'] = torch.tensor([list(cords[0:2]) + [cords[2] + cords[0], cords[3] + cords[1]] for cords in augmented['bboxes']])
            return torch.tensor(augmented['image']).permute(2, 0, 1), target
        return img, target


def visualize_random_image(dataset):
    class_ids = [0]
    image_id = 1
    while class_ids[0] == 0:
        image_id = random.choice(range(len(dataset.image_fps)))
        # image_fp = dataset.image_fps[image_id]
        image = dataset.show_image(image_id)
        boxes, class_ids = dataset.load_bbox(image_id, scale_factor=1)
        if len(boxes) == 0:
            continue

        mask_instance = np.zeros((dataset.orig_height, dataset.orig_width), dtype=np.uint8)
        xmin, ymin, xmax, ymax = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
        mask = cv2.rectangle(mask_instance, (xmin, ymin), (xmax, ymax), 255, -1)
        mask = np.array(mask)

    print(image_id)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    masked = np.zeros(image.shape[:2])
    masked = image[:, :, 0] * mask

    plt.imshow(masked, cmap='gray')
    plt.axis('off')


def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    w1 = x12 - x11
    h1 = y12 - y11
    w2 = x22 - x21
    h2 = y22 - y21

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, x2, y2)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, x2, y2)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    # Предобученная на COCO fasterrcnn
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Заменим "голову" классификатора на новую, которую обучим на нашем датасете
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                       aspect_ratios=((0.25, 0.5, 1.0, 1.5, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=32,
                                                    sampling_ratio=2)
    model.box_roi_pool = roi_pooler
    model.rpn_anchor_generator = anchor_generator

    return model


def evaluate(model, data_loader, device):
    boxes_pred = []
    boxes_true = []
    scores = []
    summ = 0
    count = 0
    with torch.no_grad():
        model.eval()

        for images, targets in tqdm_notebook(data_loader):
            boxes_true_mini_batch = [np.array(item["boxes"]) for item in targets]
            labels_true_mini_batch = [np.array(item["labels"]) for item in targets]
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            torch.cuda.synchronize()
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            boxes_pred_mini_batch = [np.array(res["boxes"].to("cpu")) for res in outputs]
            scores_mini_batch = [np.array(res["scores"].to("cpu")) for res in outputs]
            labels_mini_batch = [np.array(res["labels"].to("cpu")) for res in outputs]

            for img_num in range(len(images)):
                # Если на картинке нет пневмонии
                if np.all(labels_true_mini_batch[img_num] == 0):
                    if (labels_mini_batch[img_num].size == 0) or np.all(labels_mini_batch[img_num] == 0):
                        continue
                    else:
                        # Мы сказали, что есть
                        count += 1
                else:
                    # Если пневмония есть считаем map_iou
                    curr_map_iou = map_iou(boxes_true_mini_batch[img_num],
                                           boxes_pred_mini_batch[img_num],
                                           scores_mini_batch[img_num])
                    summ += curr_map_iou
                    count += 1

    return summ / count


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def make_validation(images_files, image_annotations, cv, params, dataset="RSNADataset", transformations=None):
    img_size = params.get("img_size", 1024)
    num_classes = params.get("num_classes", 2)
    num_epochs = params.get("num_epochs", 5)
    device = params.get("device", "cpu")
    batch_size = 6

    # # для ускорения обучения комментируем
    # scores = np.zeros(len(cv))
    for fold_num, (train_idx, val_idx) in enumerate(cv):
        train_images = list(images_files[train_idx])
        # Будем обучаться только на изображениях с пневмонией
        train_images = [filename for filename in train_images if image_annotations[filename][0].Target > 0]

        # # для ускорения обучения комментируем
        # val_images = list(images_files[val_idx])

        if dataset == "RSNADataset":
            dataset = RSNADataset(train_images, image_annotations, img_size, img_size, train=True)
        elif dataset == "RSNADatasetWithAlbument":
            dataset = RSNADatasetWithAlbument(train_images, image_annotations, img_size, img_size,
                                              transformations=transformations, train=True)

        # # для ускорения обучения комментируем
        # dataset_val = RSNADataset(val_images, image_annotations, img_size, img_size, train=False)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=collate_fn)

        # # для ускорения обучения комментируем
        # data_loader_val = torch.utils.data.DataLoader(
        #     dataset_val, batch_size=batch_size, shuffle=False, num_workers=6,
        #     collate_fn=collate_fn)

        model = get_model(num_classes)

        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.00005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        for epoch in range(num_epochs):
            print("epoch {}".format(epoch))
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

            lr_scheduler.step()

            # # для ускорения обучения комментируем
            # fold_score = evaluate(model, data_loader_val, device=device)
            # print("score: {}".format(fold_score))
            # scores[fold_num] = fold_score

        torch.save(model.state_dict(), "fold_num_{}_model".format(fold_num))
        del model

    # # для ускорения обучения комментируем
    # print("average val score: {}".format(np.mean(scores)))


def load_test_image(img_path, img_size):
    ds = pydicom.read_file(img_path)
    image = ds.pixel_array
    image = np.array(Image.fromarray(image).resize((img_size, img_size)))
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    image = np.rollaxis(image, 2, 0) / 255
    return torch.Tensor(image)


def get_test_predictions(model, test_images, device, img_size):
    """
    Предсказания для теста
    """
    # sub = []
    # min_conf = 0
    imgs_info = []
    # scale_factor = 1024 / img_size
    with torch.no_grad():
        model.eval()

        for img_path in tqdm_notebook(test_images):
            images = [load_test_image(img_path, img_size)]
            images = list(img.to(device) for img in images)

            torch.cuda.synchronize()
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

            boxes_pred_mini_batch = [np.array(res["boxes"].to("cpu")) for res in outputs]
            scores_mini_batch = [np.array(res["scores"].to("cpu")) for res in outputs]

            imagesLen = len(images)
            for i in range(imagesLen):
                # patient_id = img_path.split(".dcm")[0].split("stage_2_test_images/")[-1]
                patient_id = img_path.split(".dcm")[0].split("stage_2_test_images\\")[-1]
                img_info = dict()
                img_info["patient_id"] = patient_id
                img_info["boxes"] = boxes_pred_mini_batch[i]
                img_info["scores"] = scores_mini_batch[i]
                imgs_info.append(img_info)
    return imgs_info


def get_sub_list(imgs_info, img_size, min_conf=0.7):
    """
    Записываем предсказания в правильном формате
    """
    sub = []
    scale_factor = 1024 / img_size
    for img_info in imgs_info:
        patient_id = img_info["patient_id"]
        boxes_pred_mini_batch = img_info["boxes"]
        scores_mini_batch = img_info["scores"]

        result_str = "{},".format(patient_id)
        for bbox_num in range(boxes_pred_mini_batch.shape[0]):
            if scores_mini_batch[bbox_num] > min_conf:
                result_str += " {:1.2f} ".format(np.round(scores_mini_batch[bbox_num], 2))
                x_min = int(np.round(boxes_pred_mini_batch[bbox_num, 0] * scale_factor))
                y_min = int(np.round(boxes_pred_mini_batch[bbox_num, 1] * scale_factor))
                width = int(np.round(boxes_pred_mini_batch[bbox_num, 2] * scale_factor)) - x_min
                height = int(np.round(boxes_pred_mini_batch[bbox_num, 3] * scale_factor)) - y_min
                result_str += "{} {} {} {}".format(x_min, y_min, width, height)
        sub.append(result_str + "\n")
    return sub


def write_submission(sub, filename="submission.csv"):
    with open(filename, mode="w") as f:
        header = "patientId,PredictionString\n"
        f.write(header)
        for line in sub:
            f.write(line)


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area,
                                               min_visibility=min_visibility, label_fields=['category_id']))


def get_test_predictions_nms(models, test_images, device, img_size):
    modelListRes = []
    for m in models:
        modelListRes.append(get_test_predictions(m, test_images, device, img_size))

    resDict = {}
    for modelRes in modelListRes:
        for res in modelRes:
            resDict[res['patient_id']] = resDict.get(res['patient_id'], {'patient_id': res['patient_id']})

            if resDict[res['patient_id']].get('boxes', np.array([])).any():
                resDict[res['patient_id']]['boxes'] = np.vstack((resDict[res['patient_id']]['boxes'],
                                                                 res['boxes']))
            else:
                resDict[res['patient_id']]['boxes'] = res['boxes']

            if resDict[res['patient_id']].get('scores', np.array([])).any():
                resDict[res['patient_id']]['scores'] = np.concatenate((resDict[res['patient_id']]['scores'],
                                                                       res['scores']))
            else:
                resDict[res['patient_id']]['scores'] = res['scores']

    resList = []
    for key, value in resDict.items():
        boxes = torch.tensor(value['boxes'])
        scores = torch.tensor(value['scores'])
        nmsResult = nms(boxes, scores, 0.3)
        imgInfo = {'patient_id': key, 'boxes': [], 'scores': []}

        for i in range(len(nmsResult)):
            imgInfo['boxes'].append(boxes[nmsResult[i]].tolist())
            imgInfo['scores'].append(scores[nmsResult[i]].item())
            if i == 2:
                break

        imgInfo['boxes'] = np.array(imgInfo['boxes'])
        resList.append(imgInfo)
    return resList


if __name__ == '__main__':
    fix_seed(23)

    data_path = "data/"
    train_img_path = os.path.join(data_path, "stage_2_train_images")
    annotations = pd.read_csv(os.path.join(data_path, "stage_2_train_labels.csv"))

    image_fps, image_annotations = parse_dataset(train_img_path, anns=annotations)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 2
    img_size = 1024
    # Для аккуратной валидации используйте больше разбиений
    n_splits = 3
    num_epochs = 3
    print("your device: {}".format(device))

    # Аугментация
    # aug = get_aug([Flip(p=0.5), Rotate(limit=15, p=0.5), RandomSizedBBoxSafeCrop(img_size, img_size, p=0.5)])
    # aug = get_aug([Flip(p=0.5), Rotate(limit=15, p=0.5), RandomSizedCrop((100, 800), img_size, img_size, p=0.5)])
    # aug = get_aug([Flip(p=0.5), Rotate(limit=15, p=0.5), RandomCropNearBBox(p=0.5)])
    aug = get_aug([Flip(p=0.5), Rotate(limit=15, p=0.5)])

    cv = list(ShuffleSplit(n_splits=n_splits, random_state=15,
                           test_size=0.2).split(np.zeros((len(image_fps), 1)), np.zeros(len(image_fps))))

    params = {"img_size": img_size, "num_classes": num_classes, "num_epochs": num_epochs, "device": device}

    images_files = np.array(image_fps)

    # make_validation(images_files=images_files, image_annotations=image_annotations, cv=cv, params=params, dataset="RSNADataset", transformations=None)
    make_validation(images_files=images_files, image_annotations=image_annotations, cv=cv, params=params, dataset="RSNADatasetWithAlbument", transformations=aug)

    # Predict
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load("fold_num_0_model"))
    model.to(device)

    test_images = get_dicom_fps(os.path.join(data_path, "stage_2_test_images"))

    imgs_info = get_test_predictions(model, test_images, device, img_size)

    # min_conf -- минимальный порог уверенности для того, чтобы считать объект пневмонией
    sub_list = get_sub_list(imgs_info, img_size, min_conf=0.7)

    write_submission(sub_list, "submissionAug.csv")

    # CV
    model0 = get_model(num_classes=num_classes)
    model0.load_state_dict(torch.load("fold_num_0_model"))
    model0.to(device)

    model1 = get_model(num_classes=num_classes)
    model1.load_state_dict(torch.load("fold_num_1_model"))
    model1.to(device)

    model2 = get_model(num_classes=num_classes)
    model2.load_state_dict(torch.load("fold_num_2_model"))
    model2.to(device)

    # test_images = get_dicom_fps(os.path.join(data_path, "stage_2_test_images"))

    imgs_info = get_test_predictions_nms([model0, model1, model2], test_images, device, img_size)

    # min_conf -- минимальный порог уверенности для того, чтобы считать объект пневмонией
    sub_list = get_sub_list(imgs_info, img_size, min_conf=0.7)

    write_submission(sub_list, "submissionCV.csv")
