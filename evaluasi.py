import os
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
import cv2
import supervision as sv
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from config import *
from data.dataset import CocoDetection, collate_fn
from models.detr import Detr
from utils.helpers import prepare_for_coco_detection
from utils.logging_config import setup_logger
from coco_eval import CocoEvaluator


class EvaluationResults:
    def __init__(self, output_dir='evaluation_results'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        print(f"Created output directory: {self.output_dir}")
        self.results = []


def add_legend(image):
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    thickness = 1
    padding = 4

    legends = [
        ((0, 255, 0), "GT"),
        ((0, 0, 255), "Low"),
        ((255, 255, 0), "Mid"),
        ((255, 0, 0), "High")
    ]

    for i, (color, text) in enumerate(legends):
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        rect_start = (padding, padding + i * (text_height + padding))
        rect_end = (padding + text_width + padding * 2, padding + (i + 1) * text_height + padding)
        cv2.rectangle(image, rect_start, rect_end, color, -1)
        text_pos = (padding + padding, padding + (i + 1) * text_height)
        cv2.putText(image, text, text_pos, font, font_scale, (255, 255, 255), thickness)

    return image


def draw_predictions(image, gt_annotations, predictions, output_path, class_names):
    """
    Draw bounding boxes for ground truth and predictions on the image.
    Added confidence threshold filtering.
    """
    # Draw ground truth boxes
    for gt in gt_annotations:
        x, y, w, h = gt['bbox']
        class_name = class_names[gt['category_id']]
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(image, f"GT: {class_name}", (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Filter predictions by confidence threshold
    confidence_threshold = 0.5
    filtered_predictions = [pred for pred in predictions if pred['score'] >= confidence_threshold]

    # Draw predicted boxes
    for pred in filtered_predictions:
        x_min, y_min, x_max, y_max = pred['bbox']
        class_name = class_names[pred['label']]
        score = pred['score']
        color = (255, 0, 0) if pred['label'] == 1 else (0, 0, 255)
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                      color, 2)
        cv2.putText(image, f"Pred: {class_name} ({score:.2f})",
                    (int(x_min), int(y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Add legend
    image = add_legend(image)
    cv2.imwrite(output_path, image)

# Menghitung Intersection over Union (IoU) antara dua bounding box untuk mengevaluasi kecocokan antara prediksi dan GT.
# Ground truth dan prediksi dibandingkan menggunakan IoU.
# Prediksi yang memiliki IoU > threshold (default 0.5) dianggap cocok.
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0


def compute_confusion_matrix_with_multiple_predictions(predictions, gt_annotations, iou_threshold=0.5, num_classes=4):
    """
    Compute confusion matrix for object detection results.

    Args:
        predictions: List of prediction dictionaries with 'label', 'bbox', and 'score'
        gt_annotations: List of ground truth dictionaries with 'category_id' and 'bbox'
        iou_threshold: IoU threshold for considering a match
        num_classes: Number of classes including background
    """
    cm = np.zeros((num_classes , num_classes), dtype=int)

    # Convert ground truth boxes from [x, y, w, h] to [x1, y1, x2, y2] format
    gt_boxes_converted = []
    for gt in gt_annotations:
        x, y, w, h = gt['bbox']
        gt_boxes_converted.append({
            'bbox': [x, y, x + w, y + h],
            'category_id': gt['category_id'],
            'matched': False
        })

    # Sort predictions by confidence score (highest first)
    sorted_predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # Match predictions to ground truth boxes
    for pred in sorted_predictions:
        pred_label = pred['label']
        pred_bbox = pred['bbox']  # Assuming already in [x1, y1, x2, y2] format

        best_iou = 0
        best_gt_idx = -1

        # Find the best matching ground truth box for this prediction
        for idx, gt in enumerate(gt_boxes_converted):
            if not gt['matched']:  # Only consider unmatched ground truth boxes
                iou = calculate_iou(pred_bbox, gt['bbox'])
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

        if best_gt_idx >= 0:
            # We found a match - update confusion matrix and mark the GT as matched
            gt_label = gt_boxes_converted[best_gt_idx]['category_id']
            cm[gt_label][pred_label] += 1
            gt_boxes_converted[best_gt_idx]['matched'] = True
        else:
            # False positive - prediction without a matching ground truth
            cm[-1][pred_label] += 1

    # Count unmatched ground truth boxes as false negatives
    for gt in gt_boxes_converted:
        if not gt['matched']:
            gt_label = gt['category_id']
            cm[gt_label][-1] += 1

    return cm

def visualize_confusion_matrix(cm, class_names, save_path):
    # Remove "eyes-mouth" label and its corresponding row and column
    eyes_mouth_index = class_names.index("eyes-mouth")
    class_names = [name for i, name in enumerate(class_names) if i != eyes_mouth_index]
    cm = np.delete(cm, eyes_mouth_index, axis=0)  # Remove row
    cm = np.delete(cm, eyes_mouth_index, axis=1)  # Remove column

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix for Object Detection")
    plt.savefig(save_path)
    plt.close()


# Menggunakan CocoEvaluator untuk menghitung mAP dan AR.
def evaluate_model(model, test_dataloader, coco_gt, logger, image_processor):
    evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=["bbox"])
    model.eval()

    all_pred_boxes = []
    all_gt_annotations = []
    results_handler = EvaluationResults()
    # Assuming class names are available in dataset
    class_names = [cat['name'] for cat in test_dataloader.dataset.coco.loadCats(test_dataloader.dataset.coco.getCatIds())]

    for batch_idx, batch in enumerate(test_dataloader):
        pixel_values = batch["pixel_values"].to(DEVICE)
        pixel_mask = batch["pixel_mask"].to(DEVICE)
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=orig_target_sizes,
            threshold=0.5
        )
        predictions = {
            target['image_id'].item(): output
            for target, output in zip(labels, results)
        }
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)

        for idx, (target, result) in enumerate(zip(labels, results)):
            image_id = target['image_id'].item()
            pred_boxes = [{
                'label': int(label),
                'bbox': bbox.tolist(),
                'score': float(score)  # Add confidence score
            } for label, bbox, score in zip(result['labels'].cpu().numpy(), result['boxes'].cpu().numpy(), result['scores'].cpu().numpy())]

            gt_annotations = test_dataloader.dataset.coco.imgToAnns[image_id]
            original_filename = test_dataloader.dataset.coco.loadImgs(image_id)[0]['file_name']


            image_path = os.path.join(test_dataloader.dataset.root, original_filename)
            image = cv2.imread(image_path)

            output_filename = os.path.splitext(original_filename)[0] + "_annotated.png"
            output_path = os.path.join(results_handler.images_dir, output_filename)

            draw_predictions(image, gt_annotations, pred_boxes, output_path, class_names)

            all_pred_boxes.extend(pred_boxes)
            all_gt_annotations.extend(gt_annotations)

    cm = compute_confusion_matrix_with_multiple_predictions(all_pred_boxes, all_gt_annotations, iou_threshold=0.5, num_classes=len(test_dataloader.dataset.coco.cats))
    visualize_confusion_matrix(cm, class_names, save_path=os.path.join(results_handler.output_dir, "confusion_matrix.png"))

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = evaluator.coco_eval['bbox'].stats
    metrics = evaluator.coco_eval['bbox'].stats
    overall_ap = metrics[0]  # AP @ IoU=0.50:0.95
    ap_iou_50 = metrics[1]  # AP @ IoU=0.50
    overall_ar = metrics[8]  # AR with max detections = infinity

    mAP = (overall_ap + ap_iou_50) / 2  # Calculating mAP
    logger.info(f"Overall AP: {overall_ap:.3f}")
    logger.info(f"AP @ IoU=0.50: {ap_iou_50:.3f}")
    logger.info(f"Overall AR: {overall_ar:.3f}")
    logger.info(f"mAP: {mAP:.3f}")
    return metrics

def main():
    logger = setup_logger('detr_evaluation')
    logger.info("Starting evaluation script")
    

    dataset_path = "paralysis-face-11"
    image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
    test_dataset = CocoDetection(
        os.path.join(dataset_path, "test"),
        image_processor,
        train=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda b: collate_fn(b, image_processor),
        num_workers=NUM_WORKERS
    )


    num_classes = len(test_dataset.coco.cats)
    model = Detr.from_pretrained(MODEL_PATH, num_labels=num_classes)
    model.to(DEVICE)

    metrics = evaluate_model(
        model,
        test_dataloader,
        test_dataset.coco,
        logger,
        image_processor
    )
    metric_names = [
        'AP @ IoU=0.50:0.95',
        'AP @ IoU=0.50',
        'AP @ IoU=0.75',
        'AP for small objects',
        'AP for medium objects',
        'AP for large objects',
        'AR with max 1 detection',
        'AR with max 10 detections',
        'AR with max 100 detections',
        'AR for small objects',
        'AR for medium objects',
        'AR for large objects'
    ]
    for name, value in zip(metric_names, metrics):
        logger.info(f"{name}: {value:.3f}")

if __name__ == "__main__":
    main()

# mAP (Mean Average Precision): Mengukur performa model untuk berbagai tingkat threshold IoU.
# AR (Average Recall): Mengukur kemampuan model untuk menangkap semua objek.
# Confusion Matrix: Menyediakan evaluasi per kategori, termasuk FP, FN, dan True Positives (TP).

