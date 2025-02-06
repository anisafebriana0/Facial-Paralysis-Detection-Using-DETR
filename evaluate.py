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
        
    def add_result(self, image_id, image, detections, ground_truth, scores):
        """Save evaluation result for a single image"""
        try:
            # Save image with detections
            image_path = self.images_dir / f'image_{image_id}.jpg'
            success = cv2.imwrite(str(image_path), image)
            if not success:
                print(f"Failed to save image to {image_path}")
            else:
                print(f"Saved image to {image_path}")
            
            # Store result data
            self.results.append({
                'image_id': image_id,
                'image_path': str(image_path),
                'num_detections': len(detections),
                'num_ground_truth': len(ground_truth),
                'confidence_scores': scores.tolist() if scores is not None else [],
            })
        except Exception as e:
            print(f"Error saving result for image {image_id}: {str(e)}")
    
    def save_summary(self, metrics):
        """Save evaluation summary to CSV"""
        try:
            # Save per-image results
            df_results = pd.DataFrame(self.results)
            results_path = self.output_dir / 'per_image_results.csv'
            df_results.to_csv(results_path, index=False)
            print(f"Saved per-image results to {results_path}")
            
            # Save overall metrics
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
            
            metrics_dict = {
                'Metric': metric_names,
                'Value': metrics.tolist() if metrics is not None else [0] * len(metric_names)
            }
            
            df_metrics = pd.DataFrame(metrics_dict)
            metrics_path = self.output_dir / 'metrics_summary.csv'
            df_metrics.to_csv(metrics_path, index=False)
            print(f"Saved metrics summary to {metrics_path}")
            
        except Exception as e:
            print(f"Error saving summary: {str(e)}")

def draw_boxes(image, boxes, labels, scores=None, is_gt=False, thickness=2):
    """
    Draw boxes on image with different colors for each class
    - Ground truth: Green
    - API predictions: Red
    - ASAP predictions: Blue
    """
    # Define class mapping
    class_mapping = {
        0: "background",  # if needed
        1: "low",
        2: "mid",
        3: "high"
    }
    
    img = image.copy()
    for idx, box in enumerate(boxes):
        # Convert box coordinates to integers
        x1, y1, x2, y2 = map(int, box)
        
        # Set color based on class and whether it's ground truth
        if is_gt:
            color = (255, 255, 255)  # White for ground truth
        else:
            if int(labels[idx]) == 1:    # Low
                color = (0, 255, 0)      # Green
            elif int(labels[idx]) == 2:   # Mid
                color = (0, 165, 255)     # Orange
            else:                         # High
                color = (0, 0, 255)       # Red
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label and score if provided
        if labels is not None and idx < len(labels):
            class_name = class_mapping.get(int(labels[idx]), f"Class {labels[idx]}")
            label = class_name
            if scores is not None and idx < len(scores):
                label += f" {scores[idx]:.2f}"
            
            # Put text above the box
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, thickness=2)
    
    return img

def evaluate_model(model, test_dataloader, coco_gt, logger, results_handler, image_processor):
    """Evaluate model on test dataset using COCO metrics"""
    try:
        evaluator = CocoEvaluator(coco_gt=coco_gt, iou_types=["bbox"])
        model.eval()
        
        logger.info("Starting evaluation...")
        for batch_idx, batch in enumerate(test_dataloader):
            try:
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

                # Update evaluator
                predictions = {
                    target['image_id'].item(): output 
                    for target, output in zip(labels, results)
                }
                predictions = prepare_for_coco_detection(predictions)
                evaluator.update(predictions)

                # Save visualization for each image in batch
                for idx, (target, result) in enumerate(zip(labels, results)):
                    try:
                        image_id = target['image_id'].item()
                        image = test_dataloader.dataset.coco.loadImgs(image_id)[0]
                        image_path = os.path.join(test_dataloader.dataset.root, image['file_name'])
                        image = cv2.imread(image_path)
                        
                        if image is None:
                            logger.error(f"Failed to load image: {image_path}")
                            continue
                        
                        # Get predictions
                        pred_boxes = result['boxes'].cpu().numpy()
                        pred_scores = result['scores'].cpu().numpy()
                        pred_labels = result['labels'].cpu().numpy()
                        
                        # Filter predictions by confidence
                        mask = pred_scores >= 0.5
                        pred_boxes = pred_boxes[mask]
                        pred_scores = pred_scores[mask]
                        pred_labels = pred_labels[mask]
                        
                        # Get ground truth
                        gt_annotations = test_dataloader.dataset.coco.imgToAnns[image_id]
                        gt_boxes = []
                        gt_labels = []
                        for ann in gt_annotations:
                            x, y, w, h = ann['bbox']
                            gt_boxes.append([x, y, x + w, y + h])
                            gt_labels.append(ann['category_id'])
                        gt_boxes = np.array(gt_boxes)
                        gt_labels = np.array(gt_labels)
                        
                        # Draw boxes
                        frame = image.copy()
                        # Draw predictions (red for API, blue for ASAP)
                        if len(pred_boxes) > 0:
                            frame = draw_boxes(frame, pred_boxes, pred_labels, pred_scores, 
                                            is_gt=False)
                        # Draw ground truth in green
                        if len(gt_boxes) > 0:
                            frame = draw_boxes(frame, gt_boxes, gt_labels, 
                                            is_gt=True)
                        
                        # Save the annotated image
                        save_path = os.path.join(results_handler.images_dir, f'image_{image_id}.jpg')
                        cv2.imwrite(save_path, frame)
                        logger.info(f"Saved visualization to {save_path}")
                        
                        # Store results
                        results_handler.results.append({
                            'image_id': image_id,
                            'image_path': save_path,
                            'num_detections': len(pred_boxes),
                            'num_ground_truth': len(gt_boxes),
                            'confidence_scores': pred_scores.tolist(),
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing image {image_id}: {str(e)}")
                        continue

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue

        # Compute final metrics
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        
        metrics = evaluator.coco_eval['bbox'].stats
        results_handler.save_summary(metrics)
        
        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def setup_data(logger):
    """Setup dataset and get the test dataloader"""
    try:
        # Use existing dataset
        dataset_path = "paralysis-face-7"  # This should match the path where train.py downloaded the data
        
        # Setup image processor
        logger.info("Setting up image processor...")
        image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)
        
        # Create test dataset
        logger.info("Creating test dataset...")
        test_dataset = CocoDetection(
            os.path.join(dataset_path, "test"),
            image_processor,
            train=False
        )
        
        # Create test dataloader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=lambda b: collate_fn(b, image_processor),
            num_workers=NUM_WORKERS
        )
        
        return test_dataset, test_dataloader, image_processor
        
    except Exception as e:
        logger.error(f"Failed to setup data: {str(e)}")
        raise

def main():
    try:
        # Setup logging
        logger = setup_logger('detr_evaluation')
        logger.info("Starting evaluation script")
        
        # Initialize results handler
        results_handler = EvaluationResults()
        
        # Setup data first to get number of classes
        logger.info("Setting up test dataset...")
        test_dataset, test_dataloader, image_processor = setup_data(logger)
        
        # Get number of classes from dataset
        num_classes = len(test_dataset.coco.cats)
        logger.info(f"Number of classes: {num_classes}")
        
        # Load model
        logger.info("Loading model...")
        try:
            model = Detr.from_pretrained(MODEL_PATH, num_labels=num_classes)
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {str(e)}")
            raise
            
        # Run evaluation
        metrics = evaluate_model(
            model, 
            test_dataloader, 
            test_dataset.coco, 
            logger, 
            results_handler,
            image_processor
        )
        
        # Log results
        logger.info("Evaluation completed successfully")
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

    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 