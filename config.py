import os
import torch

# Path configurations
HOME = os.getcwd()
MODEL_PATH = os.path.join(HOME, 'paralysis-face')
ANNOTATION_FILE_NAME = "_annotations.coco.json"

# Model configurations
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = 'facebook/detr-resnet-50'
CONFIDENCE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.8

# Training configurations
MAX_EPOCHS = 300
BATCH_SIZE = 32
NUM_WORKERS = 3

# API Keys
ROBOFLOW_API_KEY = "7KpUTuDNEdsjoyIauZng" 