import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, AutoConfig
import cv2
from PIL import Image
import time
import os

class DetrProcessor:
    def __init__(self, model_path="../paralysis-face/model.safetensors", config_path=None, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")

        config = AutoConfig.from_pretrained("facebook/detr-resnet-50")
        config.num_labels = 4  # Set number of classes to 3
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        self.model = DetrForObjectDetection.from_pretrained(
            model_path,
            config=config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        # Update the labels
        self.model.config.id2label = {0: "LABEL_0", 1: "low", 2: "mid", 3: "high"}
        self.model.config.label2id = {"LABEL_0": 0, "low": 1, "mid": 2, "high": 3}

        self.output_dir = "./output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        print("Model loaded successfully!")

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device).type(torch.float16) if torch.cuda.is_available() else v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=self.confidence_threshold)[0]

        # Define color mapping for classes
        color_map = {
            "low": (0, 255, 0),    # Red in BGR
            "mid": (255, 0, 0),   # Green in BGR
            "high": (0, 0, 255) # Blue in BGR (for any other class)
        }

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = score.item()
            label_text = self.model.config.id2label[label.item()]
            box = box.cpu().numpy()
            x_min, y_min, x_max, y_max = map(int, box)
            detection = {
                "label": label_text,
                "confidence": confidence,
                "bbox": (x_min, y_min, x_max, y_max)
            }
            detections.append(detection)

            # Get color based on label
            color = color_map.get(label_text, (255, 0, 0))  # Default to blue if label not found

            # Draw bounding box and label with corresponding color
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{label_text}: {confidence:.2f}", (x_min, y_min - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.curr_frame_time = time.time()
        fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.curr_frame_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame, detections
    def process_video(self, video_path, output_filename="processed_video.mp4", save_frames=False):
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = os.path.join(self.output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frames_dir = os.path.join(self.output_dir, "frames")
        if save_frames:
            os.makedirs(frames_dir, exist_ok=True)

        try:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame, detections = self.process_frame(frame)
                writer.write(processed_frame)
                if save_frames:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, processed_frame)
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
        finally:
            cap.release()
            writer.release()
        print(f"Processing complete. Output saved to: {output_path}")
        return output_path

    def process_webcam(self, camera_id=0):
        print(f"Opening webcam {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam {camera_id}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, detections = self.process_frame(frame)
                cv2.imshow('Webcam Detection', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    try:
        # Initialize the processor
        processor = DetrProcessor(
            model_path="../paralysis-face/model.safetensors",
            confidence_threshold=0.5
        )
        
        # Get video filename from user input
        video_filename = input("Enter the video filename (with extension, e.g., test.mp4): ")
        video_path = os.path.join(os.path.dirname(__file__), video_filename)
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            print(f"Current working directory: {os.getcwd()}")
            return
            
        # Get output filename (use input filename with 'processed_' prefix)
        output_filename = f"processed_{video_filename}"
        
        print(f"Processing video: {video_filename}")
        print(f"Output will be saved as: {output_filename}")
        
        processor.process_video(
            video_path=video_path,
            output_filename=output_filename,
            save_frames=False
        )
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()