#!/usr/bin/env python3
"""
Test/Inference for NanoDet-Plus-Lite.

Usage:
    python test.py --model checkpoint.pth --image test.jpg
    python test.py --model checkpoint.pth --video test.mp4
    python test.py --model checkpoint.pth --camera 0
"""

import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import get_config
from src.models import NanoDetPlusLite
from src.data.transforms import InferenceTransform
from src.utils import draw_detections, load_checkpoint


class NanoDetPlusLiteDetector:
    """NanoDet-Plus-Lite inference wrapper."""
    
    CLASS_NAMES = [
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
    ]
    
    VIOLATION_CLASSES = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
    SAFE_CLASSES = ["Hardhat", "Mask", "Safety Vest"]
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.4
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        config = get_config()
        
        # Load checkpoint to detect model config
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Auto-detect from checkpoint metadata
        backbone_size = config.model.backbone_size
        num_classes = config.model.num_classes
        fpn_channels = config.model.fpn_out_channels
        input_size = config.model.input_size
        
        if "config" in checkpoint:
            ckpt_config = checkpoint["config"]
            backbone_size = ckpt_config.get("backbone_size", backbone_size)
            num_classes = ckpt_config.get("num_classes", num_classes)
            fpn_channels = ckpt_config.get("fpn_channels", fpn_channels)
            input_size = ckpt_config.get("input_size", input_size)
            print(f"Detected from checkpoint: backbone={backbone_size}, classes={num_classes}")
        
        self.input_size = input_size
        
        # Build model
        print(f"Loading model: {model_path}")
        print(f"Device: {self.device}")
        
        self.model = NanoDetPlusLite(
            num_classes=num_classes,
            input_size=input_size,
            backbone_size=backbone_size,
            fpn_channels=fpn_channels,
            pretrained=False,
            use_aux_head=False
        )
        
        # Load only model weights (strict=False to ignore aux_head from training)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "state_dict" in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model = self.model.to(self.device).eval()
        
        # Transform
        self.transform = InferenceTransform(input_size=self.input_size)
        
        info = self.model.get_model_info()
        print(f"Model loaded: {info['name']}")
        print(f"Parameters: {info['total_params']:,}")
    
    @torch.no_grad()
    def detect(self, image: np.ndarray):
        """
        Run detection on image.
        
        Args:
            image: BGR image
            
        Returns:
            List of (class_name, score, x1, y1, x2, y2)
        """
        h, w = image.shape[:2]
        
        # Preprocess (letterbox)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor, meta = self.transform(rgb)
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        
        # Inference
        results = self.model.predict(
            tensor, None, self.conf_thresh, self.nms_thresh
        )
        
        # Use inverse warp matrix for precise box remapping (matches official NanoDet)
        warp_matrix = meta["warp_matrix"]
        inv_warp = np.linalg.inv(warp_matrix)

        # Format results
        detections = []
        if results and len(results[0]) > 0:
            dets, labels = results[0]
            boxes_np = dets[:, :4].cpu().numpy()
            scores_np = dets[:, 4].cpu().numpy()

            # Unproject all boxes at once via inverse warp
            n = len(boxes_np)
            if n > 0:
                # Represent boxes as 4 corner points, apply inv_warp, re-fit bbox
                xy = np.ones((n * 4, 3))
                xy[:, :2] = boxes_np[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
                xy = xy @ inv_warp.T
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
                xs = xy[:, [0, 2, 4, 6]]
                ys = xy[:, [1, 3, 5, 7]]
                x1s = np.clip(xs.min(1), 0, w - 1).astype(int)
                y1s = np.clip(ys.min(1), 0, h - 1).astype(int)
                x2s = np.clip(xs.max(1), 0, w - 1).astype(int)
                y2s = np.clip(ys.max(1), 0, h - 1).astype(int)

                for i in range(n):
                    class_name = self.CLASS_NAMES[int(labels[i].cpu().item())]
                    detections.append((
                        class_name, float(scores_np[i]),
                        x1s[i], y1s[i], x2s[i], y2s[i]
                    ))
        
        return detections
    
    @staticmethod
    def count_violations(detections):
        """Count safety violations."""
        violations = []
        safe = []
        
        for det in detections:
            class_name = det[0]
            if class_name in NanoDetPlusLiteDetector.VIOLATION_CLASSES:
                violations.append(det)
            elif class_name in NanoDetPlusLiteDetector.SAFE_CLASSES:
                safe.append(det)
        
        return violations, safe


def process_image(detector, image_path, output_dir):
    """Process single image."""
    print(f"\nProcessing: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read {image_path}")
        return
    
    start = time.time()
    detections = detector.detect(image)
    elapsed = (time.time() - start) * 1000
    
    print(f"  Inference: {elapsed:.1f}ms")
    print(f"  Detections: {len(detections)}")
    
    violations, safe = detector.count_violations(detections)
    if violations:
        print(f"  Violations: {len(violations)}")
    if safe:
        print(f"  Safe PPE: {len(safe)}")
    
    # Draw and save
    output = draw_detections(image, detections)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, Path(image_path).name)
    cv2.imwrite(output_path, output)
    print(f"  Saved: {output_path}")


def process_video(detector, video_path, output_dir, show=False):
    """Process video file."""
    print(f"\nProcessing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}, Frames: {total}")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, Path(video_path).name)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        detections = detector.detect(frame)
        total_time += time.time() - start
        
        output = draw_detections(frame, detections)
        
        # FPS overlay
        current_fps = frame_count / total_time if total_time > 0 else 0
        cv2.putText(output, f"FPS: {current_fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        writer.write(output)
        frame_count += 1
        
        if show:
            cv2.imshow("NanoDet-Plus-Lite", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count}/{total} frames...")
    
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Saved: {output_path}")


def process_camera(detector, camera_id, output_dir=None):
    """Process live camera feed."""
    print(f"\nStarting camera: {camera_id}")
    print("Press 'q' to quit, 's' to screenshot")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        output = draw_detections(frame, detections)
        
        # FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Violations warning
        violations, _ = detector.count_violations(detections)
        if violations:
            cv2.putText(output, f"VIOLATIONS: {len(violations)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("NanoDet-Plus-Lite - Press Q to quit", output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and output_dir:
            save_path = os.path.join(output_dir, f"capture_{frame_count}.jpg")
            cv2.imwrite(save_path, output)
            print(f"Saved: {save_path}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="NanoDet-Plus-Lite Inference")
    parser.add_argument("--model", "-m", required=True, help="Model checkpoint")
    parser.add_argument("--image", "-i", help="Input image or directory")
    parser.add_argument("--video", "-v", help="Input video")
    parser.add_argument("--camera", type=int, help="Camera ID")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--show", action="store_true", help="Show output")
    args = parser.parse_args()
    
    if not any([args.image, args.video, args.camera is not None]):
        parser.error("Specify --image, --video, or --camera")
    
    detector = NanoDetPlusLiteDetector(
        model_path=args.model,
        device=args.device,
        conf_thresh=args.conf,
        nms_thresh=args.nms
    )
    
    if args.image:
        if os.path.isdir(args.image):
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for path in Path(args.image).glob(ext):
                    process_image(detector, str(path), args.output)
        else:
            process_image(detector, args.image, args.output)
    
    elif args.video:
        process_video(detector, args.video, args.output, args.show)
    
    elif args.camera is not None:
        process_camera(detector, args.camera, args.output)


if __name__ == "__main__":
    main()
