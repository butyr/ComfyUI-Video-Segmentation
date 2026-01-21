import os
import uuid
import folder_paths
import numpy as np
import logging
from pathlib import Path

import torch
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)

# Try to import VIDEO input type from ComfyUI API
try:
    from comfy_api.input import VideoInput
except ImportError:
    VideoInput = None
    logger.info("ComfyUI API VideoInput not available, will accept string paths only")

# Model directory setup
model_directory = os.path.join(folder_paths.models_dir, "VLM")
os.makedirs(model_directory, exist_ok=True)

# Global model cache to avoid reloading
_model_cache = {}


def _load_transnet_model(device):
    """Load TransNetV2 model, using cache if available."""
    import transnetv2_pytorch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cache_key = device
    if cache_key in _model_cache:
        logger.info(f"Using cached TransNetV2 model on {device}")
        return _model_cache[cache_key]

    model_name = "transnetv2-pytorch-weights"
    model_path = os.path.join(model_directory, model_name)
    pytorch_weights_path = os.path.join(model_path, "transnetv2-pytorch-weights.pth")

    # Download if not exists
    if not os.path.exists(pytorch_weights_path):
        os.makedirs(model_path, exist_ok=True)
        from huggingface_hub import hf_hub_download

        print("Downloading TransNetV2 model from Hugging Face...")
        hf_hub_download(
            repo_id="MiaoshouAI/transnetv2-pytorch-weights",
            filename="transnetv2-pytorch-weights.pth",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        logger.info(f"Downloaded TransNetV2 weights to {model_path}")

    # Load model
    model_instance = transnetv2_pytorch.TransNetV2()
    model_instance.load_state_dict(torch.load(pytorch_weights_path, map_location=device))
    model_instance = model_instance.to(device)
    model_instance.eval()

    transnet_model = {
        "model": model_instance,
        "model_path": model_path,
        "device": device,
    }

    _model_cache[cache_key] = transnet_model
    logger.info(f"TransNetV2 model loaded on {device}")

    return transnet_model


class DownloadAndLoadTransNetModel:
    """
    A ComfyUI node for downloading and loading TransNetV2 models.
    Automatically downloads from Hugging Face (MiaoshouAI/transnetv2-pytorch-weights) if not found locally.
    Note: TransNetV2_Run now loads the model automatically, so this node is optional.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    [
                        "transnetv2-pytorch-weights",
                    ],
                    {"default": "transnetv2-pytorch-weights"},
                ),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {"default": "auto"},
                ),
            },
        }

    RETURN_TYPES = ("TRANSNET_MODEL",)
    RETURN_NAMES = ("TransNet_model",)
    FUNCTION = "DownloadAndLoadTransNetModel"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def DownloadAndLoadTransNetModel(self, model, device):
        # model parameter kept for interface compatibility (only one model available)
        _ = model
        return (_load_transnet_model(device),)


class TransNetV2_Run:
    """
    A ComfyUI node for video scene detection using TransNetV2.
    Automatically loads model if not provided. Returns timestamps for detected scene boundaries.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "TransNet_model": ("TRANSNET_MODEL",),
            },
            "required": {
                "video": ("VIDEO",),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "min_scene_length": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
                "device": (
                    ["auto", "cpu", "cuda"],
                    {"default": "auto"},
                ),
            },
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("timestamps", "timestamps_string")
    FUNCTION = "TransNetV2_Run"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def TransNetV2_Run(self, video, threshold, min_scene_length, device, TransNet_model=None):
        # Load model if not provided (auto-load with caching)
        if TransNet_model is None:
            TransNet_model = _load_transnet_model(device)

        # Handle video input - convert to temporary file if needed
        video_path, is_temp_file = self._handle_video_input(video)

        try:
            # Run TransNetV2 scene detection
            timestamps = self._run_transnetv2(
                TransNet_model,
                video_path,
                threshold,
                min_scene_length
            )

            # Convert timestamps list to string format (start-end per line)
            timestamps_string = "\n".join(
                f"{start:.3f}-{end:.3f}" for start, end in timestamps
            )

            logger.info(f"Successfully detected {len(timestamps)} scenes")
            return (timestamps, timestamps_string)

        finally:
            # Clean up temp file if we created one
            if is_temp_file and os.path.exists(video_path):
                os.remove(video_path)
                logger.info(f"Cleaned up temporary file: {video_path}")

    def _handle_video_input(self, video):
        """Handle VIDEO input type and convert to file path."""
        if VideoInput and isinstance(video, VideoInput):
            unique_id = uuid.uuid4().hex
            video_path = (
                Path(folder_paths.temp_directory) / f"temp_video_{unique_id}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video.save_to(
                str(video_path),
                format="mp4",
                codec="h264",
            )
            logger.info(f"Video saved to temporary path: {video_path}")
            return str(video_path), True

        elif isinstance(video, str):
            if os.path.isfile(video):
                return video, False
            else:
                raise FileNotFoundError(f"Video file not found: {video}")

        else:
            raise TypeError(f"Unsupported video input type: {type(video)}. Expected VideoInput or string path.")

    def _run_transnetv2(self, transnet_model, video_path, threshold, min_scene_length):
        """Run TransNetV2 scene detection and return timestamps."""
        import cv2

        model = transnet_model["model"]

        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video properties: {total_frames} frames, {fps} fps, {width}x{height}")

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if not frames:
            raise RuntimeError("No frames could be read from video")

        frames_array = np.array(frames)

        logger.info("Running TransNetV2 scene detection...")

        # Resize frames to TransNetV2 expected size (48x27)
        resized_frames = []
        for frame in frames_array:
            pil_image = Image.fromarray(frame.astype(np.uint8))
            resized_pil = pil_image.resize((48, 27), Image.Resampling.BILINEAR)
            resized_frame = np.array(resized_pil, dtype=np.uint8)
            resized_frames.append(resized_frame)

        # Stack frames: (T, H, W, C) then add batch dim: (1, T, H, W, C)
        frames_array_resized = np.stack(resized_frames, axis=0)
        frames_array_batch = frames_array_resized[np.newaxis, ...]

        # Convert to uint8 tensor and move to device
        frames_tensor = torch.from_numpy(frames_array_batch).to(dtype=torch.uint8)
        frames_tensor = frames_tensor.to(transnet_model["device"])

        logger.info(f"Input tensor shape: {frames_tensor.shape}")

        # Run inference
        with torch.no_grad():
            predictions = model(frames_tensor)

            if isinstance(predictions, tuple):
                one_hot_predictions, _ = predictions
                single_frame_predictions = one_hot_predictions
            else:
                single_frame_predictions = predictions

        single_frame_predictions = single_frame_predictions.cpu().numpy().squeeze()

        # Find scene boundaries
        scenes = self._find_scenes(single_frame_predictions, threshold, min_scene_length)

        # Convert frame indices to timestamps
        timestamps = [(start_frame / fps, end_frame / fps) for start_frame, end_frame in scenes]

        logger.info(f"Detected {len(timestamps)} scenes")
        for i, (start, end) in enumerate(timestamps):
            logger.info(f"  Scene {i+1}: {start:.3f}s - {end:.3f}s")

        return timestamps

    def _find_scenes(self, predictions, threshold, min_scene_length):
        """Find scene boundaries from TransNetV2 predictions."""
        predictions_binary = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0

        for i, t in enumerate(predictions_binary):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t

        if t == 0:
            scenes.append([start, len(predictions_binary)])

        if len(scenes) == 0:
            return [(0, len(predictions_binary))]

        # Apply minimum scene length filtering
        filtered_scenes = []
        for start, end in scenes:
            if end - start >= min_scene_length:
                filtered_scenes.append((start, end))
            else:
                if filtered_scenes:
                    prev_start, _ = filtered_scenes[-1]
                    filtered_scenes[-1] = (prev_start, end)
                else:
                    filtered_scenes.append((start, end))

        return filtered_scenes if filtered_scenes else [(0, len(predictions_binary))]


class SelectVideo:
    """A ComfyUI node for selecting a specific timestamp from detected scenes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timestamps": ("LIST",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("start_time", "end_time", "timestamp_string")
    FUNCTION = "select_video"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def select_video(self, timestamps, index):
        if timestamps is None or len(timestamps) == 0:
            raise ValueError("No timestamps provided")

        if index < 0:
            raise IndexError(f"Index {index} is negative")
        elif index >= len(timestamps):
            raise IndexError(f"Index {index} is out of range (max: {len(timestamps)-1})")

        start_time, end_time = timestamps[index]
        timestamp_string = f"{start_time:.3f}-{end_time:.3f}"
        logger.info(f"Selected scene {index}: {timestamp_string}")

        return (start_time, end_time, timestamp_string)


class ZipCompress:
    """
    A ComfyUI node for compressing files into a zip archive.
    Note: This node is kept for backward compatibility but is less useful
    since TransNetV2_Run now returns timestamps instead of video files.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "file_paths_string": ("STRING",),
            },
            "required": {
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI",
                    "multiline": False,
                }),
                "output_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty for temp directory"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("zip_filename",)
    FUNCTION = "compress_files"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def compress_files(self, filename_prefix, output_dir, file_paths_string=None):
        """Compress files into a zip archive."""
        import zipfile
        import datetime

        if file_paths_string is None or file_paths_string.strip() == "":
            raise ValueError("No file paths provided for compression")

        # Convert path string to list
        file_paths = [path.strip() for path in file_paths_string.split('\n') if path.strip()]

        if not file_paths:
            raise ValueError("No valid file paths found")

        # Set output directory
        if not output_dir:
            output_dir = folder_paths.temp_directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate zip filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{filename_prefix}_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_filename)

        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    zipf.write(file_path, filename)
                    logger.info(f"Added to zip: {filename}")
                else:
                    raise FileNotFoundError(f"File not found: {file_path}")

        if not os.path.exists(zip_path):
            raise RuntimeError("Failed to create zip file")

        file_size = os.path.getsize(zip_path)
        logger.info(f"Created zip file: {zip_path} ({file_size / (1024*1024):.2f} MB)")

        return (os.path.abspath(zip_path),)


class SendToWebhook:
    """
    A ComfyUI node for sending JSON data to an external webhook/API.
    Useful for integrating with external services in API mode.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webhook_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "https://your-service.com/webhook"
                }),
            },
            "optional": {
                "timestamps": ("LIST",),
                "timestamps_string": ("STRING", {"forceInput": True}),
                "job_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Job/request ID for correlation"
                }),
                "custom_data": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Additional JSON data (optional)"
                }),
                "headers": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Custom headers as JSON, e.g. {\"Authorization\": \"Bearer token\"}"
                }),
                "timeout": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 300,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("response_body", "status_code", "success")
    FUNCTION = "send_to_webhook"
    CATEGORY = "MiaoshouAI Video Segmentation"

    def send_to_webhook(
        self,
        webhook_url,
        timestamps=None,
        timestamps_string=None,
        job_id="",
        custom_data="",
        headers="",
        timeout=30
    ):
        import json
        import urllib.request
        import urllib.error

        if not webhook_url or not webhook_url.strip():
            raise ValueError("webhook_url is required")

        # Build payload
        payload = {}

        # Add job_id if provided (for request correlation)
        if job_id and job_id.strip():
            payload["job_id"] = job_id.strip()

        # Add timestamps if provided
        if timestamps is not None:
            payload["timestamps"] = [
                {"start": start, "end": end} for start, end in timestamps
            ]
            payload["scene_count"] = len(timestamps)

        # Add timestamps string if provided
        if timestamps_string:
            payload["timestamps_string"] = timestamps_string

        # Parse and merge custom data if provided
        if custom_data and custom_data.strip():
            try:
                custom_json = json.loads(custom_data)
                if isinstance(custom_json, dict):
                    payload.update(custom_json)
                else:
                    payload["custom_data"] = custom_json
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in custom_data: {e}")

        # Parse headers if provided
        request_headers = {"Content-Type": "application/json"}
        if headers and headers.strip():
            try:
                custom_headers = json.loads(headers)
                if isinstance(custom_headers, dict):
                    request_headers.update(custom_headers)
                else:
                    raise ValueError("headers must be a JSON object")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in headers: {e}")

        # Encode payload
        payload_bytes = json.dumps(payload).encode("utf-8")

        logger.info(f"Sending webhook to: {webhook_url}")
        logger.info(f"Payload: {json.dumps(payload, indent=2)}")

        # Create and send request
        request = urllib.request.Request(
            webhook_url,
            data=payload_bytes,
            headers=request_headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = response.getcode()
                response_body = response.read().decode("utf-8")
                logger.info(f"Webhook response: {status_code} - {response_body[:200]}")
                return (response_body, status_code, True)

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            logger.error(f"Webhook HTTP error: {e.code} - {error_body}")
            raise RuntimeError(f"Webhook failed with HTTP {e.code}: {error_body}")

        except urllib.error.URLError as e:
            logger.error(f"Webhook URL error: {e.reason}")
            raise RuntimeError(f"Webhook failed: {e.reason}")

        except TimeoutError:
            logger.error(f"Webhook timeout after {timeout}s")
            raise RuntimeError(f"Webhook timeout after {timeout} seconds")


# Node mappings for ComfyUI - keeping original MiaoshouAI names
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadTransNetModel": DownloadAndLoadTransNetModel,
    "TransNetV2_Run": TransNetV2_Run,
    "SelectVideo": SelectVideo,
    "ZipCompress": ZipCompress,
    "SendToWebhook": SendToWebhook,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadTransNetModel": "🐾MiaoshouAI Load TransNet Model",
    "TransNetV2_Run": "🐾MiaoshouAI Segment Video",
    "SelectVideo": "🐾MiaoshouAI Select Video",
    "ZipCompress": "🐾MiaoshouAI Zip Compress",
    "SendToWebhook": "🐾MiaoshouAI Send to Webhook",
}
