# ComfyUI Video Segmentation Node

A ComfyUI custom node for automatic video scene detection using TransNetV2.

## Description

This node detects scene boundaries in videos using the TransNetV2 deep learning model and returns timestamps for each detected scene. TransNetV2 is a state-of-the-art neural network for shot boundary detection in videos.

## Features

- Automatic scene detection using TransNetV2
- Returns timestamps (start/end times) for each detected scene
- Model weights auto-download from Hugging Face
- Configurable scene detection parameters
- Temp file cleanup to avoid memory bloat

## Installation

1. Clone this repository with submodules to your ComfyUI custom_nodes directory:
   ```bash
   cd /path/to/ComfyUI/custom_nodes
   git clone --recursive https://github.com/YOUR_USERNAME/ComfyUI-Video-Segmentation.git
   ```

   If you already cloned without `--recursive`, initialize the submodule:
   ```bash
   cd ComfyUI-Video-Segmentation
   git submodule update --init --recursive
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Restart ComfyUI

The TransNetV2 model weights will be automatically downloaded from Hugging Face on first use.

## Nodes

### 🐾MiaoshouAI Segment Video (TransNetV2_Run)

Detects scene boundaries in a video and returns timestamps.

**Inputs:**
- **video**: Video input (VIDEO type)
- **threshold**: Scene detection threshold (FLOAT, default: 0.5) - lower values detect more scene changes
- **min_scene_length**: Minimum scene length in frames (INT, default: 30)
- **device**: Processing device - auto, cpu, or cuda (default: auto)
- **TransNet_model** (optional): Pre-loaded model from Load TransNet Model node

**Outputs:**
- **timestamps**: List of (start_time, end_time) tuples in seconds
- **timestamps_string**: String representation of timestamps (one per line)

### 🐾MiaoshouAI Select Video (SelectVideo)

Selects a specific scene timestamp from the detected scenes.

**Inputs:**
- **timestamps**: List of timestamps from Segment Video node
- **index**: Index of the scene to select (INT, default: 0)

**Outputs:**
- **start_time**: Start time in seconds (FLOAT)
- **end_time**: End time in seconds (FLOAT)
- **timestamp_string**: String representation of the selected timestamp

### 🐾MiaoshouAI Load TransNet Model (DownloadAndLoadTransNetModel)

Optional node for pre-loading the TransNetV2 model. The Segment Video node loads the model automatically if not provided.

**Inputs:**
- **model**: Model selection (currently only "transnetv2-pytorch-weights")
- **device**: Processing device - auto, cpu, or cuda

**Outputs:**
- **TransNet_model**: Loaded model to pass to Segment Video node

### 🐾MiaoshouAI Zip Compress (ZipCompress)

Utility node for compressing files into a zip archive.

## Example Workflow

1. Add a video input node
2. Add "🐾MiaoshouAI Segment Video" node
3. Connect video to the Segment Video node
4. Optionally add "🐾MiaoshouAI Select Video" to extract specific scene timestamps
5. Use the timestamps in downstream nodes for video processing

## Dependencies

- ComfyUI
- Python 3.8+
- PyTorch
- NumPy
- Pillow
- OpenCV
- huggingface_hub (for model auto-download)

## Configuration

### Scene Detection Parameters

- **threshold**: Controls sensitivity of scene detection. Lower values (e.g., 0.3) detect more scene changes, higher values (e.g., 0.7) are more conservative.
- **min_scene_length**: Minimum duration of detected scenes in frames. Shorter scenes are merged with adjacent ones.

## Model Storage

Model weights are stored in: `ComfyUI/models/VLM/transnetv2-pytorch-weights/`

## References

- [TransNetV2 Paper](https://arxiv.org/abs/2008.04838)
- [TransNetV2 GitHub Repository](https://github.com/soCzech/TransNetV2)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)

## License

This project is licensed under the same terms as ComfyUI. Please refer to the original TransNetV2 license for the underlying model.
