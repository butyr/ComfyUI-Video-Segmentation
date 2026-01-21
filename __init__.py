import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Add TransNetV2 inference-pytorch to path for transnetv2_pytorch import
transnet_path = os.path.join(current_dir, "TransNetV2", "inference-pytorch")
if os.path.exists(transnet_path):
    sys.path.insert(0, transnet_path)
else:
    raise ImportError(
        f"TransNetV2 submodule not found at {transnet_path}. "
        "Please run: git submodule update --init --recursive"
    )

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']