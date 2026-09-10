"""The benchmark's DINO input contract: centered 224px RGB, ImageNet scaling."""
import numpy as np
from PIL import Image


def dino_input(image):
    # Preserve aspect ratio and never upscale. Black padding is normalized too.
    thumbnail = image.convert('RGB')
    thumbnail.thumbnail((224, 224), Image.Resampling.BICUBIC)
    width, height = thumbnail.size
    left, top = (224 - width) // 2, (224 - height) // 2
    canvas = Image.new('RGB', (224, 224))
    canvas.paste(thumbnail, (left, top))
    pixels = np.asarray(canvas, dtype=np.float32) / 255
    pixels -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
    pixels /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
    centers = np.arange(14) * 16 + 8
    horizontal = (centers >= left) & (centers < left + width)
    vertical = (centers >= top) & (centers < top + height)
    # Center-based inclusion admits partially padded edge patches as well.
    mask = (vertical[:, None] & horizontal[None, :]).astype(np.float32).reshape(196)
    if mask.sum() == 0:
        raise ValueError('image contains no DINO patch centers after resizing')
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)), mask


def pool_hidden(hidden, mask):
    if hidden.shape != (201, 1024):
        raise ValueError(f'expected DINOv3-L hidden states (201,1024), got {hidden.shape}')
    return np.stack((hidden[0], (hidden[5:] * mask[:, None]).sum(0) / mask.sum()))
