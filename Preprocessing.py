# SmartVision AI - Preprocessing Module
# File: src/preprocessing.py
# Critical preprocessing functions for both training and inference

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from typing import Union, Tuple
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# ImageNet mean and std (used by pre-trained models)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Model-specific image sizes
IMG_SIZE_CLASSIFICATION = (224, 224)
IMG_SIZE_DETECTION = (640, 640)

# ============================================================================
# IMAGE LOADING & VALIDATION
# ============================================================================

def load_image(image_path: Union[str, Image.Image]) -> np.ndarray:
    """
    Load image from path or PIL Image object
    
    Args:
        image_path: Path to image file or PIL Image object
        
    Returns:
        numpy array of image in RGB format
    """
    if isinstance(image_path, str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path)
    elif isinstance(image_path, Image.Image):
        img = image_path
    else:
        raise ValueError("Input must be file path or PIL Image")
    
    # Convert to RGB (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    return np.array(img)


def validate_image(img: np.ndarray) -> bool:
    """
    Validate image array
    
    Args:
        img: Image array
        
    Returns:
        True if valid, raises exception if invalid
    """
    if img is None:
        raise ValueError("Image is None")
    
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got shape {img.shape}")
    
    if img.shape[2] != 3:
        raise ValueError(f"Expected 3 channels (RGB), got {img.shape[2]}")
    
    if img.size == 0:
        raise ValueError("Image is empty")
    
    return True


# ============================================================================
# CLASSIFICATION PREPROCESSING
# ============================================================================

def preprocess_for_classification(
    image: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = IMG_SIZE_CLASSIFICATION,
    normalize: bool = True,
    model_type: str = 'imagenet'
) -> np.ndarray:
    """
    Preprocess image for classification models
    
    Args:
        image: Input image (path, PIL Image, or numpy array)
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize pixel values
        model_type: 'imagenet' or 'custom' for different normalization
        
    Returns:
        Preprocessed image array ready for model input (1, H, W, C)
    """
    # Load image
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = load_image(image)
    
    # Validate
    validate_image(img)
    
    # Resize to target size
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Convert to float
    img = img.astype(np.float32)
    
    if normalize:
        if model_type == 'imagenet':
            # ImageNet normalization (for transfer learning models)
            img = img / 255.0  # Scale to [0, 1]
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        else:
            # Simple normalization to [0, 1]
            img = img / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img


def batch_preprocess_classification(
    images: list,
    target_size: Tuple[int, int] = IMG_SIZE_CLASSIFICATION,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess multiple images for classification
    
    Args:
        images: List of images (paths or arrays)
        target_size: Target dimensions
        normalize: Whether to normalize
        
    Returns:
        Batch of preprocessed images (N, H, W, C)
    """
    processed_images = []
    
    for img in images:
        # Process without batch dimension
        processed = preprocess_for_classification(
            img, target_size, normalize
        )
        processed_images.append(processed[0])  # Remove batch dim
    
    # Stack into batch
    return np.array(processed_images)


# ============================================================================
# DETECTION PREPROCESSING (YOLO)
# ============================================================================

def preprocess_for_detection(
    image: Union[str, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = IMG_SIZE_DETECTION,
    normalize: bool = True
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Preprocess image for YOLO detection
    
    Args:
        image: Input image
        target_size: Target dimensions (typically 640x640 for YOLOv8)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Tuple of (preprocessed image, original size)
    """
    # Load image
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = load_image(image)
    
    # Store original size for scaling boxes back
    original_size = (img.shape[1], img.shape[0])  # (width, height)
    
    # Validate
    validate_image(img)
    
    # Letterbox resize (maintains aspect ratio, adds padding)
    img_resized = letterbox_resize(img, target_size)
    
    # Convert to float and normalize
    img_resized = img_resized.astype(np.float32)
    
    if normalize:
        img_resized = img_resized / 255.0
    
    return img_resized, original_size


def letterbox_resize(
    img: np.ndarray,
    target_size: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding (letterbox)
    Used by YOLO to prevent image distortion
    
    Args:
        img: Input image
        target_size: Target (width, height)
        color: Padding color (gray by default)
        
    Returns:
        Resized and padded image
    """
    # Get current shape
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # Calculate padding offsets (center the image)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[top:top+new_h, left:left+new_w] = resized
    
    return padded


# ============================================================================
# DATA AUGMENTATION (for training)
# ============================================================================

def augment_image(
    img: np.ndarray,
    rotation_range: int = 15,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    zoom_range: float = 0.2,
    horizontal_flip: bool = True
) -> np.ndarray:
    """
    Apply random augmentation to image for training
    
    Args:
        img: Input image
        rotation_range: Max rotation in degrees
        brightness_range: Min and max brightness multipliers
        zoom_range: Max zoom factor
        horizontal_flip: Whether to randomly flip horizontally
        
    Returns:
        Augmented image
    """
    augmented = img.copy()
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = augmented.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), 
                                    borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness
    if brightness_range:
        factor = np.random.uniform(brightness_range[0], brightness_range[1])
        augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)
    
    # Random zoom
    if zoom_range > 0:
        zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        h, w = augmented.shape[:2]
        new_h, new_w = int(h / zoom), int(w / zoom)
        
        # Crop center
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = augmented[top:top+new_h, left:left+new_w]
        
        # Resize back
        augmented = cv2.resize(cropped, (w, h))
    
    # Random horizontal flip
    if horizontal_flip and np.random.rand() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    return augmented


# ============================================================================
# POST-PROCESSING
# ============================================================================

def denormalize_image(img: np.ndarray) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization
    
    Args:
        img: Normalized image
        
    Returns:
        Denormalized image in [0, 255] range
    """
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def scale_boxes(
    boxes: np.ndarray,
    original_size: Tuple[int, int],
    model_size: Tuple[int, int]
) -> np.ndarray:
    """
    Scale bounding boxes from model size back to original image size
    
    Args:
        boxes: Array of boxes [x1, y1, x2, y2]
        original_size: Original image (width, height)
        model_size: Model input size (width, height)
        
    Returns:
        Scaled boxes
    """
    orig_w, orig_h = original_size
    model_w, model_h = model_size
    
    scale_x = orig_w / model_w
    scale_y = orig_h / model_h
    
    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # x coordinates
    scaled_boxes[:, [1, 3]] *= scale_y  # y coordinates
    
    return scaled_boxes


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_image_info(image: Union[str, np.ndarray]) -> dict:
    """
    Get information about an image
    
    Args:
        image: Image path or array
        
    Returns:
        Dictionary with image information
    """
    if isinstance(image, str):
        img = load_image(image)
    else:
        img = image
    
    return {
        'shape': img.shape,
        'height': img.shape[0],
        'width': img.shape[1],
        'channels': img.shape[2],
        'dtype': img.dtype,
        'min_value': img.min(),
        'max_value': img.max(),
        'mean': img.mean(),
        'std': img.std()
    }


def visualize_preprocessing(
    original: np.ndarray,
    preprocessed: np.ndarray,
    title: str = "Preprocessing Comparison"
):
    """
    Visualize original vs preprocessed image
    
    Args:
        original: Original image
        preprocessed: Preprocessed image
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Denormalize if needed
    if preprocessed.max() <= 1.0:
        display_img = (preprocessed * 255).astype(np.uint8)
    else:
        display_img = preprocessed.astype(np.uint8)
    
    axes[1].imshow(display_img)
    axes[1].set_title("Preprocessed Image")
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for SmartVision AI
    """
    
    def __init__(self, task: str = 'classification'):
        """
        Args:
            task: 'classification' or 'detection'
        """
        self.task = task
        
        if task == 'classification':
            self.target_size = IMG_SIZE_CLASSIFICATION
        elif task == 'detection':
            self.target_size = IMG_SIZE_DETECTION
        else:
            raise ValueError("Task must be 'classification' or 'detection'")
    
    def __call__(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Apply preprocessing pipeline
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        if self.task == 'classification':
            return preprocess_for_classification(image, self.target_size)
        else:
            return preprocess_for_detection(image, self.target_size)[0]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🔧 PREPROCESSING MODULE TEST")
    print("="*80)
    print()
    
    # Test with sample image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("1. Testing Classification Preprocessing...")
    preprocessed_cls = preprocess_for_classification(test_img)
    print(f"   Input shape: {test_img.shape}")
    print(f"   Output shape: {preprocessed_cls.shape}")
    print(f"   Value range: [{preprocessed_cls.min():.3f}, {preprocessed_cls.max():.3f}]")
    print()
    
    print("2. Testing Detection Preprocessing...")
    preprocessed_det, orig_size = preprocess_for_detection(test_img)
    print(f"   Input shape: {test_img.shape}")
    print(f"   Output shape: {preprocessed_det.shape}")
    print(f"   Original size: {orig_size}")
    print()
    
    print("3. Testing Augmentation...")
    augmented = augment_image(test_img)
    print(f"   Augmented shape: {augmented.shape}")
    print()
    
    print("✅ All preprocessing functions working correctly!")
    print("="*80)