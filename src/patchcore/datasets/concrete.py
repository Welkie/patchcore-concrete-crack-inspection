import os
import torch
import PIL.Image
import numpy as np
from enum import Enum
from torchvision import transforms

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def generate_and_save_pseudo_mask(img_path, save_path):
    """
    Generates a pseudo-ground-truth mask for a crack image using Canny edge detection or adaptive thresholding,
    and saves it to save_path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to load image with OpenCV")
        
        # Smooth image to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges to form solid crack lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        cv2.imwrite(save_path, dilated)
    except Exception as e:
        # Fallback to pure PIL + numpy implementation if OpenCV is not available
        try:
            img = PIL.Image.open(img_path).convert("L")
            img_np = np.array(img)
            mean = np.mean(img_np)
            std = np.std(img_np)
            # Threshold dark pixels (cracks are darker than concrete)
            thresh = mean - 1.5 * std
            mask_np = (img_np < thresh).astype(np.uint8) * 255
            PIL.Image.fromarray(mask_np).save(save_path)
        except Exception as e2:
            # Final fallback: empty black mask
            img_black = np.zeros((224, 224), dtype=np.uint8)
            PIL.Image.fromarray(img_black).save(save_path)

class ConcreteDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Concrete Surface Crack Detection (Kaggle).
    Assumes a directory structure with "Negative" and "Positive" folders.
    """

    def __init__(
        self,
        source,
        classname="concrete",
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        seed=0,
        use_pseudo_masks=True,
        train_val_split=1.0,  # to match run_patchcore.py signature
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imagesize = imagesize
        self.use_pseudo_masks = use_pseudo_masks

        # Load all Negative and Positive images, sort them for determinism
        neg_dir = None
        pos_dir = None
        for d in os.listdir(source):
            if d.lower() == "negative":
                neg_dir = os.path.join(source, d)
            elif d.lower() == "positive":
                pos_dir = os.path.join(source, d)

        if neg_dir is None or pos_dir is None:
            # Fallback to check if source itself is the directory and check subdirs
            raise ValueError(f"Could not find Negative and Positive folders in source path: {source}")

        neg_files = sorted([
            os.path.join(neg_dir, f) for f in os.listdir(neg_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        pos_files = sorted([
            os.path.join(pos_dir, f) for f in os.listdir(pos_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Limit per-class images to keep the feature bank small enough for Kaggle's RAM.
        # 1000 negatives → ~49k patch vectors @ 512-dim → ~95 MB; 2000 would double that.
        n_neg_limit = min(1000, len(neg_files))
        n_pos_limit = min(1000, len(pos_files))
        neg_files = neg_files[:n_neg_limit]
        pos_files = pos_files[:n_pos_limit]

        # Splitting ratios:
        # Negative: 80% Train, 10% Val, 10% Test
        # Positive: 10% Val, 90% Test
        n_neg_train = int(n_neg_limit * 0.8)
        n_neg_val = int(n_neg_limit * 0.1)
        n_neg_test = n_neg_limit - n_neg_train - n_neg_val

        n_pos_val = int(n_pos_limit * 0.1)
        n_pos_test = n_pos_limit - n_pos_val

        # Setup splits
        if split == DatasetSplit.TRAIN or split == "train":
            # Train contains only negative (normal) images
            self.data = [(f, "good", 0) for f in neg_files[:n_neg_train]]
        elif split == DatasetSplit.VAL or split == "val":
            # Val contains negatives + positives
            val_neg = [(f, "good", 0) for f in neg_files[n_neg_train : n_neg_train + n_neg_val]]
            val_pos = [(f, "crack", 1) for f in pos_files[:n_pos_val]]
            self.data = val_neg + val_pos
        elif split == DatasetSplit.TEST or split == "test":
            # Test contains negatives + positives
            test_neg = [(f, "good", 0) for f in neg_files[n_neg_train + n_neg_val :]]
            test_pos = [(f, "crack", 1) for f in pos_files[n_pos_val:]]
            self.data = test_neg + test_pos
        else:
            raise ValueError(f"Unknown split: {split}")

        # Directory to cache pseudo-masks on disk (use writeable current working directory)
        self.pseudo_mask_dir = os.path.abspath(os.path.join(os.getcwd(), "pseudo_masks"))
        if self.use_pseudo_masks:
            os.makedirs(self.pseudo_mask_dir, exist_ok=True)

        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ])

        # Define transform_mean and transform_std for run_patchcore.py visualization
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

        self.imagesize = (3, imagesize, imagesize)

        # Build data_to_iterate format expected by run_patchcore.py:
        # Each entry is: [classname, anomaly_type, image_path, mask_path_or_None]
        self.data_to_iterate = []
        for img_path, anomaly_type, is_anomaly in self.data:
            if is_anomaly == 1 and self.use_pseudo_masks:
                mask_filename = os.path.basename(img_path)
                mask_path = os.path.join(self.pseudo_mask_dir, mask_filename)
                # Generate mask file if it does not exist
                if not os.path.exists(mask_path):
                    generate_and_save_pseudo_mask(img_path, mask_path)
                self.data_to_iterate.append(["concrete", anomaly_type, img_path, mask_path])
            else:
                self.data_to_iterate.append(["concrete", anomaly_type, img_path, None])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, anomaly_type, is_anomaly = self.data[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        transformed_img = self.transform_img(image)

        # Get mask
        _, _, _, mask_path = self.data_to_iterate[idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, self.imagesize[1], self.imagesize[2]])

        return {
            "image": transformed_img,
            "mask": mask,
            "classname": "concrete",
            "anomaly": anomaly_type,
            "is_anomaly": is_anomaly,
            "image_name": os.path.basename(img_path),
            "image_path": img_path,
        }
