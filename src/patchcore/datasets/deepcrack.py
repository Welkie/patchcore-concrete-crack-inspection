import os
import torch
import PIL.Image
import numpy as np
from enum import Enum
from torchvision import transforms
from patchcore.datasets.concrete import DatasetSplit, IMAGENET_MEAN, IMAGENET_STD

class DeepCrackDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for DeepCrack Dataset.
    Designed for test evaluation against ground-truth masks.
    """
    def __init__(
        self,
        source,
        classname="deepcrack",
        resize=256,
        imagesize=224,
        split=DatasetSplit.TEST,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imagesize = imagesize
        
        # Determine paths
        # DeepCrack usually has train_img, train_lab, test_img, test_lab
        # Let's support both test and train splits, but note that for Unsupervised AD,
        # training on DeepCrack directly isn't typical since all train images have cracks.
        if split == DatasetSplit.TRAIN or split == "train":
            img_dir = os.path.join(source, "train_img")
            lab_dir = os.path.join(source, "train_lab")
        else:
            img_dir = os.path.join(source, "test_img")
            lab_dir = os.path.join(source, "test_lab")
            
        if not os.path.exists(img_dir):
            # Fallback for alternative names
            for d in os.listdir(source):
                if d.lower() in [f"{split.value}_img", "test_img" if split == DatasetSplit.TEST else "train_img"]:
                    img_dir = os.path.join(source, d)
                elif d.lower() in [f"{split.value}_lab", "test_lab" if split == DatasetSplit.TEST else "train_lab"]:
                    lab_dir = os.path.join(source, d)

        if not os.path.exists(img_dir):
            raise ValueError(f"Could not find image folder {img_dir} in source {source}")
            
        img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self.data = []
        self.data_to_iterate = []
        
        for f in img_files:
            img_path = os.path.join(img_dir, f)
            # Find corresponding mask
            # Try same name with multiple extensions
            mask_path = None
            base_name = os.path.splitext(f)[0]
            if os.path.exists(lab_dir):
                for ext in ['.png', '.jpg', '.jpeg']:
                    possible_mask = os.path.join(lab_dir, base_name + ext)
                    if os.path.exists(possible_mask):
                        mask_path = possible_mask
                        break
            
            # Since this is a crack dataset, all images are anomalous/crack (except if mask is completely empty, which is rare)
            is_anomaly = 1
            anomaly_type = "crack"
            self.data.append((img_path, anomaly_type, is_anomaly))
            self.data_to_iterate.append(["deepcrack", anomaly_type, img_path, mask_path])

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
        
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.imagesize = (3, imagesize, imagesize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, anomaly_type, is_anomaly = self.data[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        transformed_img = self.transform_img(image)
        
        _, _, _, mask_path = self.data_to_iterate[idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path)
            # Convert mask to L if not already
            mask = mask.convert("L")
            mask = self.transform_mask(mask)
            # Threshold to binary 0/1
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros([1, self.imagesize[1], self.imagesize[2]])
            
        return {
            "image": transformed_img,
            "mask": mask,
            "classname": "deepcrack",
            "anomaly": anomaly_type,
            "is_anomaly": is_anomaly,
            "image_name": os.path.basename(img_path),
            "image_path": img_path,
        }
