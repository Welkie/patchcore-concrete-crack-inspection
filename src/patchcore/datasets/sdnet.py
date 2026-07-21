import os
import torch
import PIL.Image
import numpy as np
from enum import Enum
from torchvision import transforms
from patchcore.datasets.concrete import DatasetSplit, IMAGENET_MEAN, IMAGENET_STD, generate_and_save_pseudo_mask

class SDNETDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for SDNET2018 Concrete Crack Dataset.
    Supports subdatasets: "decks", "pavements", "walls" or "all".
    """
    def __init__(
        self,
        source,
        classname="decks",
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        seed=0,
        use_pseudo_masks=True,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.resize = resize
        self.imagesize = imagesize
        self.use_pseudo_masks = use_pseudo_masks
        
        # Map classname to folders in SDNET2018
        # SDNET2018 contains folders: Decks, Pavements, Walls
        folders = []
        if classname.lower() == "decks":
            folders = ["Decks"]
        elif classname.lower() == "pavements":
            folders = ["Pavements"]
        elif classname.lower() == "walls":
            folders = ["Walls"]
        elif classname.lower() == "all":
            folders = ["Decks", "Pavements", "Walls"]
        else:
            raise ValueError(f"Unknown classname for SDNET: {classname}")
            
        neg_files = []
        pos_files = []
        
        for folder in folders:
            folder_path = os.path.join(source, folder)
            if not os.path.exists(folder_path):
                # Fallback check for casing
                for d in os.listdir(source):
                    if d.lower() == folder.lower():
                        folder_path = os.path.join(source, d)
                        break
                        
            if not os.path.exists(folder_path):
                continue
                
            # Inside Decks/Pavements/Walls:
            # CD/CP/CW are cracked (Positive)
            # UD/UP/UW are uncracked (Negative)
            for sub in os.listdir(folder_path):
                sub_path = os.path.join(folder_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                # Uncracked (Negative)
                if sub.upper() in ["UD", "UP", "UW"]:
                    files = sorted([
                        os.path.join(sub_path, f) for f in os.listdir(sub_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])
                    neg_files.extend(files)
                # Cracked (Positive)
                elif sub.upper() in ["CD", "CP", "CW"]:
                    files = sorted([
                        os.path.join(sub_path, f) for f in os.listdir(sub_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])
                    pos_files.extend(files)

        # Apply deterministic split limits (to keep size manageable)
        np.random.seed(seed)
        neg_files = sorted(neg_files)
        pos_files = sorted(pos_files)
        
        n_neg_limit = min(1000, len(neg_files))
        n_pos_limit = min(1000, len(pos_files))
        neg_files = neg_files[:n_neg_limit]
        pos_files = pos_files[:n_pos_limit]
        
        n_neg_train = int(n_neg_limit * 0.8)
        n_neg_val = int(n_neg_limit * 0.1)
        
        n_pos_val = int(n_pos_limit * 0.1)
        
        # Setup splits
        if split == DatasetSplit.TRAIN or split == "train":
            self.data = [(f, "good", 0) for f in neg_files[:n_neg_train]]
        elif split == DatasetSplit.VAL or split == "val":
            val_neg = [(f, "good", 0) for f in neg_files[n_neg_train : n_neg_train + n_neg_val]]
            val_pos = [(f, "crack", 1) for f in pos_files[:n_pos_val]]
            self.data = val_neg + val_pos
        elif split == DatasetSplit.TEST or split == "test":
            test_neg = [(f, "good", 0) for f in neg_files[n_neg_train + n_neg_val :]]
            test_pos = [(f, "crack", 1) for f in pos_files[n_pos_val:]]
            self.data = test_neg + test_pos
        else:
            raise ValueError(f"Unknown split: {split}")
            
        self.pseudo_mask_dir = os.path.abspath(os.path.join(os.getcwd(), "pseudo_masks_sdnet"))
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
        
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.imagesize = (3, imagesize, imagesize)
        
        self.data_to_iterate = []
        for img_path, anomaly_type, is_anomaly in self.data:
            if is_anomaly == 1 and self.use_pseudo_masks:
                mask_filename = os.path.basename(img_path)
                mask_path = os.path.join(self.pseudo_mask_dir, mask_filename)
                if not os.path.exists(mask_path):
                    generate_and_save_pseudo_mask(img_path, mask_path)
                self.data_to_iterate.append(["sdnet", anomaly_type, img_path, mask_path])
            else:
                self.data_to_iterate.append(["sdnet", anomaly_type, img_path, None])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, anomaly_type, is_anomaly = self.data[idx]
        image = PIL.Image.open(img_path).convert("RGB")
        transformed_img = self.transform_img(image)
        
        _, _, _, mask_path = self.data_to_iterate[idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, self.imagesize[1], self.imagesize[2]])
            
        return {
            "image": transformed_img,
            "mask": mask,
            "classname": "sdnet",
            "anomaly": anomaly_type,
            "is_anomaly": is_anomaly,
            "image_name": os.path.basename(img_path),
            "image_path": img_path,
        }
