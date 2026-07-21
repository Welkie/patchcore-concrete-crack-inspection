import logging
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore.backbones
import patchcore.common
from patchcore.patchcore import PatchMaker

LOGGER = logging.getLogger(__name__)

class PaDiM(torch.nn.Module):
    def __init__(self, device):
        """PaDiM anomaly detection class."""
        super(PaDiM, self).__init__()
        self.device = device
        self.idx = None
        self.means = None
        self.inv_covs = None

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension=100,  # D parameter in PaDiM
        patchsize=3,
        patchstride=1,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        self.forward_modules["feature_aggregator"] = feature_aggregator
        self.d_dimension = pretrain_embed_dimension  # default 100 random dimensions

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

    def _embed(self, images):
        """Extracts and pools features from backbone layer outputs."""
        self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
            
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # Spatial average pooling to get feature vector per patch center
        pooled = []
        for feat in features:
            # feat shape: (batch_size * num_patches, channels, patch_h, patch_w)
            # pool to (batch_size * num_patches, channels)
            f_pooled = feat.mean(dim=[-2, -1])
            pooled.append(f_pooled)
            
        # Concatenate across layers
        concated = torch.cat(pooled, dim=-1)
        
        batchsize = images.shape[0]
        num_patches = ref_num_patches[0] * ref_num_patches[1]
        concated = concated.reshape(batchsize, num_patches, -1)
        
        return concated, ref_num_patches

    def fit(self, training_data):
        """Fit multivariate Gaussian distributions per patch position."""
        self.forward_modules.eval()
        
        all_features = []
        ref_num_patches = None
        
        with torch.no_grad():
            for image in tqdm.tqdm(training_data, desc="PaDiM: Extracting training features...", position=1, leave=False):
                if isinstance(image, dict):
                    image = image["image"]
                image = image.to(torch.float).to(self.device)
                feats, ref_num_patches = self._embed(image)
                all_features.append(feats.cpu())
                
        all_features = torch.cat(all_features, dim=0) # shape: (N_total, num_patches, C)
        N, num_patches, C = all_features.shape
        
        # Randomly select a subset of channel indices D
        if self.idx is None:
            # For reproducibility, generate deterministically on CPU
            g = torch.Generator()
            g.manual_seed(42)
            self.idx = torch.randperm(C, generator=g)[:self.d_dimension]
            
        # Restrict channels
        all_features = all_features[:, :, self.idx]
        
        # Compute mean per patch position: shape (num_patches, D)
        self.means = all_features.mean(dim=0)
        
        # Compute inverse covariance matrix per patch position: shape (num_patches, D, D)
        self.inv_covs = torch.zeros(num_patches, self.d_dimension, self.d_dimension)
        identity = torch.eye(self.d_dimension)
        
        for i in range(num_patches):
            patch_feats = all_features[:, i, :] # (N, D)
            diff = patch_feats - self.means[i] # (N, D)
            cov = (diff.T @ diff) / (N - 1) + 0.01 * identity
            self.inv_covs[i] = torch.linalg.inv(cov)

    def predict(self, dataloader):
        """Predict anomaly scores and localization heatmaps."""
        self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        
        means_dev = self.means.to(self.device)
        inv_covs_dev = self.inv_covs.to(self.device)
        
        with torch.no_grad():
            for image in tqdm.tqdm(dataloader, desc="PaDiM: Predicting...", leave=False):
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                    
                image = image.to(torch.float).to(self.device)
                batchsize = image.shape[0]
                
                feats, ref_num_patches = self._embed(image)
                feats = feats[:, :, self.idx] # (batchsize, num_patches, D)
                
                num_patches = ref_num_patches[0] * ref_num_patches[1]
                dist = torch.zeros(batchsize, num_patches, device=self.device)
                
                for i in range(num_patches):
                    x = feats[:, i, :] # (batchsize, D)
                    mu = means_dev[i] # (D)
                    inv_cov = inv_covs_dev[i] # (D, D)
                    
                    diff = x - mu # (batchsize, D)
                    mult = diff @ inv_cov # (batchsize, D)
                    dist_sq = torch.sum(mult * diff, dim=-1) # (batchsize)
                    dist[:, i] = torch.sqrt(torch.clamp(dist_sq, min=0.0))
                    
                patch_scores = dist.reshape(batchsize, ref_num_patches[0], ref_num_patches[1])
                image_scores = patch_scores.reshape(batchsize, -1).max(dim=-1).values.cpu().numpy().tolist()
                
                masks_batch = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
                
                scores.extend(image_scores)
                masks.extend([m for m in masks_batch])
                
        return scores, masks, labels_gt, masks_gt

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "d_dimension": self.d_dimension,
            "idx": self.idx,
            "means": self.means,
            "inv_covs": self.inv_covs,
        }
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, prepend + "padim_params.pkl"), "wb") as f:
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    def load_from_path(self, load_path: str, device: torch.device, prepend: str = "") -> None:
        with open(os.path.join(load_path, prepend + "padim_params.pkl"), "rb") as f:
            params = pickle.load(f)
        backbone = patchcore.backbones.load(params["backbone.name"])
        backbone.name = params["backbone.name"]
        self.load(
            backbone=backbone,
            layers_to_extract_from=params["layers_to_extract_from"],
            device=device,
            input_shape=params["input_shape"],
            pretrain_embed_dimension=params["d_dimension"],
        )
        self.idx = params["idx"]
        self.means = params["means"]
        self.inv_covs = params["inv_covs"]
