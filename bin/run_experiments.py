import os
import sys
import time
import csv
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.Image
from torch.utils.data import DataLoader

# Add src/ to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from patchcore.datasets.concrete import ConcreteDataset, DatasetSplit
from patchcore.datasets.sdnet import SDNETDataset
from patchcore.datasets.deepcrack import DeepCrackDataset
from patchcore.padim import PaDiM

# Set plotting styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def run_unsupervised_experiment(dataset_name, method_name, backbone_name, ratio, train_loader, test_loader, device, num_workers=2):
    print(f"\n>>> Running: Dataset={dataset_name} | Method={method_name} | Backbone={backbone_name} (Coreset Ratio={ratio*100:.0f}%) <<<")
    
    # Free GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_train = time.time()
    
    if method_name.lower() == "padim":
        # PaDiM Initialization
        backbone = patchcore.backbones.load(backbone_name)
        backbone.name = backbone_name
        model = PaDiM(device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=device,
            input_shape=(3, 224, 224),
            pretrain_embed_dimension=100,  # D parameter for PaDiM
            patchsize=3,
        )
    else:
        # SPADE / PatchCore Initialization
        backbone = patchcore.backbones.load(backbone_name)
        backbone.name = backbone_name
        
        # Sampler
        if ratio >= 1.0 or np.isclose(ratio, 1.0):
            sampler = patchcore.sampler.IdentitySampler()
        else:
            sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(ratio, device)
            
        nn_method = patchcore.common.FaissNN(on_gpu=torch.cuda.is_available(), num_workers=num_workers)
        model = patchcore.patchcore.PatchCore(device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=device,
            input_shape=(3, 224, 224),
            pretrain_embed_dimension=512,
            target_embed_dimension=512,
            patchsize=3,
            featuresampler=sampler,
            anomaly_score_num_nn=1,
            nn_method=nn_method,
        )

    # 1. Training (Fitting)
    model.fit(train_loader)
    train_time = time.time() - start_train
    print(f"  Fitting completed in {train_time:.2f}s")

    # Get memory/size metrics
    mem_points = 0
    mem_size_mb = 0.0
    if method_name.lower() == "padim":
        if model.means is not None:
            # PaDiM stores means (num_patches, D) and inv_covs (num_patches, D, D)
            mem_points = model.means.shape[0]
            d = model.means.shape[1]
            # Means + Covs size in MB
            mem_size_mb = (mem_points * d * 4 + mem_points * d * d * 4) / (1024 * 1024)
    else:
        try:
            mem_points = model.anomaly_scorer.detection_features.shape[0]
            dim = model.anomaly_scorer.detection_features.shape[1]
            mem_size_mb = (mem_points * dim * 4) / (1024 * 1024)
        except Exception:
            pass

    # 2. Inference
    start_infer = time.time()
    scores, segmentations, labels_gt, masks_gt = model.predict(test_loader)
    infer_time = time.time() - start_infer
    
    test_size = len(test_loader.dataset)
    avg_infer_speed_ms = (infer_time / test_size) * 1000.0 if test_size > 0 else 0.0
    print(f"  Inference completed in {infer_time:.2f}s ({avg_infer_speed_ms:.2f} ms/image)")

    # 3. Compute Metrics
    scores = np.array(scores)
    segmentations = np.array(segmentations)
    labels_gt = np.array(labels_gt)
    masks_gt = np.array(masks_gt)

    image_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]
    pixel_results = patchcore.metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt)
    pixel_auroc = pixel_results["auroc"]

    # F1-score at image level
    min_score, max_score = scores.min(), scores.max()
    norm_scores = (scores - min_score) / (max_score - min_score) if max_score > min_score else scores
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(labels_gt, norm_scores)
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    best_f1 = np.max(f1_scores)

    print(f"  Metrics - Image AUROC: {image_auroc:.4f} | Pixel AUROC: {pixel_auroc:.4f} | Image F1: {best_f1:.4f}")

    # Sieve up to 3 positive images for visualization
    anomaly_indices = np.where(labels_gt == 1)[0]
    vis_indices = anomaly_indices[:3] if len(anomaly_indices) >= 3 else np.arange(min(3, len(labels_gt)))
    
    return {
        "dataset": dataset_name,
        "method": method_name,
        "backbone": backbone_name,
        "ratio": ratio,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "best_f1": best_f1,
        "train_time_sec": train_time,
        "infer_speed_ms": avg_infer_speed_ms,
        "memory_points": mem_points,
        "memory_size_mb": mem_size_mb,
        "sample_segmentations": [segmentations[i] for i in vis_indices],
        "sample_masks": [masks_gt[i] for i in vis_indices],
        "sample_images": [test_loader.dataset[i]["image_path"] for i in vis_indices]
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unsupervised Crack Anomaly Detection - Hướng A.1 Experiments.")
    parser.add_argument("--surface_crack_path", type=str, required=True, help="Path to Mendeley Surface Crack Detection folder.")
    parser.add_argument("--sdnet_path", type=str, required=True, help="Path to SDNET2018 dataset folder.")
    parser.add_argument("--deepcrack_path", type=str, required=True, help="Path to DeepCrack dataset folder.")
    parser.add_argument("--results_path", type=str, default="results", help="Directory to save final tables/plots.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--quick_run", action="store_true", help="If True, runs a quick verification test.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_path, exist_ok=True)
    pin = torch.cuda.is_available()

    # Define the 3 unsupervised methods
    # SPADE: PatchCore with 100% coreset (IdentitySampler)
    # PaDiM: Gaussian Patch model
    # PatchCore: PatchCore with 10% coreset (ApproximateGreedyCoresetSampler)
    experiments_list = [
        {"method": "SPADE", "backbone": "resnet18", "ratio": 1.0},
        {"method": "PaDiM", "backbone": "resnet18", "ratio": 0.0},
        {"method": "PatchCore", "backbone": "resnet18", "ratio": 0.10},
        {"method": "PatchCore", "backbone": "resnet50", "ratio": 0.10},
    ]

    if args.quick_run:
        experiments_list = [
            {"method": "PaDiM", "backbone": "resnet18", "ratio": 0.0},
            {"method": "PatchCore", "backbone": "resnet18", "ratio": 0.10},
        ]

    # Initialize datasets and loaders
    print("\n--- Initializing Datasets and Loaders ---")
    
    # 1. Surface Crack Detection (Mendeley)
    sc_train = ConcreteDataset(args.surface_crack_path, split=DatasetSplit.TRAIN)
    sc_test = ConcreteDataset(args.surface_crack_path, split=DatasetSplit.TEST)
    sc_train_loader = DataLoader(sc_train, batch_size=4, shuffle=False, num_workers=2, pin_memory=pin)
    sc_test_loader = DataLoader(sc_test, batch_size=4, shuffle=False, num_workers=2, pin_memory=pin)
    print(f"Surface Crack: Train samples={len(sc_train)}, Test samples={len(sc_test)}")

    # 2. SDNET2018 (Decks subset)
    sd_train = SDNETDataset(args.sdnet_path, classname="decks", split=DatasetSplit.TRAIN)
    sd_test = SDNETDataset(args.sdnet_path, classname="decks", split=DatasetSplit.TEST)
    sd_train_loader = DataLoader(sd_train, batch_size=4, shuffle=False, num_workers=2, pin_memory=pin)
    sd_test_loader = DataLoader(sd_test, batch_size=4, shuffle=False, num_workers=2, pin_memory=pin)
    print(f"SDNET2018 (Decks): Train samples={len(sd_train)}, Test samples={len(sd_test)}")

    # 3. DeepCrack (Cross-dataset evaluation)
    # Train on Surface Crack normal images, test on DeepCrack
    dc_test = DeepCrackDataset(args.deepcrack_path, normal_source=args.surface_crack_path, split=DatasetSplit.TEST)
    dc_test_loader = DataLoader(dc_test, batch_size=4, shuffle=False, num_workers=2, pin_memory=pin)
    print(f"DeepCrack (Cross-dataset): Train samples={len(sc_train)} (using Surface Crack normal), Test samples={len(dc_test)}")

    all_results = []
    sample_visualizations = {}

    csv_file = os.path.join(args.results_path, "unsupervised_comparison_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Method", "Backbone", "Coreset Ratio", "Image AUROC", "Pixel AUROC", "Image F1-Score",
            "Fitting Time (s)", "Inference Speed (ms/img)", "Memory Points", "Memory Size (MB)"
        ])

    datasets = [
        {"name": "Surface Crack", "train_loader": sc_train_loader, "test_loader": sc_test_loader},
        {"name": "SDNET2018 (Decks)", "train_loader": sd_train_loader, "test_loader": sd_test_loader},
        {"name": "DeepCrack (Cross-eval)", "train_loader": sc_train_loader, "test_loader": dc_test_loader},
    ]

    if args.quick_run:
        datasets = [datasets[0]]  # only run Surface Crack for quick test

    for ds in datasets:
        for exp in experiments_list:
            try:
                res = run_unsupervised_experiment(
                    dataset_name=ds["name"],
                    method_name=exp["method"],
                    backbone_name=exp["backbone"],
                    ratio=exp["ratio"],
                    train_loader=ds["train_loader"],
                    test_loader=ds["test_loader"],
                    device=device,
                )
                all_results.append(res)
                
                # Save to CSV
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    ratio_str = f"{res['ratio']*100:.0f}%" if res["method"] != "PaDiM" else "N/A"
                    writer.writerow([
                        res["dataset"], res["method"], res["backbone"], ratio_str,
                        f"{res['image_auroc']:.4f}", f"{res['pixel_auroc']:.4f}", f"{res['best_f1']:.4f}",
                        f"{res['train_time_sec']:.2f}", f"{res['infer_speed_ms']:.2f}",
                        res["memory_points"], f"{res['memory_size_mb']:.2f}"
                    ])

                # Save sample visualization for PatchCore ResNet18
                if res["method"] == "PatchCore" and res["backbone"] == "resnet18":
                    sample_visualizations[res["dataset"]] = {
                        "imgs": res["sample_images"],
                        "masks": res["sample_masks"],
                        "segs": res["sample_segmentations"]
                    }
                    
            except Exception as e:
                print(f"Error running {exp['method']} on {ds['name']}: {e}")
                # Clean memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Generate comparative plots
    generate_plots(args.results_path, all_results, sample_visualizations)
    print("\n>>> All experiments completed! Results saved to results/ folder. <<<")

def generate_plots(results_path, all_results, sample_visualizations):
    if not all_results:
        return
        
    # Plot 1: Image AUROC comparison across datasets
    plt.figure(figsize=(10, 6))
    datasets = sorted(list(set(r["dataset"] for r in all_results)))
    methods = [f"{r['method']} ({r['backbone']})" for r in all_results if r["dataset"] == datasets[0]]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, m_info in enumerate(set((r["method"], r["backbone"]) for r in all_results)):
        m_name, bb = m_info
        aurocs = []
        for ds in datasets:
            match = [r for r in all_results if r["dataset"] == ds and r["method"] == m_name and r["backbone"] == bb]
            aurocs.append(match[0]["image_auroc"] if match else 0.0)
            
        plt.bar(x + (i - 1.5) * width, aurocs, width, label=f"{m_name} ({bb})")

    plt.xlabel("Datasets")
    plt.ylabel("Image-level AUROC")
    plt.title("Image-level Detection Performance (AUROC) comparison")
    plt.xticks(x, datasets)
    plt.ylim(0.70, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "plot_unsupervised_auroc_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: Pixel AUROC comparison (localization) across datasets
    plt.figure(figsize=(10, 6))
    for i, m_info in enumerate(set((r["method"], r["backbone"]) for r in all_results)):
        m_name, bb = m_info
        aurocs = []
        for ds in datasets:
            match = [r for r in all_results if r["dataset"] == ds and r["method"] == m_name and r["backbone"] == bb]
            aurocs.append(match[0]["pixel_auroc"] if match else 0.0)
            
        plt.bar(x + (i - 1.5) * width, aurocs, width, label=f"{m_name} ({bb})")

    plt.xlabel("Datasets")
    plt.ylabel("Pixel-level (Localization) AUROC")
    plt.title("Pixel-level Localization Performance (AUROC) comparison")
    plt.xticks(x, datasets)
    plt.ylim(0.60, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "plot_unsupervised_localization_comparison.png"), dpi=300)
    plt.close()

    # Plot 3: Inference Latency Comparison
    plt.figure(figsize=(9, 5))
    unique_methods = []
    avg_latencies = []
    for m_info in sorted(list(set((r["method"], r["backbone"]) for r in all_results))):
        m_name, bb = m_info
        match = [r["infer_speed_ms"] for r in all_results if r["method"] == m_name and r["backbone"] == bb]
        if match:
            unique_methods.append(f"{m_name}\n({bb})")
            avg_latencies.append(np.mean(match))
            
    plt.bar(unique_methods, avg_latencies, color=["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"][:len(unique_methods)])
    plt.ylabel("Inference Latency (ms / image)")
    plt.title("Average Inference Latency Comparison (Lower is Better)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "plot_unsupervised_latency_comparison.png"), dpi=300)
    plt.close()

    # Plot 4: Qualitative visualizations
    # We display input image, pseudo-mask, and anomaly heatmap side-by-side
    for ds_name, data in sample_visualizations.items():
        fig, axes = plt.subplots(3, 3, figsize=(10, 9))
        for idx in range(min(3, len(data["imgs"]))):
            img_path = data["imgs"][idx]
            mask = data["masks"][idx]
            seg = data["segs"][idx]
            
            # Read image
            img = PIL.Image.open(img_path).convert("RGB")
            
            # Display Original
            axes[idx, 0].imshow(img)
            axes[idx, 0].axis('off')
            if idx == 0:
                axes[idx, 0].set_title("Input Image")
                
            # Display Mask
            mask_2d = np.squeeze(mask)
            if hasattr(mask_2d, 'ndim') and mask_2d.ndim == 3 and mask_2d.shape[0] == 1:
                mask_2d = mask_2d[0]
            axes[idx, 1].imshow(mask_2d, cmap='gray')
            axes[idx, 1].axis('off')
            if idx == 0:
                axes[idx, 1].set_title("Ground-Truth Mask")
                
            # Display Anomaly Heatmap
            seg_2d = np.squeeze(seg)
            if hasattr(seg_2d, 'ndim') and seg_2d.ndim == 3 and seg_2d.shape[0] == 1:
                seg_2d = seg_2d[0]
            gray_img = np.array(img.convert("L"))
            axes[idx, 2].imshow(gray_img, cmap='gray')
            axes[idx, 2].imshow(seg_2d, cmap='jet', alpha=0.5)
            axes[idx, 2].axis('off')
            if idx == 0:
                axes[idx, 2].set_title("Anomaly Heatmap")
                
        plt.suptitle(f"Localization Overlays - {ds_name}", fontsize=16)
        plt.tight_layout()
        safe_name = ds_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(results_path, f"visual_localization_{safe_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
