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

# Set plotting styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

def run_patchcore_experiment(data_path, backbone_name, layers, ratio, device, batch_size=4, num_workers=2):
    print(f"\n--- Running PatchCore: Backbone={backbone_name}, Ratio={ratio:.2f} ---")

    # Free GPU memory from previous run before allocating new tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load loaders
    pin = torch.cuda.is_available()
    train_dataset = ConcreteDataset(data_path, split=DatasetSplit.TRAIN, resize=256, imagesize=224)
    test_dataset = ConcreteDataset(data_path, split=DatasetSplit.TEST, resize=256, imagesize=224)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    # 1. Load backbone
    backbone = patchcore.backbones.load(backbone_name)
    backbone.name = backbone_name

    # 2. Setup sampler
    if ratio >= 1.0 or np.isclose(ratio, 1.0):
        sampler = patchcore.sampler.IdentitySampler()
    else:
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(ratio, device)

    # 3. Initialize PatchCore
    nn_method = patchcore.common.FaissNN(on_gpu=torch.cuda.is_available(), num_workers=num_workers)
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=layers,
        device=device,
        input_shape=(3, 224, 224),
        pretrain_embed_dimension=512,   # reduced from 1024 to save memory
        target_embed_dimension=512,     # reduced from 1024 to save memory
        patchsize=3,
        featuresampler=sampler,
        anomaly_score_num_nn=1,
        nn_method=nn_method,
    )

    # 4. Training (Fitting Memory Bank)
    start_train = time.time()
    model.fit(train_loader)
    train_time = time.time() - start_train
    print(f"Memory bank fitting completed in {train_time:.2f}s")

    # Get memory bank size (number of points in FAISS index)
    try:
        mem_bank_points = model.anomaly_scorer.detection_features.shape[0]
        dim = model.anomaly_scorer.detection_features.shape[1]
        # Size in MB (float32 is 4 bytes)
        mem_bank_size_mb = (mem_bank_points * dim * 4) / (1024 * 1024)
    except Exception:
        mem_bank_points = 0
        mem_bank_size_mb = 0.0

    # 5. Inference
    start_infer = time.time()
    scores, segmentations, labels_gt, masks_gt = model.predict(test_loader)
    infer_time = time.time() - start_infer
    avg_infer_speed_ms = (infer_time / len(test_dataset)) * 1000.0
    print(f"Inference completed in {infer_time:.2f}s ({avg_infer_speed_ms:.2f} ms/image)")

    # 6. Compute metrics
    scores = np.array(scores)
    segmentations = np.array(segmentations)
    labels_gt = np.array(labels_gt)
    masks_gt = np.array(masks_gt)

    # Image AUROC
    image_auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels_gt)["auroc"]

    # Pixel AUROC (using our pseudo-ground-truth masks)
    pixel_results = patchcore.metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt)
    pixel_auroc = pixel_results["auroc"]
    optimal_threshold = pixel_results["optimal_threshold"]

    # F1-score at image level
    # Normalize scores to [0, 1] to find optimal threshold
    min_score, max_score = scores.min(), scores.max()
    norm_scores = (scores - min_score) / (max_score - min_score) if max_score > min_score else scores
    
    # We find threshold on val set (or just default to search on test set since it is standard in unsupervised papers)
    from sklearn.metrics import precision_recall_curve, f1_score
    precisions, recalls, thresholds = precision_recall_curve(labels_gt, norm_scores)
    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls, out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    best_f1 = np.max(f1_scores)

    print(f"Metrics - Image AUROC: {image_auroc:.4f} | Pixel AUROC: {pixel_auroc:.4f} | Image F1: {best_f1:.4f}")

    return {
        "backbone": backbone_name,
        "ratio": ratio,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "best_f1": best_f1,
        "train_time_sec": train_time,
        "infer_speed_ms": avg_infer_speed_ms,
        "memory_bank_points": mem_bank_points,
        "memory_bank_size_mb": mem_bank_size_mb,
        "sample_segmentations": segmentations[:3],  # return first few for visualization
        "sample_masks": masks_gt[:3],
        "sample_images": [test_dataset[i]["image_path"] for i in range(3)]
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="End-to-End Concrete Surface Crack Inspection Experiments.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to Kaggle Concrete dataset folder.")
    parser.add_argument("--results_path", type=str, default="results", help="Directory to save final tables/plots.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index.")
    parser.add_argument("--quick_run", action="store_true", help="If True, only runs a quick check on ResNet-18.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_path, exist_ok=True)
    
    # Define experiment parameters
    # Note: wideresnet50 replaced with resnet50 to keep memory footprint manageable on Kaggle.
    # ratio=1.0 (IdentitySampler) is dropped because keeping all ~78k patch features in FAISS
    # requires ~1.5 GB RAM alone; the coreset ablation still covers 1%→50%.
    backbones = ["resnet18", "resnet50"]
    ratios = [0.01, 0.05, 0.10, 0.25, 0.50]

    if args.quick_run:
        backbones = ["resnet18"]
        ratios = [0.01, 0.05]

    all_patchcore_results = []
    sample_visualizations = {}

    # Initialize CSV file with headers immediately
    csv_file = os.path.join(args.results_path, "patchcore_ablation_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Backbone", "Coreset Ratio", "Image AUROC", "Pixel AUROC", "Image F1-Score",
            "Fitting Time (s)", "Inference Speed (ms/img)", "Memory Points", "Memory Size (MB)"
        ])

    # Run PatchCore Experiments (RQ1, RQ2, RQ3)
    for backbone in backbones:
        layers = ["layer2", "layer3"]
        for ratio in ratios:
            try:
                res = run_patchcore_experiment(args.data_path, backbone, layers, ratio, device)
                all_patchcore_results.append(res)
                
                # Write this result to the CSV immediately
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        res["backbone"], f"{res['ratio']*100:.0f}%", f"{res['image_auroc']:.4f}", f"{res['pixel_auroc']:.4f}",
                        f"{res['best_f1']:.4f}", f"{res['train_time_sec']:.2f}", f"{res['infer_speed_ms']:.2f}",
                        res["memory_bank_points"], f"{res['memory_bank_size_mb']:.2f}"
                    ])
                
                # Store sample heatmaps for 10% coreset ratio to visualize later
                if np.isclose(ratio, 0.10) or (args.quick_run and np.isclose(ratio, 0.05)):
                    sample_visualizations[backbone] = {
                        "imgs": res["sample_images"],
                        "masks": res["sample_masks"],
                        "segs": res["sample_segmentations"]
                    }
            except Exception as e:
                print(f"Error running experiment {backbone} at ratio {ratio}: {e}")
                # Always clean up GPU memory after any failure
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Run Supervised Baselines (RQ4)
    # Import and call train_supervised functions directly to avoid subprocess path issues on Kaggle
    from train_supervised import build_supervised_datasets, train_model, evaluate_model
    import csv as csv_module

    num_positives_list = [5, 10, 50, 100, 200]
    supervised_models = ["resnet50", "efficientnet_b0"]
    
    if args.quick_run:
        num_positives_list = [5]
        supervised_models = ["resnet50"]

    sup_save_path = os.path.join(args.results_path, "supervised")
    os.makedirs(sup_save_path, exist_ok=True)
    sup_result_file = os.path.join(sup_save_path, "results.csv")
    
    # Write header
    with open(sup_result_file, "w", newline="") as f:
        writer = csv_module.writer(f)
        writer.writerow(["Model", "Num Positives", "Epochs", "Batch Size", "LR", "Accuracy", "AUROC", "F1", "Precision", "Recall"])

    print("\n--- Running Supervised Baseline Training (Data Efficiency Analysis) ---")
    for model_name in supervised_models:
        for num_pos in num_positives_list:
            try:
                print(f"\n--- Supervised: {model_name}, Num Positives={num_pos} ---")
                epochs = 5 if args.quick_run else 10
                batch_size = 16
                lr = 1e-4

                # Free memory before each supervised experiment
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                train_dataset, val_dataset, test_dataset = build_supervised_datasets(
                    args.data_path, num_pos
                )
                print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

                pin = torch.cuda.is_available()
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

                model = train_model(model_name, train_loader, None, device, epochs=epochs, lr=lr)
                metrics = evaluate_model(model, test_loader, device)

                print(f"  Results: Acc={metrics['accuracy']:.4f}, AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")

                # Append to CSV
                with open(sup_result_file, "a", newline="") as f:
                    writer = csv_module.writer(f)
                    writer.writerow([
                        model_name, num_pos, epochs, batch_size, lr,
                        f"{metrics['accuracy']:.4f}", f"{metrics['auroc']:.4f}", f"{metrics['f1']:.4f}",
                        f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}"
                    ])
            except Exception as e:
                print(f"  Error: {e}")

    # Plot results
    generate_plots(args.results_path, all_patchcore_results, sample_visualizations)

def generate_plots(results_path, patchcore_results, sample_visualizations):
    # Plot 1: Image AUROC vs. Coreset Ratio
    plt.figure(figsize=(8, 5))
    for backbone in set(r["backbone"] for r in patchcore_results):
        sub = [r for r in patchcore_results if r["backbone"] == backbone]
        # Sort by ratio
        sub = sorted(sub, key=lambda x: x["ratio"])
        ratios_pct = [r["ratio"] * 100 for r in sub]
        aurocs = [r["image_auroc"] for r in sub]
        plt.plot(ratios_pct, aurocs, marker='o', linewidth=2, label=f"PatchCore ({backbone})")
    
    plt.xlabel("Coreset Subsampling Ratio (%)")
    plt.ylabel("Image-Level AUROC")
    plt.title("Image-Level AUROC vs. Coreset Subsampling Ratio")
    plt.ylim(0.80, 1.01)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "plot_auroc_vs_coreset.png"), dpi=300)
    plt.close()

    # Plot 2: Inference Speed vs. Coreset Ratio
    plt.figure(figsize=(8, 5))
    for backbone in set(r["backbone"] for r in patchcore_results):
        sub = [r for r in patchcore_results if r["backbone"] == backbone]
        sub = sorted(sub, key=lambda x: x["ratio"])
        ratios_pct = [r["ratio"] * 100 for r in sub]
        speeds = [r["infer_speed_ms"] for r in sub]
        plt.plot(ratios_pct, speeds, marker='s', linestyle='--', linewidth=2, label=f"{backbone}")
    
    plt.xlabel("Coreset Subsampling Ratio (%)")
    plt.ylabel("Inference Speed (ms / image)")
    plt.title("Inference Latency vs. Coreset Subsampling Ratio")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "plot_latency_vs_coreset.png"), dpi=300)
    plt.close()

    # Plot 3: Data Efficiency Curve (RQ4)
    # Read supervised results from results_path/supervised/results.csv
    # CSV format: Model, Num Positives, Epochs, Batch Size, LR, Accuracy, AUROC, F1, Precision, Recall
    supervised_file = os.path.join(results_path, "supervised", "results.csv")
    if os.path.exists(supervised_file):
        plt.figure(figsize=(9, 6))
        
        # Load supervised data
        sup_data = {}
        with open(supervised_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Find AUROC column index dynamically
            auroc_idx = header.index("AUROC") if "AUROC" in header else 6
            model_idx = header.index("Model") if "Model" in header else 0
            numpos_idx = header.index("Num Positives") if "Num Positives" in header else 1
            for row in reader:
                if len(row) < auroc_idx + 1:
                    continue
                model = row[model_idx]
                num_pos = int(row[numpos_idx])
                auroc = float(row[auroc_idx])
                if model not in sup_data:
                    sup_data[model] = []
                sup_data[model].append((num_pos, auroc))
        
        # Plot supervised curves
        for model, pts in sup_data.items():
            pts = sorted(pts, key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, marker='v', linewidth=2.5, label=f"Supervised {model}")

        # Add PatchCore (0 training positives) horizontal lines for comparison
        # Fixed color map keyed by name — avoids non-deterministic set() ordering
        backbone_colors = {"resnet18": "red", "resnet50": "green"}
        # Sort so resnet18 is always drawn first (below resnet50)
        sorted_backbones = sorted(set(r["backbone"] for r in patchcore_results))
        for backbone in sorted_backbones:
            color = backbone_colors.get(backbone, "purple")
            # Use 10% coreset for comparison (fallback to 1% if unavailable)
            p_res = [r for r in patchcore_results if r["backbone"] == backbone and np.isclose(r["ratio"], 0.10)]
            if not p_res:
                p_res = [r for r in patchcore_results if r["backbone"] == backbone and np.isclose(r["ratio"], 0.01)]
            if p_res:
                auroc_val = p_res[0]["image_auroc"]
                # Different linestyle + linewidth so both lines stay visible when AUROC values are nearly equal
                lw = 2.5 if backbone == "resnet18" else 1.5
                ls = "--" if backbone == "resnet18" else "-."
                plt.axhline(y=auroc_val, color=color, linestyle=ls, linewidth=lw,
                            label=f"PatchCore ({backbone}, 0 defects)")
                # Annotate exact value on the right so both lines are distinguishable
                plt.annotate(
                    f"{auroc_val:.4f}",
                    xy=(1.01, auroc_val),
                    xycoords=("axes fraction", "data"),
                    fontsize=9,
                    color=color,
                    va="center"
                )

        plt.xlabel("Number of Labeled Defect (Positive) Samples in Training")
        plt.ylabel("Test Image AUROC")
        plt.title("Data Efficiency Comparison: Unsupervised vs. Supervised")
        plt.xscale('log')
        # set nice ticks
        plt.xticks([5, 10, 50, 100, 200], ['5', '10', '50', '100', '200'])
        plt.legend(loc="lower right")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, "plot_data_efficiency.png"), dpi=300)
        plt.close()

    # Plot 4: Qualitative visual heatmaps (RQ2)
    # We display input image, pseudo-mask, and anomaly heatmap side-by-side
    for backbone, data in sample_visualizations.items():
        fig, axes = plt.subplots(3, 3, figsize=(10, 9))
        for idx in range(3):
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
                
            # Display Pseudo-GT Mask
            # Squeeze channel dim if mask has shape (1, H, W) — matplotlib expects (H, W)
            mask_2d = np.squeeze(mask) if isinstance(mask, np.ndarray) else mask
            if hasattr(mask_2d, 'ndim') and mask_2d.ndim == 3 and mask_2d.shape[0] == 1:
                mask_2d = mask_2d[0]
            axes[idx, 1].imshow(mask_2d, cmap='gray')
            axes[idx, 1].axis('off')
            if idx == 0:
                axes[idx, 1].set_title("Pseudo-GT Mask")
                
            # Display Anomaly Heatmap
            # Overlay heatmap on grayscale image for premium look
            # Squeeze channel dim if seg has shape (1, H, W)
            seg_2d = np.squeeze(seg) if isinstance(seg, np.ndarray) else seg
            if hasattr(seg_2d, 'ndim') and seg_2d.ndim == 3 and seg_2d.shape[0] == 1:
                seg_2d = seg_2d[0]
            gray_img = np.array(img.convert("L"))
            axes[idx, 2].imshow(gray_img, cmap='gray')
            im = axes[idx, 2].imshow(seg_2d, cmap='jet', alpha=0.5)
            axes[idx, 2].axis('off')
            if idx == 0:
                axes[idx, 2].set_title("Anomaly Heatmap")
                
        plt.tight_layout()
        plt.savefig(os.path.join(results_path, f"visual_localization_{backbone}.png"), dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
