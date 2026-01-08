import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import LoveDADataset
from model import UNetResNet18
import argparse
import os
import cv2

def visualize(args):
    device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_dataset = LoveDADataset(root_dir=args.data_dir, split='Val', points_per_class=10)
    
    model = UNetResNet18(num_classes=8, pretrained=False).to(device)
    
    # Logic to find model
    model_path = args.model_path
    
    # If default name, check if user meant a specific experiment
    if model_path == 'best_model.pth' and args.exp_name:
         path_candidate = os.path.join("result", f"best_model_{args.exp_name}.pth")
         if os.path.exists(path_candidate):
             model_path = path_candidate

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        # Fallback search in result/
        print(f"Model file {model_path} not found. Searching result/...")
        candidates = [
            os.path.join("result", "best_model.pth"),
            os.path.join("result", "best_model_exp2.pth"), # Legacy
            "best_model.pth" # Legacy
        ]
        found = False
        for c in candidates:
            if os.path.exists(c):
                print(f"Found {c}, loading...")
                model.load_state_dict(torch.load(c, map_location=device))
                found = True
                break
        if not found:
             print("Error: No model found.")
             return

    model.eval()
    
    indices = np.random.choice(len(val_dataset), args.num_images, replace=False)
    
    fig, axes = plt.subplots(args.num_images, 3, figsize=(15, 5 * args.num_images))
    if args.num_images == 1:
        axes = [axes] 
        
    for i, idx in enumerate(indices):
        image, mask, _ = val_dataset[idx] 
        
        input_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
        img_disp = image.permute(1, 2, 0).numpy()
        img_disp = (img_disp * 255).astype(np.uint8)
        
        if args.num_images > 1:
            ax_img, ax_gt, ax_pred = axes[i]
        else:
            ax_img, ax_gt, ax_pred = axes[0]
            
        ax_img.imshow(img_disp)
        ax_img.set_title("Original Image")
        ax_img.axis('off')
        
        ax_gt.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=7)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis('off')
        
        ax_pred.imshow(pred, cmap='nipy_spectral', vmin=0, vmax=7)
        ax_pred.set_title("Prediction")
        ax_pred.axis('off')
        
    plt.tight_layout()
    output_file = "visualization_results.png"
    plt.savefig(output_file)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='e:/Esraa/semi_supervised/LoveDA', help='Path to LoveDA dataset')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to trained model')
    parser.add_argument('--num_images', type=int, default=3, help='Number of images to visualize')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name to load')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
    else:
        visualize(args)
