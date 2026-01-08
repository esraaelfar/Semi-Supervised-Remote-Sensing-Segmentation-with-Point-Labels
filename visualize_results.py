import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import LoveDADataset
from model import UNetResNet18
import argparse
import os
import cv2

def visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_dataset = LoveDADataset(root_dir=args.data_dir, split='Val', points_per_class=10)
    
    model = UNetResNet18(num_classes=8, pretrained=False).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Error: Model file {args.model_path} not found.")
        if os.path.exists("result/best_model_exp2.pth"):
             print("Found best_model_exp2.pth, loading that instead.")
             model.load_state_dict(torch.load("result/best_model_exp2.pth", map_location=device))
        elif os.path.exists("result/best_model.pth"):
             print("Found best_model.pth, loading that instead.")
             model.load_state_dict(torch.load("result/best_model.pth", map_location=device))
        else:
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
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
    else:
        visualize(args)
