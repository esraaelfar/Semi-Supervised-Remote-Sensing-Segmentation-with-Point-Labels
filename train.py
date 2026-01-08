"""
train.py
--------
Main training script for the semi-supervised segmentation model.
Tracks experiments and saves best models based on Validation mIoU.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import LoveDADataset
from model import UNetResNet18
from loss import PartialCrossEntropyLoss
import argparse
from tqdm import tqdm
import numpy as np

def calculate_iou(pred, label, num_classes, ignore_index=0):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    
    iou_list = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        
        pred_c = (pred == c)
        label_c = (label == c)
        
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        
        if union == 0:
            iou_list.append(np.nan) 
        else:
            iou_list.append(intersection / union)
            
    return np.nanmean(iou_list)

def evaluate(model, loader, device, num_classes=8):
    model.eval()
    total_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Evaluation"):
            images = images.to(device)
            masks = masks.to(device) 
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) 
            
            batch_iou = calculate_iou(preds, masks, num_classes)
            if not np.isnan(batch_iou):
                total_iou += batch_iou
                num_batches += 1
                
    model.train()
    return total_iou / num_batches if num_batches > 0 else 0.0

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    points_per_class = args.points
    
    train_dataset = LoveDADataset(root_dir=args.data_dir, split='Train', points_per_class=points_per_class)
    
    if len(train_dataset) == 0:
        print("No training images found. Please check data directory.")
        return

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

    val_dataset = LoveDADataset(root_dir=args.data_dir, split='Val', points_per_class=points_per_class)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = UNetResNet18(num_classes=8, pretrained=True).to(device) 
    
    criterion = PartialCrossEntropyLoss(ignore_index=0, reduction='mean') 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_iou = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for i, (images, masks, point_masks) in enumerate(pbar):
            images = images.to(device)
            targets = point_masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        iou = evaluate(model, val_loader, device)
        print(f"Validation mIoU: {iou:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), "result/best_model.pth")
            print("Saved Best Model")

    print("Training finished.")
    torch.save(model.state_dict(), "result/model_final.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='e:/Esraa/semi_supervised/LoveDA', help='Path to LoveDA dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--points', type=int, default=10, help='Points per class for simulation')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
    else:
        train(args)
