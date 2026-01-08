import os
import cv2
import numpy as np
from collections import Counter
import argparse
import matplotlib.pyplot as plt

def get_mask_dir(split, domain, root_dir='e:/Esraa/semi_supervised/LoveDA'):
    """
    Resolves the mask directory based on structure.
    Checks:
    1. root_dir/split/domain/masks_png
    2. root_dir/split/split/domain/masks_png
    """
    p1 = os.path.join(root_dir, split, domain, 'masks_png')
    if os.path.exists(p1):
        return p1
        
    p2 = os.path.join(root_dir, split, split, domain, 'masks_png')
    if os.path.exists(p2):
        return p2
        
    print(f"Warning: Could not find mask dir for {split}/{domain}. Checked:\n{p1}\n{p2}")
    return p1

def get_image_dir(split, domain, root_dir='e:/Esraa/semi_supervised/LoveDA'):
    """
    Resolves the image directory based on structure.
    """
    p1 = os.path.join(root_dir, split, domain, 'images_png')
    if os.path.exists(p1):
        return p1
        
    p2 = os.path.join(root_dir, split, split, domain, 'images_png')
    if os.path.exists(p2):
        return p2
        
    return p1

def value_hist(mask_dir, n=50, seed=0):
    if not os.path.exists(mask_dir):
        print(f"Error: Directory {mask_dir} does not exist.")
        return {}
        
    files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    if not files:
        print(f"Error: No png files found in {mask_dir}")
        return {}
        
    rng = np.random.default_rng(seed)
    pick = rng.choice(files, size=min(n, len(files)), replace=False)
    
    cnt = Counter()
    total = 0
    
    print(f"Processing {len(pick)} images from {mask_dir}...")
    
    for fn in pick:
        m = cv2.imread(os.path.join(mask_dir, fn), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
            
        vals, counts = np.unique(m, return_counts=True)
        cnt.update(dict(zip(vals.tolist(), counts.tolist())))
        total += m.size
        
    if total == 0:
        return {}
        
    return {k: v/total for k, v in sorted(cnt.items())}

def scan_unique_values(mask_dir, n=50, seed=42):
    if not os.path.exists(mask_dir):
        print(f"Error: Directory {mask_dir} does not exist.")
        return [], Counter()
        
    files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    if not files:
        return [], Counter()
        
    rng = np.random.default_rng(seed)
    pick = rng.choice(files, size=min(n, len(files)), replace=False)

    global_set = set()
    global_counter = Counter()

    for fn in pick:
        m = cv2.imread(os.path.join(mask_dir, fn), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        vals, counts = np.unique(m, return_counts=True)
        global_set.update(vals.tolist())
        global_counter.update(dict(zip(vals.tolist(), counts.tolist())))

    return sorted(list(global_set)), global_counter

def find_example_with_value(root_dir, split="Train", domain="Urban", value=7, max_checks=200):
    img_dir = get_image_dir(split, domain, root_dir)
    mask_dir = get_mask_dir(split, domain, root_dir)
    
    if not os.path.exists(mask_dir) or not os.path.exists(img_dir):
        print(f"Error: Dirs not found: \n{mask_dir}\n{img_dir}")
        return None, None, None
        
    files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    
    found = False
    for fn in files[:max_checks]:
        m = cv2.imread(os.path.join(mask_dir, fn), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        
        if (m == value).any():
            img_path = os.path.join(img_dir, fn)
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(f"Found value {value} in {fn}")
            print(f"Unique values in mask: {np.unique(m)}")
            
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1); plt.imshow(img); plt.title(f"Image: {fn}"); plt.axis("off")
            plt.subplot(1,2,2); plt.imshow(m, vmin=0, vmax=7, cmap='nipy_spectral'); plt.title(f"Mask (Contains {value})"); plt.axis("off")
            
            out_name = f"found_val_{value}_{fn}"
            plt.savefig(out_name)
            print(f"Saved plot to {out_name}")
            return fn, img, m
            
    print(f"Value {value} not found in first {max_checks} images of {split}/{domain}")
    return None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='e:/Esraa/semi_supervised/LoveDA')
    parser.add_argument('--split', type=str, default='Train')
    parser.add_argument('--domain', type=str, default='Urban')
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--value', type=int, default=7, help='Value to search for in find mode')
    parser.add_argument('--mode', type=str, default='hist', choices=['hist', 'scan', 'find'], help='Analysis mode: hist (distribution), scan (unique values), or find (visualize example)')
    args = parser.parse_args()
    
    if args.mode == 'hist':
        mask_dir = get_mask_dir(args.split, args.domain, args.root_dir)
        hist = value_hist(mask_dir, n=args.n)
        
        print(f"Class Distribution for {args.split}/{args.domain}:")
        for k, v in hist.items():
            print(f"Class {k}: {v:.4f}")
            
    elif args.mode == 'scan':
        splits = ["Train", "Val"] if args.split == 'all' else [args.split]
        domains = ["Urban", "Rural"] if args.domain == 'all' else [args.domain]
        
        for s in splits:
            for d in domains:
                mask_dir = get_mask_dir(s, d, args.root_dir)
                vals, cnt = scan_unique_values(mask_dir, n=args.n)
                print(f"{s}-{d}: unique values (sample n={args.n}) = {vals}")

    elif args.mode == 'find':
        find_example_with_value(args.root_dir, args.split, args.domain, args.value)

