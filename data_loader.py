"""
data_loader.py
--------------
Implements the LoveDADataset class for loading Remote Sensing images and masks.
Includes functionality to simulate point-level supervision from dense masks.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class LoveDADataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, points_per_class=10):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): 'Train' or 'Val'.
            transform (callable, optional): Optional transform to be applied on a sample.
            points_per_class (int): Number of points to sample per class for partial supervision.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.points_per_class = points_per_class
        
        self.image_dir = os.path.join(root_dir, split, 'images_png')
        self.mask_dir = os.path.join(root_dir, split, 'masks_png')
        
        self.images = []
        self.image_paths = []
        self.mask_paths = []

        domains = ['Urban', 'Rural']
        
        def add_files_from_dir(img_d, msk_d):
            if os.path.exists(img_d) and os.path.exists(msk_d):
                files = sorted([f for f in os.listdir(img_d) if f.endswith('.png')])
                for f in files:
                    self.images.append(f)
                    self.image_paths.append(os.path.join(img_d, f))
                    self.mask_paths.append(os.path.join(msk_d, f))
                return True
            return False

        if add_files_from_dir(self.image_dir, self.mask_dir):
            print(f"Found data in simple structure for {split}")
        
        found_nested = False
        for domain in domains:
            img_d = os.path.join(root_dir, split, domain, 'images_png')
            msk_d = os.path.join(root_dir, split, domain, 'masks_png')
            if add_files_from_dir(img_d, msk_d):
                found_nested = True
        
        if not found_nested:
             for domain in domains:
                img_d = os.path.join(root_dir, split, split, domain, 'images_png')
                msk_d = os.path.join(root_dir, split, split, domain, 'masks_png')
                if add_files_from_dir(img_d, msk_d):
                    found_nested = True

        if len(self.images) == 0:
            print(f"Warning: No images found for split {split} in {root_dir}. Checked paths including Urban/Rural subdirs.")


        # Class definition (LoveDA)
        # 0: Ignore/No-Data
        # 1: Background
        # 2: Building
        # 3: Road
        # 4: Water
        # 5: Barren
        # 6: Forest
        # 7: Agriculture
        self.ignore_index = 0

    def __len__(self):
        return len(self.images)

    def simulate_points(self, mask):
        """
        Randomly samples 'points_per_class' pixels for each class present in the mask.
        Returns a sparse mask where unlabeled pixels are 0 (or a specific ignore value).
        
        Actually, for Partial CE, we usually need:
        - A mask indicating WHICH pixels are labeled.
        - The labels for those pixels.
        
        Here, we will return a 'point_mask' which has specific class values at selected points
        and 'ignore_index' (0) elsewhere.
        """
        point_mask = np.zeros_like(mask)
        
        unique_classes = np.unique(mask)
        
        for cls in unique_classes:
            if cls == self.ignore_index:
                continue
            
            coords = np.argwhere(mask == cls)
            
            if len(coords) > 0:
                replace = len(coords) < self.points_per_class
                indices = np.random.choice(len(coords), size=self.points_per_class, replace=replace)
                selected_coords = coords[indices]
                
                for coord in selected_coords:
                    point_mask[coord[0], coord[1]] = cls
                    
        return point_mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        point_mask = self.simulate_points(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask, point_mask=point_mask)
            image = augmented['image']
            mask = augmented['mask']
            point_mask = augmented['point_mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
            point_mask = torch.from_numpy(point_mask).long()
            
        return image, mask, point_mask
