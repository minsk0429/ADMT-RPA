import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


# # # # # # # # # # # # # # # # # # # # # 
# # 0. random box
# # # # # # # # # # # # # # # # # # # # # 
def rand_bbox(size, lam=None):
    # past implementation
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)


    return bbx1, bby1, bbx2, bby2


# # # # # # # # # # # # # # # # # # # # # 
# # 1. cutmix for 2d
# # # # # # # # # # # # # # # # # # # # # 
# def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits):
#     mix_unlabeled_image = unlabeled_image.clone()
#     mix_unlabeled_target = unlabeled_mask.clone()
#     mix_unlabeled_logits = unlabeled_logits.clone()
    
#     # get the random mixing objects
#     u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
#     # print(u_rand_index)
    
#     # get box
#     u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
#     # cut & paste
#     for i in range(0, mix_unlabeled_image.shape[0]):
#         mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         # label is of 3 dimensions
# #         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
# #             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
#         mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
#         mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

#     del unlabeled_image, unlabeled_mask, unlabeled_logits

#     return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


def cut_mix(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]
    # print(u_rand_index)
    
    # get box
    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(unlabeled_image.size(), lam=np.random.beta(4, 4))
    
    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        # label is of 3 dimensions
#         mix_unlabeled_target[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
#             unlabeled_mask[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
        
        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # 
# # 1.5 copy-paste for AD-MT integration
# # # # # # # # # # # # # # # # # # # # # 

def copy_paste(img_ulb, pseudo_outputs, pseudo_logits, img_lb, target_lb, paste_prob=0.5):
    """
    Simple Copy-Paste augmentation for AD-MT integration
    Copies objects from labeled data to unlabeled data
    
    Args:
        img_ulb: unlabeled images [B, C, H, W]
        pseudo_outputs: pseudo labels [B, H, W] 
        pseudo_logits: pseudo confidence [B, H, W]
        img_lb: labeled images [B, C, H, W]
        target_lb: ground truth labels [B, H, W]
        paste_prob: probability to apply copy-paste per sample
        
    Returns:
        augmented (img_ulb, pseudo_outputs, pseudo_logits)
    """
    batch_size = img_ulb.shape[0]
    
    # Clone inputs to avoid modifying originals
    aug_img_ulb = img_ulb.clone()
    aug_pseudo_outputs = pseudo_outputs.clone()
    aug_pseudo_logits = pseudo_logits.clone()
    
    for i in range(batch_size):
        if np.random.random() < paste_prob:
            # Select random labeled sample
            j = np.random.randint(0, img_lb.shape[0])
            
            # Get object mask from labeled data (exclude background class 0)
            obj_mask = (target_lb[j] > 0).float()
            
            if obj_mask.sum() > 100:  # Minimum object size threshold
                # Copy object pixels from labeled to unlabeled image
                aug_img_ulb[i] = aug_img_ulb[i] * (1 - obj_mask.unsqueeze(0)) + img_lb[j] * obj_mask.unsqueeze(0)
                
                # Update pseudo labels with ground truth labels in pasted regions
                aug_pseudo_outputs[i] = aug_pseudo_outputs[i] * (1 - obj_mask).long() + target_lb[j] * obj_mask.long()
                
                # Set high confidence for pasted regions (reliable ground truth)
                aug_pseudo_logits[i] = aug_pseudo_logits[i] * (1 - obj_mask) + obj_mask * 0.99
    
    return aug_img_ulb, aug_pseudo_outputs, aug_pseudo_logits


# # # # # # # # # # # # # # # # # # # # # 
# # 2. copy-paste for 2d (based on Google Research paper)
# # # # # # # # # # # # # # # # # # # # # 

def copy_paste_batch(unlabeled_image, unlabeled_mask, unlabeled_logits=None, unlabeled_conflict=None, 
                    num_classes=4, paste_prob=0.5, blur_sigma=1.0, use_blending=False):
    """
    Apply Copy-Paste augmentation to a batch of images
    Based on "Simple Copy-Paste is a Strong Data Augmentation Method"
    
    Args:
        unlabeled_image: tensor [B, C, H, W]
        unlabeled_mask: tensor [B, H, W]  
        unlabeled_logits: tensor [B, num_classes, H, W] (optional)
        unlabeled_conflict: tensor [B, H, W] (optional)
        num_classes: number of segmentation classes
        paste_prob: probability of pasting each object
        blur_sigma: gaussian blur sigma
        use_blending: whether to apply gaussian blending
    
    Returns:
        augmented batch tensors
    """
    
    batch_size = unlabeled_image.shape[0]
    
    # Clone tensors
    aug_image = unlabeled_image.clone()
    aug_mask = unlabeled_mask.clone()
    if unlabeled_logits is not None:
        aug_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        aug_conflict = unlabeled_conflict.clone()
    
    # Create random pairs for copy-paste
    source_indices = torch.randperm(batch_size)
    
    for i in range(batch_size):
        source_idx = source_indices[i]
        target_idx = i
        
        if source_idx == target_idx:
            continue  # Skip self-pairing
            
        # Find unique objects in source mask (excluding background class 0)
        source_mask = unlabeled_mask[source_idx]
        target_mask = unlabeled_mask[target_idx]
        unique_classes = torch.unique(source_mask)
        paste_classes = [cls for cls in unique_classes if cls > 0]  # exclude background
        
        if len(paste_classes) == 0:
            continue
            
        # Randomly select subset of objects to paste (following paper's approach)
        num_to_paste = random.randint(1, len(paste_classes))
        selected_classes = random.sample(paste_classes, num_to_paste)
        
        for cls in selected_classes:
            # Skip with paste_prob
            if random.random() > paste_prob:
                continue
                
            # Create binary mask for current object
            object_mask = (source_mask == cls).float()  # [H, W]
            
            if object_mask.sum() == 0:  # No pixels for this class
                continue
                
            # Apply gaussian blurring to mask edges if enabled
            if use_blending and blur_sigma > 0:
                object_mask_np = object_mask.cpu().numpy()
                blurred_mask = gaussian_filter(object_mask_np, sigma=blur_sigma)
                object_mask = torch.from_numpy(blurred_mask).to(object_mask.device)
                # Ensure values are in [0, 1] range
                object_mask = torch.clamp(object_mask, 0, 1)
            
            # Expand mask to match image dimensions [C, H, W]
            alpha = object_mask.unsqueeze(0).repeat(unlabeled_image.shape[1], 1, 1)
            
            # Copy-paste using alpha blending: I1 * alpha + I2 * (1 - alpha)
            aug_image[target_idx] = unlabeled_image[source_idx] * alpha + aug_image[target_idx] * (1 - alpha)
            
            # Update segmentation mask
            aug_mask[target_idx] = torch.where(object_mask > 0.5, source_mask, aug_mask[target_idx])
            
            # Handle logits if provided - copy corresponding logits
            if unlabeled_logits is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    for c in range(num_classes):
                        aug_logits[target_idx, c] = torch.where(
                            pasted_mask, 
                            unlabeled_logits[source_idx, c], 
                            aug_logits[target_idx, c]
                        )
            
            # Handle conflict if provided
            if unlabeled_conflict is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    aug_conflict[target_idx] = torch.where(
                        pasted_mask,
                        unlabeled_conflict[source_idx],
                        aug_conflict[target_idx]
                    )
    
    # Clean up memory
    del unlabeled_image, unlabeled_mask
    if unlabeled_logits is not None:
        del unlabeled_logits
        if unlabeled_conflict is not None:
            del unlabeled_conflict
            return aug_image, aug_mask, aug_logits, aug_conflict
        return aug_image, aug_mask, aug_logits
    
    if unlabeled_conflict is not None:
        del unlabeled_conflict
        return aug_image, aug_mask, aug_conflict
    
    return aug_image, aug_mask





# # # # # # # # # # # # # # # # # # # # # 
# # 4. cutmix for 3d
# # # # # # # # # # # # # # # # # # # # # 

def rand_bbox_3d(size, lam=None):
    # img: B x C x H x W x D, lb: B x H x W x D
    if len(size) == 5:
        W = size[2]
        H = size[3]
        D = size[4]
    elif len(size) == 4:
        W = size[1]
        H = size[2]
        D = size[3]
    else:
        raise Exception
    B = size[0]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cut_d = int(D * cut_rat)

    cx = np.random.randint(size=[B, ], low=int(W/8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H/8), high=H)
    cz = np.random.randint(size=[B, ], low=int(D/8), high=D)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)

    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)


    return bbx1, bby1, bbz1, bbx2, bby2, bbz2



# def cut_mix_3d(unlabeled_image, unlabeled_mask, unlabeled_logits):
#     mix_unlabeled_image = unlabeled_image.clone()
#     mix_unlabeled_target = unlabeled_mask.clone()
#     mix_unlabeled_logits = unlabeled_logits.clone()
    
#     # get the random mixing objects
#     u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

#     # get box
#     # img: B x C x H x W x D, lb: B x H x W x D
#     u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

#     # cut & paste
#     for i in range(0, mix_unlabeled_image.shape[0]):
#         mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
#         mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
#         mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
#             unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]

#     del unlabeled_image, unlabeled_mask, unlabeled_logits

#     return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits




def cut_mix_3d(unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict=None):
    mix_unlabeled_image = unlabeled_image.clone()
    mix_unlabeled_target = unlabeled_mask.clone()
    mix_unlabeled_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        mix_unlabeled_conflict = unlabeled_conflict.clone()
    
    # get the random mixing objects
    u_rand_index = torch.randperm(unlabeled_image.size()[0])[:unlabeled_image.size()[0]]

    # get box
    # img: B x C x H x W x D, lb: B x H x W x D
    u_bbx1, u_bby1, u_bbz1, u_bbx2, u_bby2, u_bbz2 = rand_bbox_3d(unlabeled_image.size(), lam=np.random.beta(4, 4))

    # cut & paste
    for i in range(0, mix_unlabeled_image.shape[0]):
        mix_unlabeled_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_image[u_rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        mix_unlabeled_target[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_mask[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        mix_unlabeled_logits[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
            unlabeled_logits[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
        if unlabeled_conflict is not None:
            mix_unlabeled_conflict[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]] = \
                unlabeled_conflict[u_rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i], u_bbz1[i]:u_bbz2[i]]
        
    if unlabeled_conflict is not None:
        del unlabeled_image, unlabeled_mask, unlabeled_logits, unlabeled_conflict
        return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits, mix_unlabeled_conflict

    del unlabeled_image, unlabeled_mask, unlabeled_logits
    return mix_unlabeled_image, mix_unlabeled_target, mix_unlabeled_logits


# # # # # # # # # # # # # # # # # # # # # 
# # 2. copy-paste for 2d (based on Google Research paper)
# # # # # # # # # # # # # # # # # # # # # 

def copy_paste_batch(unlabeled_image, unlabeled_mask, unlabeled_logits=None, unlabeled_conflict=None, 
                    num_classes=4, paste_prob=0.5, blur_sigma=1.0, use_blending=False):
    """
    Apply Copy-Paste augmentation to a batch of images
    Based on "Simple Copy-Paste is a Strong Data Augmentation Method"
    
    Args:
        unlabeled_image: tensor [B, C, H, W]
        unlabeled_mask: tensor [B, H, W]  
        unlabeled_logits: tensor [B, num_classes, H, W] (optional)
        unlabeled_conflict: tensor [B, H, W] (optional)
        num_classes: number of segmentation classes
        paste_prob: probability of pasting each object
        blur_sigma: gaussian blur sigma
        use_blending: whether to apply gaussian blending
    
    Returns:
        augmented batch tensors
    """
    
    batch_size = unlabeled_image.shape[0]
    
    # Clone tensors
    aug_image = unlabeled_image.clone()
    aug_mask = unlabeled_mask.clone()
    if unlabeled_logits is not None:
        aug_logits = unlabeled_logits.clone()
    if unlabeled_conflict is not None:
        aug_conflict = unlabeled_conflict.clone()
    
    # Create random pairs for copy-paste
    source_indices = torch.randperm(batch_size)
    
    for i in range(batch_size):
        source_idx = source_indices[i]
        target_idx = i
        
        if source_idx == target_idx:
            continue  # Skip self-pairing
            
        # Find unique objects in source mask (excluding background class 0)
        source_mask = unlabeled_mask[source_idx]
        target_mask = unlabeled_mask[target_idx]
        unique_classes = torch.unique(source_mask)
        paste_classes = [cls for cls in unique_classes if cls > 0]  # exclude background
        
        if len(paste_classes) == 0:
            continue
            
        # Randomly select subset of objects to paste (following paper's approach)
        num_to_paste = random.randint(1, len(paste_classes))
        selected_classes = random.sample(paste_classes, num_to_paste)
        
        for cls in selected_classes:
            # Skip with paste_prob
            if random.random() > paste_prob:
                continue
                
            # Create binary mask for current object
            object_mask = (source_mask == cls).float()  # [H, W]
            
            if object_mask.sum() == 0:  # No pixels for this class
                continue
                
            # Apply gaussian blurring to mask edges if enabled
            if use_blending and blur_sigma > 0:
                object_mask_np = object_mask.cpu().numpy()
                blurred_mask = gaussian_filter(object_mask_np, sigma=blur_sigma)
                object_mask = torch.from_numpy(blurred_mask).to(object_mask.device)
                # Ensure values are in [0, 1] range
                object_mask = torch.clamp(object_mask, 0, 1)
            
            # Expand mask to match image dimensions [C, H, W]
            alpha = object_mask.unsqueeze(0).repeat(unlabeled_image.shape[1], 1, 1)
            
            # Copy-paste using alpha blending: I1 * alpha + I2 * (1 - alpha)
            aug_image[target_idx] = unlabeled_image[source_idx] * alpha + aug_image[target_idx] * (1 - alpha)
            
            # Update segmentation mask
            aug_mask[target_idx] = torch.where(object_mask > 0.5, source_mask, aug_mask[target_idx])
            
            # Handle logits if provided - copy corresponding logits
            if unlabeled_logits is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    for c in range(num_classes):
                        aug_logits[target_idx, c] = torch.where(
                            pasted_mask, 
                            unlabeled_logits[source_idx, c], 
                            aug_logits[target_idx, c]
                        )
            
            # Handle conflict if provided
            if unlabeled_conflict is not None:
                pasted_mask = (object_mask > 0.5)
                if pasted_mask.sum() > 0:
                    aug_conflict[target_idx] = torch.where(
                        pasted_mask,
                        unlabeled_conflict[source_idx],
                        aug_conflict[target_idx]
                    )
    
    # Clean up memory
    del unlabeled_image, unlabeled_mask
    if unlabeled_logits is not None:
        del unlabeled_logits
        if unlabeled_conflict is not None:
            del unlabeled_conflict
            return aug_image, aug_mask, aug_logits, aug_conflict
        return aug_image, aug_mask, aug_logits
    
    if unlabeled_conflict is not None:
        del unlabeled_conflict
        return aug_image, aug_mask, aug_conflict
    
    return aug_image, aug_mask

