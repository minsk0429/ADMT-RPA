import argparse
import logging
import os
import os.path as osp
import random
import sys
import yaml


import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


from dataloaders.mixaugs import cut_mix, copy_paste
from dataloaders.mixaugs_visualization import (visualize_cutmix_process, visualize_copypaste_process,
                                               extract_cutmix_bbox_coords, extract_copypaste_regions)
# [추가] 수동 Jitter 클래스 정의
import torchvision.transforms as T

class ConsistentStrongAug(nn.Module):
    """
    dataset_2d.py의 func_strong_augs와 동일한 스펙을 가진 Tensor 전용 증강 모듈
    - ColorJitter(0.5, 0.5, 0.5, 0.25) with p=0.8
    - GaussianBlur(sigma=(0.1, 2.0)) with p=0.2
    """
    def __init__(self):
        super().__init__()
        self.color_jitter = T.RandomApply(
            [T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)],
            p=0.8
        )
        self.gaussian_blur = T.RandomApply(
            [T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
            p=0.2
        )

    def forward(self, img_tensor):
        # img_tensor: (B, C, H, W) range [0.0, 1.0]
        img_tensor = self.color_jitter(img_tensor)
        img_tensor = self.gaussian_blur(img_tensor)
        return img_tensor

# 전역 인스턴스 생성
manual_strong_aug = ConsistentStrongAug()

def simple_color_copy_paste(img_ulb, pseudo_outputs, pseudo_logits, img_lb, target_lb,
                           paste_prob=1.0, color_threshold=0.6,
                           min_size_ratio=0.05, max_size_ratio=0.45):
    """
    색상 및 크기 조건부 Copy-Paste (EXP6)

    조건:
    1. 용종 크기: 이미지의 5%~45% (3277~29491 픽셀 at 256x256)
    2. 색상 유사도: color_threshold 이상

    조건 불충족 시 배치 내 모든 라벨 데이터를 시도
    """
    batch_size = img_ulb.shape[0]
    lb_batch_size = img_lb.shape[0]
    color_matches = 0

    pasted_flags = [False] * batch_size
    # 이미지 크기 기반 픽셀 수 계산 (256x256 = 65536)
    _, _, H, W = img_ulb.shape
    total_pixels = H * W
    min_pixels = int(total_pixels * min_size_ratio)  # 5% = 3277
    max_pixels = int(total_pixels * max_size_ratio)  # 45% = 29491

    for i in range(batch_size):
        if np.random.random() < paste_prob:
            # RGB 채널별 평균 벡터 [C]
            ulb_mean_rgb = img_ulb[i].mean(dim=(1, 2))  # [C]

            # 배치 내 모든 라벨 데이터를 랜덤 순서로 시도
            indices = list(range(lb_batch_size))
            np.random.shuffle(indices)

            for j in indices:
                # 조건 1: 용종 크기 확인 (5%~45%)
                obj_mask = (target_lb[j] > 0)
                polyp_size = obj_mask.sum().item()

                if polyp_size < min_pixels or polyp_size > max_pixels:
                    continue  # 크기 조건 불충족 -> 다음 라벨로

                # 조건 2: 색상 유사도 확인 (RGB L2 거리 기반)
                lb_mean_rgb = img_lb[j].mean(dim=(1, 2))  # [C]
                color_dist = torch.norm(ulb_mean_rgb - lb_mean_rgb).item() / np.sqrt(3)  # 0~1 정규화
                color_similarity = 1.0 - color_dist

                if color_similarity < color_threshold:
                    continue  # 색상 유사도 불충족 -> 다음 라벨로

                # 두 조건 모두 충족 -> Copy-Paste 실행
                aug_img, aug_pseudo_out, aug_pseudo_log = copy_paste(
                    img_ulb[i:i+1], pseudo_outputs[i:i+1], pseudo_logits[i:i+1],
                    img_lb[j:j+1], target_lb[j:j+1]
                )

                # 🔥 [수정] 진짜로 이미지가 변했는지 픽셀 차이 계산 (Diff Check)
                # 부동소수점 오차 등을 고려해 1e-4보다 큰 변화가 있어야 함
                pixel_diff = torch.sum(torch.abs(img_ulb[i] - aug_img[0]))

                if pixel_diff > 1e-4:
                    # 변화가 확실히 있을 때만 업데이트 및 True 설정
                    img_ulb[i] = aug_img[0]
                    pseudo_outputs[i] = aug_pseudo_out[0]
                    pseudo_logits[i] = aug_pseudo_log[0]
                    color_matches += 1
                    pasted_flags[i] = True
                    break
                else:
                    # 변화가 없으면(실패했으면) False 유지
                    pasted_flags[i] = False
                    # 이미지를 업데이트하지 않음 (원본 유지)
                
                # 성공 여부와 관계없이 한 번 시도했으면 루프 탈출 (기존 로직 유지)
                # 만약 '성공할 때까지' 찾고 싶다면 if pasted_flags[i]: break 로 바꾸세요.
                #if pasted_flags[i]:
                     #break

    return img_ulb, pseudo_outputs, pseudo_logits, color_matches, pasted_flags
from dataloaders.dataset_2d import (BaseDataSets, KvasirDataSets, TwoStreamBatchSampler, WeakStrongAugment)
from networks.net_factory import net_factory
from utils import losses, ramps
from utils.util import update_values, time_str, AverageMeter
from train_utils import AlternateUpdate, get_compromise_pseudo_btw_tea_stu
from val_2D import test_single_volume
# All trackers disabled
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                   I. Adaptive Augmentation Scheduler
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

class AdaptiveAugmentationSchedulerExp6:
    """
    Validation Dice Score 기반 적응적 증강 전환 (EXP6)
    Copy-Paste (쉬움) -> CutMix (어려움) - 2단계만

    EXP5 + 크기 조건 추가: 용종 크기 5%~45% 제한
    """
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience  # plateau 판단 validation 횟수
        self.min_delta = min_delta  # 의미있는 Dice 개선량 (1%)

        # 증강 단계 (쉬운 것부터 어려운 순서) - 2단계만!
        self.stages = [
            {'name': 'copy_paste', 'cp_prob': 1.0, 'cutmix_prob': 0.0, 'description': 'Copy-Paste Only (Easy Learning)'},
            {'name': 'cutmix',     'cp_prob': 0.0, 'cutmix_prob': 1.0, 'description': 'CutMix Only (Hard Regularization)'},
        ]
        self.current_stage = 0

        # Dice 추적
        self.dice_history = []
        self.best_dice = 0.0  # Dice는 높을수록 좋음
        self.patience_counter = 0
        self.switched = False

        # 전환 이력
        self.switch_history = []

    def step_dice(self, current_dice, iteration):
        """Validation 때마다 호출하여 Dice 기준으로 증강 전략 업데이트"""
        self.dice_history.append(current_dice)

        # Dice가 개선되었는가? (높을수록 좋음)
        if current_dice > self.best_dice + self.min_delta:
            self.best_dice = current_dice
            self.patience_counter = 0
            self.switched = False
        else:
            self.patience_counter += 1

        # Plateau 감지 -> 다음 단계로 전환
        if self.patience_counter >= self.patience and not self.switched:
            if self.current_stage < len(self.stages) - 1:
                old_stage = self.stages[self.current_stage]['name']
                self.current_stage += 1
                new_stage = self.stages[self.current_stage]['name']

                self.patience_counter = 0
                self.switched = True
                self.best_dice = 0.0  # Reset

                # 전환 이력 기록
                self.switch_history.append({
                    'iteration': iteration,
                    'from_stage': old_stage,
                    'to_stage': new_stage,
                    'current_dice': current_dice,
                    'best_dice_before': self.best_dice
                })

                print(f"\n{'='*70}")
                print(f"[EXP5] AUGMENTATION SWITCH at Iteration {iteration}")
                print(f"{'='*70}")
                print(f"From: {old_stage} -> To: {new_stage}")
                print(f"Reason: Dice plateau detected (patience={self.patience} validations)")
                print(f"Current Dice: {current_dice:.4f} | Best Dice: {self.best_dice:.4f}")
                print(f"Description: {self.stages[self.current_stage]['description']}")
                print(f"{'='*70}\n")

        return self.stages[self.current_stage]

    def get_current_stage_name(self):
        """현재 증강 단계 이름 반환"""
        return self.stages[self.current_stage]['name']

    def get_switch_history(self):
        """전환 이력 반환"""
        return self.switch_history


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def track_target_images(sampled_batch, num_ulb):
    """
    여러 타겟 이미지가 unlabeled 배치에 포함되어 있는지 확인
    Returns: (found, batch_position, target_info)
    """
    batch_indices = sampled_batch.get("idx", None)
    if batch_indices is None:
        return False, -1, None

    # 5개의 타겟 이미지들과 정보 (언라벨 영역 500+ 인덱스)
    target_images = {
        510: "cju6vgdmivcvb08018fra5lnv",  # 인덱스 510
        515: "cju6vucxvvlda0755j7msqnya",  # 인덱스 515
        525: "cju6x0yqbvxqt0755dhxislgb",  # 인덱스 525
        535: "cju6yywx1whbb0871ksgfgf9f",  # 인덱스 535
        610: "cju7d3oc82cho0755dajlwldz",  # 인덱스 610
    }

    # unlabeled 부분의 인덱스 추출 (첫 num_ulb개가 unlabeled)
    unlabeled_indices = batch_indices[:num_ulb]

    # 디버그: 현재 배치의 unlabeled 인덱스들 확인
    unlabeled_list = [idx.item() for idx in unlabeled_indices]

    # 타겟 이미지들 중 하나라도 있는지 확인
    for target_idx, target_name in target_images.items():
        if target_idx in unlabeled_list:
            print(f"TARGET FOUND! {target_name} (index {target_idx}) in unlabeled batch: {unlabeled_list}")

            # 배치에서 해당 타겟 이미지 위치 찾기
            for batch_pos, dataset_idx in enumerate(unlabeled_indices):
                if dataset_idx.item() == target_idx:
                    print(f"Target image {target_name} found at batch position {batch_pos}")
                    return True, batch_pos, {"index": target_idx, "filename": target_name}

    return False, -1, None


def save_target_image_visualization(iter_num, target_info, batch_pos, img_ulb_w, img_ulb_s, img_ulb_s_original,
                                  ema_outputs_soft_1, ema_outputs_soft_2, weighted_outputs,
                                  pseudo_outputs, pseudo_logits,
                                  pseudo_outputs_before_aug, pseudo_logits_before_aug,  # 증강 전 수도라벨 추가
                                  pred_ulb_w, pred_ulb_s,
                                  mtx_bool_conflict, cutmix_applied, copypaste_applied,
                                  alternate_state, snapshot_path, args,
                                  target_gt=None, target_gt_before_aug=None):  # GT 라벨 (증강 전/후)
    """
    타겟 이미지의 AD-MT 파이프라인 전체 과정을 시각화하여 저장
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    import os

    # 저장 디렉토리 생성 (실험 폴더 내부에)
    target_filename = target_info["filename"]
    target_index = target_info["index"]
    save_dir = os.path.join(snapshot_path, "target_tracking", target_filename)
    os.makedirs(save_dir, exist_ok=True)

    # 타겟 이미지의 배치 내 위치 사용
    idx = batch_pos

    # 이미지를 numpy로 변환 (시각화용)
    def tensor_to_numpy(tensor_img):
        if tensor_img.dim() == 4:  # batch 차원에서 해당 위치 선택
            tensor_img = tensor_img[idx]
        img = tensor_img.detach().cpu().numpy()
        if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            img = img.transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        return img

    # 예측 마스크를 numpy로 변환
    def pred_to_numpy(pred_tensor):
        if pred_tensor.dim() == 4:
            pred_tensor = pred_tensor[idx]
        elif pred_tensor.dim() == 3:
            pred_tensor = pred_tensor[idx]

        if pred_tensor.dim() == 3:  # (C, H, W) - softmax 적용
            pred = F.softmax(pred_tensor, dim=0)
            pred_mask = torch.argmax(pred, dim=0).cpu().numpy()
        else:  # (H, W) - 이미 argmax 적용된 경우
            pred_mask = pred_tensor.cpu().numpy()
        return pred_mask

    try:
        # 전체 파이프라인 시각화 - 5x3 그리드로 확장하여 GT 전후 비교 추가
        fig, axes = plt.subplots(5, 3, figsize=(12, 20))
        fig.suptitle(f'AD-MT Pipeline Tracking - {target_filename} (Index: {target_index}, Iter: {iter_num}, Batch Pos: {batch_pos})', fontsize=14)

        # Row 1: 입력 이미지들
        axes[0, 0].imshow(tensor_to_numpy(img_ulb_w))
        axes[0, 0].set_title('Weak Augmented')
        axes[0, 0].axis('off')

        if img_ulb_s_original is not None:
            axes[0, 1].imshow(tensor_to_numpy(img_ulb_s_original))
            axes[0, 1].set_title('Strong Aug (Original)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Original', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(tensor_to_numpy(img_ulb_s))
        aug_title = f'Final Strong Aug\n(CP:{copypaste_applied})'
        axes[0, 2].set_title(aug_title)
        axes[0, 2].axis('off')

        # Row 2: Teacher 예측들
        if ema_outputs_soft_1 is not None:
            teacher1_mask = pred_to_numpy(ema_outputs_soft_1)
            axes[1, 0].imshow(teacher1_mask, cmap='jet', alpha=0.7)
            axes[1, 0].set_title('Teacher 1 Prediction')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Teacher 1', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')

        if ema_outputs_soft_2 is not None:
            teacher2_mask = pred_to_numpy(ema_outputs_soft_2)
            axes[1, 1].imshow(teacher2_mask, cmap='jet', alpha=0.7)
            axes[1, 1].set_title('Teacher 2 Prediction')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Teacher 2', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')

        if weighted_outputs is not None:
            ensemble_mask = pred_to_numpy(weighted_outputs)
            axes[1, 2].imshow(ensemble_mask, cmap='jet', alpha=0.7)
            axes[1, 2].set_title('Teacher Ensemble')
            axes[1, 2].axis('off')
        else:
            axes[1, 2].text(0.5, 0.5, 'No Ensemble', ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].axis('off')

        # Row 3: 수도라벨 (증강 전/후)
        # 증강 전 수도라벨
        if pseudo_outputs_before_aug is not None:
            if pseudo_outputs_before_aug.dim() == 3:  # (B, H, W)
                pseudo_mask_before = pseudo_outputs_before_aug[idx].cpu().numpy()
                conf_before = pseudo_logits_before_aug[idx].cpu().numpy().mean()
            else:  # (H, W)
                pseudo_mask_before = pseudo_outputs_before_aug.cpu().numpy()
                conf_before = pseudo_logits_before_aug.cpu().numpy().mean()
            axes[2, 0].imshow(pseudo_mask_before, cmap='jet', alpha=0.7)
            axes[2, 0].set_title(f'Pseudo Before Aug\n(Conf: {conf_before:.3f})')
            axes[2, 0].axis('off')
        else:
            axes[2, 0].text(0.5, 0.5, 'No Pseudo Before', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].axis('off')

        # 빈 공간 (추후 다른 시각화 요소 추가 가능)
        axes[2, 1].text(0.5, 0.5, 'Comparison\nSpace', ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].axis('off')

        # 증강 후 수도라벨 (최종)
        if pseudo_outputs is not None:
            if pseudo_outputs.dim() == 3:  # (B, H, W)
                pseudo_mask = pseudo_outputs[idx].cpu().numpy()
                confidence = pseudo_logits[idx].cpu().numpy().mean()
            else:  # (H, W)
                pseudo_mask = pseudo_outputs.cpu().numpy()
                confidence = pseudo_logits.cpu().numpy().mean()
            axes[2, 2].imshow(pseudo_mask, cmap='jet', alpha=0.7)
            axes[2, 2].set_title(f'Pseudo After Aug\n(Conf: {confidence:.3f})')
            axes[2, 2].axis('off')
        else:
            axes[2, 2].text(0.5, 0.5, 'No Pseudo After', ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].axis('off')

        # Row 4: GT 비교 (증강 전/후)
        # GT Before Augmentation
        if target_gt_before_aug is not None:
            if target_gt_before_aug.dim() == 3:  # (B, H, W)
                gt_before_mask = target_gt_before_aug[idx].cpu().numpy()
            elif target_gt_before_aug.dim() == 2:  # (H, W)
                gt_before_mask = target_gt_before_aug.cpu().numpy()
            else:  # (B, C, H, W) - one-hot encoded
                gt_before_mask = target_gt_before_aug[idx].argmax(dim=0).cpu().numpy()
            axes[3, 0].imshow(gt_before_mask, cmap='jet', alpha=0.7)
            axes[3, 0].set_title('GT Before Aug')
            axes[3, 0].axis('off')
        else:
            axes[3, 0].text(0.5, 0.5, 'No GT Before', ha='center', va='center', transform=axes[3, 0].transAxes)
            axes[3, 0].axis('off')

        # GT After Augmentation (기존)
        if target_gt is not None:
            if target_gt.dim() == 3:  # (B, H, W)
                gt_mask = target_gt[idx].cpu().numpy()
            elif target_gt.dim() == 2:  # (H, W)
                gt_mask = target_gt.cpu().numpy()
            else:  # (B, C, H, W) - one-hot encoded
                gt_mask = target_gt[idx].argmax(dim=0).cpu().numpy()
            axes[3, 1].imshow(gt_mask, cmap='jet', alpha=0.7)
            axes[3, 1].set_title('GT After Aug')
            axes[3, 1].axis('off')
        else:
            axes[3, 1].text(0.5, 0.5, 'No GT After', ha='center', va='center', transform=axes[3, 1].transAxes)
            axes[3, 1].axis('off')

        # Augmentation Effect (차이)
        if target_gt_before_aug is not None and target_gt is not None:
            # Calculate difference between before and after
            if target_gt_before_aug.dim() == 3 and target_gt.dim() == 3:
                gt_before = target_gt_before_aug[idx].cpu().numpy()
                gt_after = target_gt[idx].cpu().numpy()
                diff_mask = (gt_before != gt_after).astype(np.float32)
                axes[3, 2].imshow(diff_mask, cmap='Reds', alpha=0.8)
                change_ratio = diff_mask.mean()
                axes[3, 2].set_title(f'GT Changes\n({change_ratio:.2%})')
                axes[3, 2].axis('off')
            else:
                axes[3, 2].text(0.5, 0.5, 'GT Diff\nN/A', ha='center', va='center', transform=axes[3, 2].transAxes)
                axes[3, 2].axis('off')
        else:
            axes[3, 2].text(0.5, 0.5, 'No GT\nComparison', ha='center', va='center', transform=axes[3, 2].transAxes)
            axes[3, 2].axis('off')

        # Row 5: Student 예측 및 정보
        student_w_mask = pred_to_numpy(pred_ulb_w)
        axes[4, 0].imshow(student_w_mask, cmap='jet', alpha=0.7)
        axes[4, 0].set_title('Student Pred (Weak)')
        axes[4, 0].axis('off')

        student_s_mask = pred_to_numpy(pred_ulb_s)
        axes[4, 1].imshow(student_s_mask, cmap='jet', alpha=0.7)
        axes[4, 1].set_title('Student Pred (Strong)')
        axes[4, 1].axis('off')

        # 갈등 영역 시각화 또는 정보 요약
        if mtx_bool_conflict is not None:
            conflict_map = mtx_bool_conflict[idx].cpu().numpy()
            axes[4, 2].imshow(conflict_map, cmap='Reds', alpha=0.8)
            conflict_ratio = conflict_map.mean()
            axes[4, 2].set_title(f'Conflict Map\n({conflict_ratio:.2%})')
            axes[4, 2].axis('off')
        else:
            # 정보 요약
            info_text = f'Iteration: {iter_num}\n'
            info_text += f'Target: {target_filename}\n'
            info_text += f'Dataset Index: {target_index}\n'
            info_text += f'Batch Position: {batch_pos}\n'
            info_text += f'Alternate: {alternate_state}\n'
            info_text += f'Augmentations Applied:\n'
            info_text += f'  - CutMix: {cutmix_applied}\n'
            info_text += f'  - Copy-Paste: {copypaste_applied}'

            axes[4, 2].text(0.1, 0.1, info_text, transform=axes[4, 2].transAxes,
                           fontsize=8, verticalalignment='bottom')
            axes[4, 2].axis('off')

        plt.tight_layout()

        # 저장
        save_path = os.path.join(save_dir, f'pipeline_iter_{iter_num:06d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Target image visualization saved: {save_path}")

    except Exception as e:
        print(f"Error in target image visualization: {e}")
        if 'fig' in locals():
            plt.close(fig)


def visualize_unlabeled_loss_process(*args, **kwargs):
    """Visualization disabled"""
    pass


def visualize_augmentation_pipeline(*args, **kwargs):
    """Visualization disabled"""
    pass




def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
        return ref_dict[str(patiens_num)]
    elif "Prostate" in dataset:
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
        return ref_dict[str(patiens_num)]
    elif "Kvasir" in dataset:
        # 879 total samples: return the number directly
        return patiens_num
    else:
        print("Error")
        return 0


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args["consistency"] * ramps.sigmoid_rampup(epoch, args["consistency_rampup"])


def update_ema_variables(model, ema_model, alpha, global_step, args):
    # adjust the momentum param
    if global_step < args["consistency_rampup"]:
        alpha = 0.0 
    else:
        alpha = min(1 - 1 / (global_step - args["consistency_rampup"] + 1), alpha)
    
    # update weights
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    
    # update buffers
    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.data = buffer_eval.data * alpha + buffer_train.data * (1 - alpha)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        II. trainer
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def train(args, snapshot_path):
    base_lr = args["base_lr"]
    num_classes = args["num_classes"]
    batch_size = args["batch_size"]
    max_iterations = args["max_iterations"]
    cur_time = time_str()
    log_dir = os.path.join(snapshot_path, 'log')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    csv_train = os.path.join(log_dir, "seg_{}_train_iter.csv".format(cur_time))
    csv_test = os.path.join(log_dir, "seg_{}_validate_ep.csv".format(cur_time))

    def create_model(ema=False):
        # Network definition
        in_channels = 3 if "Kvasir" in args["root_path"] else 1  # RGB for Kvasir, grayscale for others
        model = net_factory(net_type=args["model"], in_chns=in_channels,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    def worker_init_fn(worker_id):
        random.seed(args["seed"] + worker_id)

    # + + + + + + + + + + + #
    # 1. create model
    # + + + + + + + + + + + #
    model = create_model()
    ema_model = create_model(ema=True)
    ema_model_another = create_model(ema=True)
    model.cuda()
    ema_model.cuda()
    ema_model_another.cuda()
    model.train()
    ema_model.train()
    ema_model_another.train()

    # + + + + + + + + + + + #
    # 2. dataset
    # + + + + + + + + + + + #
    if "Kvasir" in args["root_path"]:
        db_train = KvasirDataSets(base_dir=args["root_path"], split="train", num=None, 
                                    transform=transforms.Compose([WeakStrongAugment(args["patch_size"])])
                                    )
        db_val = KvasirDataSets(base_dir=args["root_path"], split="val")
    else:
        db_train = BaseDataSets(base_dir=args["root_path"], split="train", num=None, 
                                    transform=transforms.Compose([WeakStrongAugment(args["patch_size"])])
                                    )
        db_val = BaseDataSets(base_dir=args["root_path"], split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args["root_path"], args["labeled_num"])
    logging.info("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))

    if args.get("flag_sampling_based_on_lb", False):
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, batch_size, batch_size-args["labeled_bs"])
    else:
        batch_sampler = TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, batch_size, args["labeled_bs"])

    # + + + + + + + + + + + #
    # 3. dataloader
    # + + + + + + + + + + + #
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)
    logging.info("{} iterations per epoch".format(len(trainloader)))

    # + + + + + + + + + + + #
    # 4. optim, scheduler
    # + + + + + + + + + + + #
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    # + + + + + + + + + + + #
    # 5. training loop
    # + + + + + + + + + + + #
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance_another = 0.0
    best_performance_stu = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # alternate params
    alt_flag_epoch_shuffle_teachers = args["alt_flag_epoch_shuffle_teachers"]
    alt_flag_conflict_mode = args["alt_flag_conflict_mode"]
    alt_flag_conflict_stu_use_more = args["alt_flag_conflict_stu_use_more"]
    alt_param_ensemble_temp = args["alt_param_ensemble_temp"]

    alt_param_conflict_weight = args["alt_param_conflict_weight"]
    alt_param_updating_period_iters = args["alt_param_updating_period_iters"]
    alt_flag_updating_period_random = args["alt_flag_updating_period_random"]
    
    alt_param_updating_period_iters = args["alt_param_updating_period_iters"]

    alternate_indicator = AlternateUpdate(alt_param_updating_period_iters,
                                          initial_flag=True, flag_random=alt_flag_updating_period_random)

    # 🔥 EXP3: Dice-based Adaptive Augmentation Scheduler 초기화 (2단계만)
    aug_scheduler = AdaptiveAugmentationSchedulerExp6(
        patience=args.get("aug_patience", 5),       # 5 validations (10 epochs)
        min_delta=args.get("aug_min_delta", 0.01)  # 1% Dice improvement
    )
    logging.info(f"[EXP6] Dice-based Adaptive Augmentation Scheduler (2-Stage + Size Condition 5%~45%)")
    logging.info(f"  - Patience: {aug_scheduler.patience} validations")
    logging.info(f"  - Min Delta: {aug_scheduler.min_delta:.3f} (Dice)")
    logging.info(f"  - Stages: Copy-Paste → CutMix (2-stage only)")

    # All trackers disabled
    prediction_tracker = None
    unlabeled_loss_tracker = None
    complete_process_tracker = None
    for epoch_num in iterator:
        # update alt_params
        # a) alternate flag
        if alt_flag_epoch_shuffle_teachers:
            alternate_indicator.reset(alt_param_updating_period_iters, 
                                      initial_flag=(epoch_num % 2 == 0), 
                                      flag_random=alt_flag_updating_period_random)
        # b) conflict weight
        var_param_conflict_weight = alt_param_conflict_weight
        # c) flag of starting self training
        flag_start_self_learning = False

        # metric indicators
        meter_sup_losses = AverageMeter()
        meter_uns_losses = AverageMeter(20)
        meter_train_losses = AverageMeter(20)
        meter_learning_rates = AverageMeter()
        meter_highc_ratio = AverageMeter()
        meter_conflict_ratio = AverageMeter()
        meter_uns_losses_consist = AverageMeter(20)
        meter_uns_losses_conflict = AverageMeter(20)
        meter_copypaste_ratio = AverageMeter()  # Copy-Paste 적용 비율 추적
        meter_cutmix_ratio = AverageMeter()  # CutMix 적용 비율 추적

        

        for i_batch, sampled_batch in enumerate(trainloader):
            num_lb = args["labeled_bs"]
            num_ulb = batch_size - num_lb

            # 👇 [수정] A1과 A2의 성적표를 미리 분리해서 만들어둡니다.
            pasted_flags_A1 = [False] * batch_size 
            pasted_flags_A2 = [False] * batch_size

            # 1) get augmented data
            weak_batch, strong_batch, label_batch = (
                sampled_batch["image_weak"],
                sampled_batch["image_strong"],
                sampled_batch["label_aug"],
            )
            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            # get batched data
            if args.get("flag_sampling_based_on_lb", False):
                img_lb_w, target_lb = weak_batch[:num_lb], label_batch[:num_lb]
                img_ulb_w, img_ulb_s = weak_batch[num_lb:], strong_batch[num_lb:]
            else:
                img_lb_w, target_lb = weak_batch[num_ulb:], label_batch[num_ulb:]
                img_ulb_w, img_ulb_s = weak_batch[:num_ulb], strong_batch[:num_ulb]
                # GT 라벨도 저장 (시각화용)
                target_ulb = label_batch[:num_ulb]

            # 변수를 미리 초기화해둡니다.
            cutmix_applied = False
            copypaste_applied = False

            # 2) getting pseudo labels
            loss_ulb_consist, loss_ulb_conflict = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            with torch.no_grad():
                if not flag_start_self_learning and not args["flag_pseudo_from_student"]:
                    if alternate_indicator.get_alternate_state():
                        ema_model.train()
                        ema_model_another.eval()
                    else:
                        ema_model_another.train()
                        ema_model.eval()

                    
                    ema_outputs_soft_1 = torch.softmax(ema_model(img_ulb_w), dim=1)
                    ema_outputs_soft_1 = ema_outputs_soft_1.detach()

                    ema_outputs_soft_2 = torch.softmax(ema_model_another(img_ulb_w), dim=1)
                    ema_outputs_soft_2 = ema_outputs_soft_2.detach()

                    _, pseudo_outputs_1 = torch.max(ema_outputs_soft_1, dim=1)
                    _, pseudo_outputs_2 = torch.max(ema_outputs_soft_2, dim=1)

                    mtx_bool_conflict = pseudo_outputs_1 != pseudo_outputs_2
                    conflict_ratio = mtx_bool_conflict.float().sum() / num_ulb

                    # entropy
                    entropy_1 = -torch.sum(ema_outputs_soft_1 * torch.log2(ema_outputs_soft_1 + 1e-10), dim=1)
                    entropy_2 = -torch.sum(ema_outputs_soft_2 * torch.log2(ema_outputs_soft_2 + 1e-10), dim=1)

                    # weighted sum (Teacher 앙상블)
                    weights_1 = torch.exp(-entropy_1) / (torch.exp(-entropy_1) + torch.exp(-entropy_2))
                    weights_2 = 1.0 - weights_1
                    weighted_outputs = weights_1.unsqueeze(1) * ema_outputs_soft_1 + weights_2.unsqueeze(1) * ema_outputs_soft_2

                    weighted_outputs = torch.pow(weighted_outputs, 1.0 / alt_param_ensemble_temp)
                    weighted_outputs = weighted_outputs / torch.sum(weighted_outputs, dim=1, keepdim=True)

                    # Teacher ensemble prediction
                    _, teacher_ensemble_pred = torch.max(weighted_outputs, dim=1)

                    # Teacher ensemble prediction만 사용 (원본 AD-MT 방식)
                    pseudo_logits, pseudo_outputs = torch.max(weighted_outputs, dim=1)

                    # Student prediction은 loss 계산할 때만 사용하도록 준비
                    model.eval()
                    student_outputs_soft = torch.softmax(model(img_ulb_w), dim=1)
                    pseudo_logits_stu, pseudo_outputs_stu = torch.max(student_outputs_soft, dim=1)
                    model.train()

                    # Store for visualization (clone to avoid deletion)
                    ema_outputs_soft_1_vis = ema_outputs_soft_1.clone()
                    ema_outputs_soft_2_vis = ema_outputs_soft_2.clone()
                    weighted_outputs_vis = weighted_outputs.clone()

                    del ema_outputs_soft_1, ema_outputs_soft_2, weighted_outputs, entropy_1, entropy_2

                else:
                    model.eval()
                    ema_outputs_soft = torch.softmax(model(img_ulb_w), dim=1)
                    pseudo_logits, pseudo_outputs = torch.max(ema_outputs_soft.detach(), dim=1)
                    mtx_bool_conflict = None
                    conflict_ratio = torch.tensor(0.0).cuda()
                    model.train()
                
                    # For visualization when not using teacher ensemble
                    ema_outputs_soft_1_vis = None
                    ema_outputs_soft_2_vis = None
                    weighted_outputs_vis = None

                pseudo_outputs_before_aug = pseudo_outputs.clone().detach()
                pseudo_logits_before_aug = pseudo_logits.clone().detach()
            

            # ==================================================================
            # [위치 이동됨] Stage 1 재조립: Teacher 라벨 생성 "이후"에 실행!
            # ==================================================================
            current_stage_name = aug_scheduler.get_current_stage_name()
            aug_config = aug_scheduler.stages[aug_scheduler.current_stage]

            # 덮어쓰기 전에 깨끗한 원본 라벨을 'pseudo_outputs_weak'에 피신시킵니다.
            pseudo_outputs_weak = pseudo_outputs.clone()

            if current_stage_name == 'copy_paste':
                # 1. 기존 지터링된 이미지는 버리고, 깨끗한 'img_ulb_w'를 베이스로 복사
                img_ulb_s = img_ulb_w.clone()

                # 2. 임시 라벨 준비 (Teacher가 방금 만든 따끈따끈한 라벨을 복사해옵니다)
                temp_pseudo_labels = pseudo_outputs.clone()
                temp_pseudo_logits = pseudo_logits.clone()

                # 3. Copy-Paste 적용 (Clean 배경 + Clean 용종)
                if aug_config['cp_prob'] > 0:
                    img_ulb_s, temp_pseudo_labels, temp_pseudo_logits, cp_count, pasted_flags_A1 = simple_color_copy_paste(
                        img_ulb_s, 
                        temp_pseudo_labels, 
                        temp_pseudo_logits, 
                        img_lb_w, 
                        target_lb,
                        paste_prob=aug_config['cp_prob'],
                        color_threshold=args.get("color_threshold", 0.6)
                    )
                    
                    # 🔥 [핵심] 변경된 라벨을 메인 변수에 '업데이트' 합니다.
                    # 이제 시각화 함수도, Loss 계산도 이 '새 라벨'을 보게 됩니다.
                    pseudo_outputs = temp_pseudo_labels
                    pseudo_logits = temp_pseudo_logits

                else:
                    copypaste_applied = False

                # 4. 수동 Jitter 적용 (톤 일치 작업)
                manual_strong_aug.to(img_ulb_s.device)
                with torch.no_grad():
                    img_ulb_s = manual_strong_aug(img_ulb_s)
            
            # ==================================================================    

            # 3) apply cutmix and copy-paste
            img_ulb_s_original = img_ulb_s.clone()  # Store original for comparison
            # Store pseudo labels BEFORE augmentation for visualization
            #pseudo_outputs_before_aug = pseudo_outputs.clone()
            #pseudo_logits_before_aug = pseudo_logits.clone()
            # Store GT BEFORE augmentation for visualization
            target_ulb_before_aug = target_ulb.clone() if 'target_ulb' in locals() else None

            # 🔥 EXP3: Use current augmentation config (2-stage: CP → CutMix) (updated only during validation)
            # aug_config is updated by step_dice() during validation
            aug_config = aug_scheduler.stages[aug_scheduler.current_stage]

            if flag_start_self_learning: # strongest augs (A2)
                img_ulb_s, pseudo_outputs, pseudo_logits = cut_mix(
                                img_ulb_s,
                                pseudo_outputs,
                                pseudo_logits
                )
                cutmix_applied = True

                # Save A2 sample if needed

            else:
                if alternate_indicator.get_alternate_state() == False:  # A2 턴 (학생 훈련)
                    
                    # [상황 1] CutMix 단계 (Stage 2)
                    if current_stage_name == 'cutmix':
                        del img_ulb_s
                        img_ulb_s = img_ulb_w.clone() # 리셋 (깨끗한 상태로)

                        # CutMix 적용
                        if np.random.random() < aug_config['cutmix_prob']:
                             if mtx_bool_conflict is None:
                                img_ulb_s, pseudo_outputs, pseudo_logits = cut_mix(img_ulb_w, pseudo_outputs, pseudo_logits)
                             else:
                                img_ulb_s, pseudo_outputs, pseudo_logits, mtx_bool_conflict = cut_mix(img_ulb_w, pseudo_outputs, pseudo_logits, mtx_bool_conflict)
                             cutmix_applied = True
                             conflict_ratio = mtx_bool_conflict.float().sum() / num_ulb
                        else:
                             cutmix_applied = False

                    # ------------------------------------------------------------------
                    # [상황 2] Copy-Paste 단계 (Stage 1) - Student Variant (A2)
                    # ------------------------------------------------------------------
                    elif current_stage_name == 'copy_paste':
                        # 1. A1(Teacher)에서 만든 Jitter 이미지는 버리고, 깨끗한 원본으로 리셋
                        del img_ulb_s
                        img_ulb_s = img_ulb_w.clone()
                        
                        # 라벨도 깨끗한 원본(백업본)으로 되돌리기
                        pseudo_outputs = pseudo_outputs_before_aug.clone()
                        pseudo_logits = pseudo_logits_before_aug.clone()

                        # 2. Copy-Paste 다시 적용 (Jitter 없이!)
                        if aug_config['cp_prob'] > 0:
                             # 🚨 [중요 변경]
                             # 1) 'if target_ulb' 조건문 삭제 (구형 함수 제거)
                             # 2) 무조건 신형 함수 사용
                             # 3) 결과는 A2 전용 변수 'pasted_flags_A2'에 저장 (A1과 분리!)
                             img_ulb_s, pseudo_outputs, pseudo_logits, cp_count, pasted_flags_A2 = simple_color_copy_paste(
                                img_ulb_s, pseudo_outputs, pseudo_logits, img_lb_w, target_lb,
                                paste_prob=aug_config['cp_prob'],
                                color_threshold=args.get("color_threshold", 0.6)
                             )
                             
                             copypaste_applied = (cp_count > 0)
                        else:
                             copypaste_applied = False
                             # pasted_flags_A2는 루프 시작 시 [False]로 초기화했으므로 건드릴 필요 없음                                
                        
                        

            # 4) forward
            img = torch.cat((img_lb_w, img_ulb_w, img_ulb_s))
            pred = model(img)
            pred_lb = pred[:args["labeled_bs"]]
            pred_ulb_w, pred_ulb_s = pred[args["labeled_bs"]:].chunk(2)
            
            # Get student prediction on original weak augmented images
            with torch.no_grad():
                pred_ulb_w_stu_soft = torch.softmax(pred_ulb_w.detach(), dim=1)
                pseudo_logits_stu, pseudo_outputs_stu = torch.max(pred_ulb_w_stu_soft, dim=1)

            # TARGET IMAGES TRACKING (5 targets)
            # Track multiple target images for higher discovery probability
            found, batch_pos, target_info = track_target_images(sampled_batch, num_ulb)
            if found:
                # ------------------------------------------------------------------
                # [수정 4] 시각화할 때 A1 결과인지 A2 결과인지 판단하는 로직
                # ------------------------------------------------------------------
                
                # 1. 기본적으로는 A1(Teacher/Common) 결과를 봅니다.
                # (루프 시작할 때 pasted_flags_A1 = [False]*bs 로 초기화했으므로 안전함)
                current_flags = pasted_flags_A1
                
                # 2. 하지만 A2(Student Variant) 구간으로 들어왔다면 A2 결과를 봐야 합니다.
                # alternate_indicator가 False이면 A2(학생) 턴입니다.
                if alternate_indicator.get_alternate_state() == False:
                     # Copy-Paste 단계라면 A2 플래그 사용 (A2 변수도 위에서 A1과 분리했죠?)
                     if current_stage_name == 'copy_paste':
                         current_flags = pasted_flags_A2
                     # CutMix 단계라면 CP는 무조건 False
                     elif current_stage_name == 'cutmix':
                         current_flags = [False] * args['labeled_bs'] # 길이는 적절히 맞춰줌

                # 3. 내 번호(batch_pos)에 해당하는 값 가져오기
                if batch_pos < len(current_flags):
                    is_target_pasted = current_flags[batch_pos]
                else:
                    is_target_pasted = False
                # -------------------------------------------------------

                save_target_image_visualization(
                    iter_num=iter_num,
                    target_info=target_info,
                    batch_pos=batch_pos,
                    img_ulb_w=img_ulb_w,
                    img_ulb_s=img_ulb_s,
                    img_ulb_s_original=img_ulb_s_original,
                    ema_outputs_soft_1=ema_outputs_soft_1_vis,
                    ema_outputs_soft_2=ema_outputs_soft_2_vis,
                    weighted_outputs=weighted_outputs_vis,
                    pseudo_outputs=pseudo_outputs,
                    pseudo_logits=pseudo_logits,
                    pseudo_outputs_before_aug=pseudo_outputs_before_aug.clone(),  # 증강 전 수도라벨 추가
                    pseudo_logits_before_aug=pseudo_logits_before_aug,    # 증강 전 신뢰도 추가
                    pred_ulb_w=pred_ulb_w_stu_soft,
                    pred_ulb_s=pred_ulb_s,
                    mtx_bool_conflict=mtx_bool_conflict,
                    copypaste_applied=is_target_pasted,  # 👈 [수정] 진짜 성공 여부 전달!
                    cutmix_applied=cutmix_applied,
                    alternate_state=alternate_indicator.get_alternate_state(),
                    snapshot_path=snapshot_path,
                    args=args,
                    target_gt=target_ulb,  # GT 라벨 (증강 후)
                    target_gt_before_aug=target_ulb_before_aug if 'target_ulb_before_aug' in locals() else None  # GT 라벨 (증강 전)
                )
            
            # 5) supervised loss
            loss_lb = (ce_loss(pred_lb, target_lb.long()) +
                        dice_loss(torch.softmax(pred_lb, dim=1),
                                target_lb.unsqueeze(1).float(), 
                                ignore=torch.zeros_like(target_lb).float())
                        ) / 2.0
            
            # 6) unsupervised loss
            pseudo_mask = pseudo_logits.ge(args["conf_threshold"]).bool()
            high_ratio = pseudo_mask.float().mean()
            if "dice" == args["flag_ulb_loss_type"]:
                if flag_start_self_learning:
                    loss_ulb = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                    pseudo_outputs.unsqueeze(1).float(),
                                    ignore=(pseudo_logits < args["alt_param_threshold_self_training"]).float())
                else:
                    # CCM 적용 전 teacher ensemble 예측 저장 (시각화용)
                    teacher_ensemble_pred = pseudo_outputs.clone()
                    
                    pseudo_outputs, pseudo_logits, conflict_tea_stu = get_compromise_pseudo_btw_tea_stu(pseudo_outputs, pseudo_logits, 
                                                                                        pseudo_outputs_stu, pseudo_logits_stu, 
                                                                                        alt_flag_conflict_mode, 
                                                                                        None if alt_flag_conflict_stu_use_more else mtx_bool_conflict)
                    

                    if var_param_conflict_weight > 1 or var_param_conflict_weight < 1:
                        pseudo_logits_consist = pseudo_logits.clone()
                        pseudo_logits_consist[mtx_bool_conflict] = -0.1
                        loss_ulb_consist = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                                        pseudo_outputs.unsqueeze(1).float(),
                                                        ignore=(pseudo_logits_consist < args["conf_threshold"]).float())
                        if var_param_conflict_weight > 0 or var_param_conflict_weight < 0:
                            # print("#"*20, var_param_conflict_weight, pseudo_outputs.requires_grad, pseudo_logits_consist.requires_grad)
                            pseudo_logits_conflict = pseudo_logits.clone()
                            pseudo_logits_conflict[~mtx_bool_conflict] = -0.1
                            loss_ulb_conflict = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                                            pseudo_outputs.unsqueeze(1).float(),
                                                            ignore=(pseudo_logits_conflict < args["conf_threshold"]).float())
                            if loss_ulb_conflict > 0.0:
                                loss_ulb = loss_ulb_consist + var_param_conflict_weight * loss_ulb_conflict
                            else:
                                # print("-"*100, loss_ulb_conflict.item(), loss_ulb_consist.item())
                                loss_ulb = loss_ulb_consist * 1.0
                        else:
                            loss_ulb = loss_ulb_consist * 1.0

                    else:
                        loss_ulb = dice_loss(torch.softmax(pred_ulb_s, dim=1),
                                        pseudo_outputs.unsqueeze(1).float(),
                                        ignore=(pseudo_logits < args["conf_threshold"]).float())
            else:
                pseudo_outputs[~pseudo_mask] = -100
                loss_ulb = F.cross_entropy(pred_ulb_s, pseudo_outputs.long(), ignore_index=-100, reduction="mean")

            # All tracking disabled
            pass

            # All visualizations disabled
            pass

            # 7) total loss
            consistency_weight = get_current_consistency_weight(iter_num//150, args)
            loss = loss_lb + consistency_weight * loss_ulb

            # 8) update student model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EXP4: No loss tracking needed (Dice-based switching)

            # 9) update teacher model
            if not flag_start_self_learning:
                if alternate_indicator.get_alternate_state():
                    update_ema_variables(model, ema_model, args["ema_decay"], iter_num//2, args)
                else:
                    update_ema_variables(model, ema_model_another, args["ema_decay"], iter_num//2, args)
                
                alternate_indicator.update()
            else:
                update_ema_variables(model, ema_model, args["ema_decay"], iter_num, args)

            # 10) udpate learing rate
            if args["poly"]:
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = base_lr


            # 11) record statistics
            iter_num = iter_num + 1
            # --- a) writer
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_lb', loss_lb, iter_num)
            writer.add_scalar('info/loss_ulb', loss_ulb, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('info/conflict_ratio', conflict_ratio.item(), iter_num)
            writer.add_scalar('info/copypaste_applied', 1.0 if copypaste_applied else 0.0, iter_num)
            writer.add_scalar('info/cutmix_applied', 1.0 if cutmix_applied else 0.0, iter_num)
            # 🔥 Adaptive Augmentation Logging
            writer.add_scalar('info/aug_stage', aug_scheduler.current_stage, iter_num)
            writer.add_scalar('info/aug_cp_prob', aug_config['cp_prob'], iter_num)
            writer.add_scalar('info/aug_cutmix_prob', aug_config['cutmix_prob'], iter_num)

            # [수정된 코드] Best Dice (Student, Teacher) 정보 추가
            logging.info("iteration:{}  t-loss:{:.4f}, loss-lb:{:.4f}, loss-ulb:{:.4f}, conflict/consist:{:.4f}/{:.4f}, weight:{:.2f}, high-r:{:.2f}, conflic:{}, cp:{}, cm:{}, lr:{:.4f}, alt:{}, aug_stage:{}, Best-S:{:.4f}, Best-T:{:.4f}".format(iter_num,
            loss.item(), loss_lb.item(), loss_ulb.item(), loss_ulb_conflict.item(), loss_ulb_consist.item(),
            consistency_weight, high_ratio, int(conflict_ratio.item()), int(copypaste_applied), int(cutmix_applied), lr_, alternate_indicator.get_alternate_state(), aug_config['name'], best_performance_stu, best_performance))
            # --- c) avg meters
            meter_sup_losses.update(loss_lb.item())
            meter_uns_losses.update(loss_ulb.item())
            meter_train_losses.update(loss.item())
            meter_highc_ratio.update(high_ratio.item())
            meter_learning_rates.update(lr_)
            meter_conflict_ratio.update(conflict_ratio.item())
            meter_uns_losses_consist.update(loss_ulb_consist.item())
            meter_uns_losses_conflict.update(loss_ulb_conflict.item())
            meter_copypaste_ratio.update(1.0 if copypaste_applied else 0.0)
            meter_cutmix_ratio.update(1.0 if cutmix_applied else 0.0)


            # --- e) csv
            tmp_results = {
                        'loss_total': loss.item(),
                        'loss_lb': loss_lb.item(),
                        'loss_ulb': loss_ulb.item(),
                        'loss_ulb_conflict': loss_ulb_conflict.item(),
                        'loss_ulb_consist': loss_ulb_consist.item(),
                        'lweight_ub': consistency_weight,
                        'high_ratio': high_ratio.item(),
                        'conflict_ratio': conflict_ratio.item(),
                        'copypaste_applied': 1.0 if copypaste_applied else 0.0,
                        'cutmix_applied': 1.0 if cutmix_applied else 0.0,
                        "lr":lr_}
            data_frame = pd.DataFrame(data=tmp_results, index=range(iter_num, iter_num+1))
            if iter_num > 1 and osp.exists(csv_train):
                data_frame.to_csv(csv_train, mode='a', header=None, index_label='iter')
            else:
                data_frame.to_csv(csv_train, index_label='iter')

            if iter_num >= max_iterations:
                break

        # 12) validating
        if epoch_num % args.get("test_interval_ep", 10) == 0 or iter_num >= max_iterations:
            model.eval()
            ema_model.eval()
            ema_model_another.eval()

            metric_list = 0.0
            ema_metric_list = 0.0
            ema_metric_list = 0.0
            ema_metric_another_list = 0.0

            for _, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume(
                    sampled_batch["image"], 
                    sampled_batch["label"], 
                    model, 
                    classes=num_classes)
                metric_list += np.array(metric_i)

                ema_metric_i = test_single_volume(
                    sampled_batch["image"], 
                    sampled_batch["label"], 
                    ema_model, 
                    classes=num_classes)
                ema_metric_list += np.array(ema_metric_i)

                if not flag_start_self_learning:
                    ema_another_metric_i = test_single_volume(
                        sampled_batch["image"], 
                        sampled_batch["label"], 
                        ema_model_another, 
                        classes=num_classes)
                    ema_metric_another_list += np.array(ema_another_metric_i)

            metric_list = metric_list / len(db_val)
            ema_metric_list = ema_metric_list / len(db_val)
            if not flag_start_self_learning:
                ema_metric_another_list = ema_metric_another_list / len(db_val)

            for class_i in range(num_classes-1):
                writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], epoch_num)
                writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], epoch_num)

                writer.add_scalar('info/ema_val_{}_dice'.format(class_i+1), ema_metric_list[class_i, 0], epoch_num)
                writer.add_scalar('info/ema_val_{}_hd95'.format(class_i+1), ema_metric_list[class_i, 1], epoch_num)

            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]
            writer.add_scalar('info/val_mean_dice', performance, epoch_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, epoch_num)

            ema_performance = np.mean(ema_metric_list, axis=0)[0]
            ema_mean_hd95 = np.mean(ema_metric_list, axis=0)[1]
            writer.add_scalar('info/ema_val_mean_dice', ema_performance, epoch_num)
            writer.add_scalar('info/ema_val_mean_hd95', ema_mean_hd95, epoch_num)

            if not flag_start_self_learning:
                ema_another_performance = np.mean(ema_metric_another_list, axis=0)[0]
                ema_another_mean_hd95 = np.mean(ema_metric_another_list, axis=0)[1]
            else:
                ema_another_performance = 0.0
                ema_another_mean_hd95 = 0.0
            writer.add_scalar('info/ema_another_val_mean_dice', ema_another_performance, epoch_num)
            writer.add_scalar('info/ema_another_val_mean_hd95', ema_another_mean_hd95, epoch_num)

            # 🔥 Best Dice 추이 기록
            writer.add_scalar('info/best_dice_student', best_performance_stu, epoch_num)
            writer.add_scalar('info/best_dice_teacher', best_performance, epoch_num)
            writer.add_scalar('info/best_dice_teacher_another', best_performance_another, epoch_num)

            if performance > best_performance_stu:
                best_performance_stu = performance
                tmp_stu_snapshot_path = os.path.join(snapshot_path, "student")
                if not os.path.exists(tmp_stu_snapshot_path):
                    os.makedirs(tmp_stu_snapshot_path,exist_ok=True)
                save_mode_path_stu = os.path.join(tmp_stu_snapshot_path, 'ep_{:0>3}_dice_{}.pth'.format(epoch_num, round(best_performance_stu, 4)))
                torch.save(model.state_dict(), save_mode_path_stu)

                save_best_path_stu = os.path.join(snapshot_path,'{}_best_stu_model.pth'.format(args["model"]))
                torch.save(model.state_dict(), save_best_path_stu)
                logging.info("New best student model! Dice: {:.4f}, saved to {}".format(best_performance_stu, save_best_path_stu))

            # [EXP5] Update augmentation strategy based on Student Dice Score (2-stage) based on Student Dice Score
            current_aug_config = aug_scheduler.step_dice(performance, iter_num)
            logging.info("[Validation] Dice: {:.4f} | Aug Stage: {} | CP:{:.1f} CM:{:.1f}".format(
                performance, aug_scheduler.get_current_stage_name(),
                current_aug_config['cp_prob'], current_aug_config['cutmix_prob']))


            if ema_performance > best_performance:
                best_performance = ema_performance
                tmp_tea_snapshot_path = os.path.join(snapshot_path, "teacher")
                if not os.path.exists(tmp_tea_snapshot_path):
                    os.makedirs(tmp_tea_snapshot_path,exist_ok=True)
                save_mode_path = os.path.join(tmp_tea_snapshot_path, 'ep_{:0>3}_dice_{}.pth'.format(epoch_num, round(best_performance, 4)))
                torch.save(ema_model.state_dict(), save_mode_path)

                save_best_path = os.path.join(snapshot_path,'{}_best_tea_model.pth'.format(args["model"]))
                torch.save(ema_model.state_dict(), save_best_path)
                logging.info("New best teacher model! Dice: {:.4f}, saved to {}".format(best_performance, save_best_path))
            
            if ema_another_performance > best_performance_another:
                best_performance_another = ema_another_performance

            # csv
            tmp_results_ts = {
                    'loss_total': meter_train_losses.avg,
                    'loss_sup': meter_sup_losses.avg,
                    'loss_unsup': meter_uns_losses.avg,
                    'loss_unsup_consist': meter_uns_losses_consist.avg,
                    'loss_unsup_conflict': meter_uns_losses_conflict.avg,
                    'avg_high_ratio': meter_highc_ratio.avg,
                    'avg_conflict_ratio': meter_conflict_ratio.avg,
                    'avg_copypaste_ratio': meter_copypaste_ratio.avg,  # Copy-Paste 적용 비율
                    'learning_rate': meter_learning_rates.avg,
                    'Dice_tea': ema_performance,
                    'Dice_tea_best': best_performance,
                    'Dice_tea_another': ema_another_performance,
                    'Dice_tea_another_best': best_performance_another,
                    'Dice_stu': performance,
                    'Dice_stu_best': best_performance_stu}
            data_frame = pd.DataFrame(data=tmp_results_ts, index=range(epoch_num, epoch_num+1))
            if epoch_num > 0 and osp.exists(csv_test):
                data_frame.to_csv(csv_test, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_test, index_label='epoch')

            # logs
            logging.info(" <<Test>> - Ep:{}  - mean_dice/mean_h95 - S:{:.2f}/{:.2f}, Best-S:{:.2f}, T:{:.2f}/{:.2f}, Best-T:{:.2f}, T-a:{:.2f}/{:.2f}, Best-T-a:{:.2f} - CP-ratio:{:.2f}".format(epoch_num, 
                    performance*100, mean_hd95, best_performance_stu*100, ema_performance*100, ema_mean_hd95, best_performance*100, ema_another_performance*100, ema_another_mean_hd95, best_performance_another*100, meter_copypaste_ratio.avg))
            logging.info("          - AvgLoss(lb/ulb/all):{:.4f}/{:.4f}/{:.4f}, AvgConflict/consist:{:.4f}/{:.4f} highR:{:.2f}, conflict:{:.2f} ".format( 
                    meter_sup_losses.avg, meter_uns_losses.avg, meter_train_losses.avg, 
                    meter_uns_losses_conflict.avg, meter_uns_losses_consist.avg,
                    meter_highc_ratio.avg, meter_conflict_ratio.avg,
                    ))
            
            model.train()
            ema_model.train()
            ema_model_another.train()

        # Remove epoch-based saving - only save best models in validation section
        
        if iter_num >= max_iterations:
            # Save final models
            save_final_path_stu = os.path.join(snapshot_path, '{}_final_stu_model.pth'.format(args["model"]))
            torch.save(model.state_dict(), save_final_path_stu)
            logging.info("Saved final student model to {}".format(save_final_path_stu))

            save_final_path_tea = os.path.join(snapshot_path, '{}_final_tea_model.pth'.format(args["model"]))
            torch.save(ema_model.state_dict(), save_final_path_tea)
            logging.info("Saved final teacher model to {}".format(save_final_path_tea))

            # 🔥 Save augmentation switch history
            if len(aug_scheduler.switch_history) > 0:
                switch_history_path = os.path.join(snapshot_path, 'augmentation_switch_history.csv')
                switch_df = pd.DataFrame(aug_scheduler.switch_history)
                switch_df.to_csv(switch_history_path, index=False)
                logging.info("Saved augmentation switch history to {}".format(switch_history_path))
                logging.info("Total augmentation switches: {}".format(len(aug_scheduler.switch_history)))
                for switch in aug_scheduler.switch_history:
                    logging.info("  - Iter {}: {} → {} (avg_loss={:.4f})".format(
                        switch['iteration'], switch['from_stage'], switch['to_stage'], switch['avg_loss']))
            else:
                logging.info("No augmentation switches occurred during training")

            iterator.close()
            break
    
    # All tracking disabled
    pass
    
    writer.close()
    return "Training Finished!"


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        III. main process
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if __name__ == "__main__":
    # 1. set up config
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', type=str, default='',
                        help='configuration file')
    
    # Basics: Data, results, model
    parser.add_argument('--root_path', type=str,
                        default='./', help='Name of Experiment')
    parser.add_argument('--res_path', type=str, 
                        default='./results/ACDC', help='Path to save resutls')
    parser.add_argument('--exp', type=str,
                        default='ACDC/POST-NoT', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--num_classes', type=int,  default=4,
                        help='output channel of network')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of gpu used to train the model')
    
    # Training Basics
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input')
    parser.add_argument("--deterministic", action='store_true', 
                        help="whether use deterministic training")
    parser.add_argument('--seed', type=int,  default=2023, help='random seed')
    parser.add_argument('--test_interval_ep', type=int,
                        default=1, help='')
    parser.add_argument('--save_interval_epoch', type=int,
                        default=1000000, help='')
    parser.add_argument("-p", "--poly", default=False, 
                        action='store_true', help="whether poly scheduler")
    

    # label and unlabel
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=12,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=136,
                        help='labeled data')
    
    # model related
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument("--flag_pseudo_from_student", default=False, 
                        action='store_true', help="using pseudo from student itself")
    
    # augmentation
    parser.add_argument('--cutmix_prob', type=float,  
                        default=0.5, help='probability of applying cutmix')
    parser.add_argument('--copy_paste_prob', type=float,
                        default=0.5, help='probability of applying copy-paste')  # NEW
    parser.add_argument('--aug_patience', type=int,
                        default=5, help='EXP4: patience for Dice-based adaptive augmentation (validations)')
    parser.add_argument('--aug_min_delta', type=float,
                        default=0.01, help='EXP4: minimum Dice improvement to reset patience (1%)')
    parser.add_argument('--color_threshold', type=float,
                        default=0.6, help='color similarity threshold for copy-paste')
    parser.add_argument('--color_method', type=str,
                        default='rgb_distance', help='color similarity calculation method')

    # unlabeled loss
    parser.add_argument('--consistency', type=float,
                        default=1.0, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=150.0, help='consistency_rampup')
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.95,
        help="confidence threshold for using pseudo-labels",
    )
    parser.add_argument('--flag_ulb_loss_type', type=str,
                        default="dice", help='loss type, ce, dice, dice+ce')
    parser.add_argument("--flag_sampling_based_on_lb", 
                        default=False, action='store_true', help="using dynamic cutmix")
    
    # parse args
    args = parser.parse_args()
    args = vars(args)

    # 2. update from the config files
    cfgs_file = args['cfg']
    if not os.path.isabs(cfgs_file) and not cfgs_file.startswith('../'):
        cfgs_file = os.path.join('./cfgs',cfgs_file)
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    # convert "1e-x" to float
    for each in options_yaml.keys():
        tmp_var = options_yaml[each]
        if type(tmp_var) == str and "1e-" in tmp_var:
            options_yaml[each] = float(tmp_var)
    # update original parameters of argparse
    update_values(options_yaml, args)

    # Override with command line arguments again (명령행 인자를 config 파일 값보다 우선시)
    cmd_args = parser.parse_args()
    for arg_name, arg_value in vars(cmd_args).items():
        # argparse의 default 값이 아닌 실제 명령행에서 설정된 값만 덮어쓰기
        if arg_name in ['labeled_num', 'copy_paste_prob', 'cutmix_prob', 'exp', 'root_path', 'model', 'color_threshold', 'color_method']:
            args[arg_name] = arg_value

    # print confg information
    import pprint
    # print("{}".format(pprint.pformat(args)))
    # assert 1==0, "break here"

    # 3. setup gpus and randomness
    # if args["gpu_id"] in range(8):
    if args["gpu_id"] in range(10):
        gid = args["gpu_id"]
    else:
        gid = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

    if not args["deterministic"]:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args["seed"] > 0:
        random.seed(args["seed"])
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])

    # 4. outputs and logger
    snapshot_path = "{}/{}_{}_labeled/{}".format(
        args["res_path"], args["exp"], args["labeled_num"], args["model"])
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info(str(args))
    logging.info("{}".format(pprint.pformat(args)))

    train(args, snapshot_path)