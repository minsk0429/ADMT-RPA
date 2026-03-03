import argparse
import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from networks.net_factory import net_factory
from dataloaders.dataset_2d import KvasirDataSets
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='C:/ai-agent/data/Kvasir', help='Path to dataset')
parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--save_csv', action='store_true', help='Save detailed results to CSV')
parser.add_argument('--save_predictions', action='store_true', help='Save prediction images')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for predictions')
parser.add_argument('--compare_dice', action='store_true', help='Compare old (buggy) vs new (fixed) Dice calculation')

def calculate_metric_percase_old(pred, gt):
    """
    기존 (버그 있는) 메트릭 계산
    - 버그: pred.sum() == 0 일 때 무조건 Dice=1 반환
    - 문제: GT에 폴립이 있는데 예측이 비어있으면 Dice=0이어야 함
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    elif pred.sum() == 0:
        return 1, 1, 0, 0  # BUG: returns 1 regardless of GT
    else:
        return 0, 0, 0, 0


def calculate_metric_percase_new(pred, gt):
    """
    수정된 메트릭 계산
    - pred > 0, gt > 0: 실제 Dice 계산
    - pred == 0, gt == 0: 둘 다 비어있음 → Dice=1 (True Negative)
    - pred > 0, gt == 0: 예측만 있음 → Dice=0 (False Positive)
    - pred == 0, gt > 0: GT만 있음 → Dice=0 (False Negative)
    """
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum > 0 and gt_sum > 0:
        # 둘 다 있음 → 실제 계산
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    elif pred_sum == 0 and gt_sum == 0:
        # 둘 다 비어있음 → True Negative
        return 1, 1, 0, 0
    else:
        # 하나만 있음 → 완전 불일치
        return 0, 0, 0, 0


def calculate_metric_percase(pred, gt):
    """기본 호출 시 수정된 버전 사용"""
    return calculate_metric_percase_new(pred, gt)

def test_single_case(net, image, label, patch_size=[256, 256], return_pred=False, compare_mode=False):
    """
    단일 케이스 테스트

    Args:
        compare_mode: True일 경우 Old/New 메트릭 둘 다 반환

    Returns:
        compare_mode=False: (dice, jc, hd95, asd)
        compare_mode=True: ((old_metrics), (new_metrics))
    """
    # image: (C, H, W), label: (H, W)
    c, h, w = image.shape

    # Resize image
    image_zoom = zoom(image, (1, patch_size[0] / h, patch_size[1] / w), order=0)
    input = torch.from_numpy(image_zoom).unsqueeze(0).float().cuda()  # (1, C, H, W)

    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
        # Resize prediction back to original size
        prediction = zoom(prediction, (h / patch_size[0], w / patch_size[1]), order=0)

    if compare_mode:
        # Old/New 둘 다 계산 (copy 사용하여 원본 보존)
        old_metrics = calculate_metric_percase_old(prediction.copy(), label.copy())
        new_metrics = calculate_metric_percase_new(prediction.copy(), label.copy())
        if return_pred:
            return old_metrics, new_metrics, prediction
        return old_metrics, new_metrics
    else:
        metrics = calculate_metric_percase(prediction.copy(), label.copy())
        if return_pred:
            return metrics, prediction
        return metrics


def save_prediction_images(image, label, prediction, case_name, output_dir, dice_score):
    """예측 결과 이미지 저장 (원본, GT, 예측 3분할)"""
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 정규화 (C, H, W) -> (H, W, C)
    if image.shape[0] == 3:
        img_display = np.transpose(image, (1, 2, 0))
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
    else:
        img_display = image

    # 3분할 이미지 생성 (원본, GT, 예측)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. 원본 이미지
    axes[0].imshow(img_display)
    axes[0].set_title('Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. Ground Truth
    axes[1].imshow(label, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 3. Prediction
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title(f'Prediction (Dice: {dice_score:.4f})', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{case_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def inference(args):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Create network
    net = net_factory(net_type=args.model, in_chns=3, class_num=args.num_classes)

    # Load model
    net.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")
    net.eval()
    net.cuda()

    # Create test dataset
    db_test = KvasirDataSets(base_dir=args.root_path, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    print(f"total {len(testloader)} samples")

    # Compare mode 안내
    if args.compare_dice:
        print("\n" + "="*60)
        print("[COMPARE MODE] Old (buggy) vs New (fixed) Dice calculation")
        print("="*60)
        print("Old: pred.sum()==0 -> Dice=1 (bug: ignores GT)")
        print("New: pred.sum()==0 and gt.sum()>0 -> Dice=0 (correct)")
        print("="*60 + "\n")

    # Output directory 설정
    if args.save_predictions:
        if args.output_dir:
            output_dir = args.output_dir
        else:
            model_dir = os.path.dirname(args.model_path)
            output_dir = os.path.join(model_dir, 'test_results')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving predictions to: {output_dir}")

    # Test
    old_metric_list = []
    new_metric_list = []
    metric_list = []
    case_names = []
    diff_cases = []  # Old/New 차이나는 케이스 기록

    saved_count = 0
    max_save = 20  # 최대 20장만 저장

    for i_batch, sampled_batch in enumerate(tqdm(testloader)):
        image = sampled_batch["image"].squeeze(0).cpu().numpy()  # (C, H, W)
        label = sampled_batch["label"].squeeze(0).cpu().numpy()  # (H, W)
        case_name = sampled_batch["idx"][0] if "idx" in sampled_batch else f"case_{i_batch:04d}"

        if args.compare_dice:
            # Old/New 둘 다 계산
            if args.save_predictions and saved_count < max_save:
                old_metric, new_metric, prediction = test_single_case(
                    net, image, label, return_pred=True, compare_mode=True)
                save_prediction_images(image, label, prediction, case_name, output_dir, new_metric[0])
                saved_count += 1
            else:
                old_metric, new_metric = test_single_case(
                    net, image, label, compare_mode=True)

            old_metric_list.append(old_metric)
            new_metric_list.append(new_metric)

            # 차이나는 케이스 기록
            if abs(old_metric[0] - new_metric[0]) > 0.001:
                diff_cases.append({
                    'case': case_name,
                    'old_dice': old_metric[0],
                    'new_dice': new_metric[0],
                    'diff': old_metric[0] - new_metric[0]
                })
        else:
            if args.save_predictions and saved_count < max_save:
                metric_i, prediction = test_single_case(net, image, label, return_pred=True)
                save_prediction_images(image, label, prediction, case_name, output_dir, metric_i[0])
                saved_count += 1
            else:
                metric_i = test_single_case(net, image, label)
            metric_list.append(metric_i)

        case_names.append(case_name)

    if args.compare_dice:
        old_metric_list = np.array(old_metric_list)
        new_metric_list = np.array(new_metric_list)

        # Old 메트릭
        old_mean_dice = np.mean(old_metric_list[:, 0])
        old_mean_jc = np.mean(old_metric_list[:, 1])

        # New 메트릭
        new_mean_dice = np.mean(new_metric_list[:, 0])
        new_mean_jc = np.mean(new_metric_list[:, 1])
        new_mean_hd95 = np.mean(new_metric_list[:, 2])
        new_mean_asd = np.mean(new_metric_list[:, 3])

        print(f"\n{'='*60}")
        print("Test Results Comparison")
        print(f"{'='*60}")
        print(f"\n[OLD] Buggy Dice Calculation:")
        print(f"   Mean Dice:    {old_mean_dice:.4f}")
        print(f"   Mean Jaccard: {old_mean_jc:.4f}")

        print(f"\n[NEW] Fixed Dice Calculation:")
        print(f"   Mean Dice:    {new_mean_dice:.4f}")
        print(f"   Mean Jaccard: {new_mean_jc:.4f}")
        print(f"   Mean HD95:    {new_mean_hd95:.4f}")
        print(f"   Mean ASD:     {new_mean_asd:.4f}")

        print(f"\n[DIFF] Difference (Old - New):")
        print(f"   Dice diff:    {old_mean_dice - new_mean_dice:+.4f}")
        print(f"   Jaccard diff: {old_mean_jc - new_mean_jc:+.4f}")

        if diff_cases:
            print(f"\n[WARNING] Cases with different Dice scores ({len(diff_cases)} cases):")
            for case in diff_cases[:10]:  # 최대 10개만 표시
                print(f"   {case['case']}: Old={case['old_dice']:.4f}, New={case['new_dice']:.4f}, Diff={case['diff']:+.4f}")
            if len(diff_cases) > 10:
                print(f"   ... and {len(diff_cases) - 10} more cases")
        print(f"{'='*60}\n")

        # return용 메트릭 (new 버전 사용)
        mean_dice = new_mean_dice
        mean_jc = new_mean_jc
        mean_hd95 = new_mean_hd95
        mean_asd = new_mean_asd
        metric_list = new_metric_list
    else:
        metric_list = np.array(metric_list)

        # Calculate mean metrics
        mean_dice = np.mean(metric_list[:, 0])
        mean_jc = np.mean(metric_list[:, 1])
        mean_hd95 = np.mean(metric_list[:, 2])
        mean_asd = np.mean(metric_list[:, 3])

        print(f"\nTest Results:")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Mean Jaccard: {mean_jc:.4f}")
        print(f"Mean HD95: {mean_hd95:.4f}")
        print(f"Mean ASD: {mean_asd:.4f}")
    
    # Save results if requested
    if args.save_csv:
        model_dir = os.path.dirname(args.model_path)

        if args.compare_dice:
            # Compare 모드: Old/New 둘 다 저장
            results_df = pd.DataFrame({
                'case': case_names,
                'dice_old': old_metric_list[:, 0],
                'dice_new': new_metric_list[:, 0],
                'dice_diff': old_metric_list[:, 0] - new_metric_list[:, 0],
                'jaccard_old': old_metric_list[:, 1],
                'jaccard_new': new_metric_list[:, 1],
                'hd95': new_metric_list[:, 2],
                'asd': new_metric_list[:, 3]
            })
            csv_path = os.path.join(model_dir, 'test_metrics_compare.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Saved comparison results to {csv_path}")

            # Summary 저장
            summary = {
                'OLD Mean Dice': old_mean_dice,
                'OLD Std Dice': np.std(old_metric_list[:, 0]),
                'NEW Mean Dice': new_mean_dice,
                'NEW Std Dice': np.std(new_metric_list[:, 0]),
                'Dice Difference': old_mean_dice - new_mean_dice,
                'OLD Mean Jaccard': old_mean_jc,
                'NEW Mean Jaccard': new_mean_jc,
                'NEW Mean HD95': new_mean_hd95,
                'NEW Mean ASD': new_mean_asd,
                'Cases with difference': len(diff_cases),
            }

            summary_path = os.path.join(model_dir, 'test_summary_compare.txt')
            with open(summary_path, 'w') as f:
                f.write("=" * 50 + "\n")
                f.write("Old vs New Dice Calculation Comparison\n")
                f.write("=" * 50 + "\n\n")
                for key, value in summary.items():
                    if isinstance(value, int):
                        f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")
            print(f"Saved comparison summary to {summary_path}")
        else:
            # 일반 모드: 기존 방식
            results_df = pd.DataFrame({
                'case': case_names,
                'dice': metric_list[:, 0],
                'jaccard': metric_list[:, 1],
                'hd95': metric_list[:, 2],
                'asd': metric_list[:, 3]
            })
            csv_path = os.path.join(model_dir, 'test_metrics.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Saved detailed results to {csv_path}")

            # Save summary
            summary = {
                'Mean Dice': mean_dice,
                'Std Dice': np.std(metric_list[:, 0]),
                'Mean Jaccard': mean_jc,
                'Std Jaccard': np.std(metric_list[:, 1]),
                'Mean HD95': mean_hd95,
                'Std HD95': np.std(metric_list[:, 2]),
                'Mean ASD': mean_asd,
                'Std ASD': np.std(metric_list[:, 3]),
            }

            summary_path = os.path.join(model_dir, 'test_summary.txt')
            with open(summary_path, 'w') as f:
                for key, value in summary.items():
                    f.write(f"{key}: {value:.4f}\n")
            print(f"Saved summary to {summary_path}")

    return mean_dice, mean_jc, mean_hd95, mean_asd


if __name__ == "__main__":
    import glob
    
    args = parser.parse_args()
    
    # model_path가 제공되지 않으면 대화형 모드
    if args.model_path is None:
        print("="*80)
        print("Kvasir Test - Interactive Mode")
        print("="*80)
        
        # 사용 가능한 모델 검색
        results_base = "c:/ai-agent/repo/AD-MT/results/Kvasir"
        model_patterns = [
            os.path.join(results_base, "*_labeled/unet/*best_stu*.pth"),
            os.path.join(results_base, "*_labeled/unet/*best*.pth"),
        ]
        
        models = []
        for pattern in model_patterns:
            found = glob.glob(pattern)
            for model_path in found:
                if model_path not in models:
                    models.append(model_path)
        
        if not models:
            print("No trained models found!")
            print(f"Searched in: {results_base}")
            exit(1)
        
        print(f"\nFound {len(models)} trained models:")
        for i, model in enumerate(models, 1):
            exp_name = model.split('Kvasir\\')[-1].split('\\')[0] if 'Kvasir\\' in model else os.path.basename(os.path.dirname(os.path.dirname(model)))
            model_name = os.path.basename(model)
            print(f"  {i}. [{exp_name}] {model_name}")
        
        # 모델 선택
        while True:
            choice = input(f"\nSelect model (1-{len(models)}) or 'q' to quit: ").strip().lower()
            if choice == 'q':
                exit(0)
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    args.model_path = models[idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                print("Invalid input! Please enter a number or 'q'")
        
        # 옵션 선택
        print("\n" + "-"*80)
        print("Test Options:")
        print("-"*80)
        
        save_csv = input("Save detailed results to CSV? (y/n, default=n): ").strip().lower()
        args.save_csv = (save_csv == 'y')
        
        save_pred = input("Save prediction images? (y/n, default=n): ").strip().lower()
        args.save_predictions = (save_pred == 'y')
        
        compare = input("Compare old vs new Dice calculation? (y/n, default=n): ").strip().lower()
        args.compare_dice = (compare == 'y')
        
        print("\n" + "="*80)
        print("Starting test...")
        print("="*80)
        print(f"Model: {args.model_path}")
        print(f"Data: {args.root_path}")
        print(f"Save CSV: {args.save_csv}")
        print(f"Save Predictions: {args.save_predictions}")
        print(f"Compare Dice: {args.compare_dice}")
        print("="*80 + "\n")
    
    inference(args)