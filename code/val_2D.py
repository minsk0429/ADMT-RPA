import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            dice = metric.binary.dc(pred, gt)
            hd95 = metric.binary.hd95(pred, gt)
            return dice, hd95
        except RuntimeError:
            # HD95 computation failed, return dice with 0 for hd95
            dice = metric.binary.dc(pred, gt)
            return dice, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    # For 2D single image input
    if len(image.shape) == 4 and image.shape[0] == 1:  # Single RGB image: (1, C, H, W)
        image = image.squeeze(0).cpu().detach().numpy()  # (C, H, W)
        label = label.squeeze(0).cpu().detach().numpy()  # (H, W)
        
        c, x, y = image.shape
        # Resize image
        image_resized = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(image_resized).unsqueeze(0).float().cuda()  # (1, C, H, W)
        
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        
        metric_list = []
        for i in range(1, classes):
            metric_list.append(calculate_metric_percase(
                prediction == i, label == i))
        return metric_list
    
    # Original code for batch processing
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        if len(image.shape) == 4:  # RGB: (batch, channel, H, W)
            slice = image[ind, :, :, :]  # (C, H, W)
            c, x, y = slice.shape
            slice = zoom(slice, (1, patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(0).float().cuda()  # (1, C, H, W)
        else:  # Grayscale: (batch, H, W)
            slice = image[ind, :, :]  # (H, W)
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  # (1, 1, H, W)
        
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
