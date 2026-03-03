import torchvision.transforms as T
# Torch 기반 ColorJitter만 적용하는 증강 (PIL 변환 없이 값 손실 최소화)
class TorchColorJitterAugment(object):
    def __init__(self, output_size, debug_save_dir=None):
        self.output_size = output_size
        # 파라미터 강화: 0.3, 0.3, 0.3, 0.1 (시각적으로 명확한 변화)
        self.jitter = T.ColorJitter(0.3, 0.3, 0.3, 0.1)
        self.debug_save_dir = debug_save_dir

    def __call__(self, sample):
        import torch
        import os
        image = sample["image"]
        # (C, H, W) 텐서로 가정
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        image = image.float()
        # 값 범위/타입 체크 (전)
        print(f"[CJITTER] before: dtype={image.dtype}, min={image.min().item():.4f}, max={image.max().item():.4f}, nan={torch.isnan(image).any().item()}, inf={torch.isinf(image).any().item()}")
        # 디버깅용 이미지 저장 (선택)
        if self.debug_save_dir is not None:
            os.makedirs(self.debug_save_dir, exist_ok=True)
            img_np = image.detach().cpu().numpy()
            if img_np.shape[0] == 3:
                from matplotlib import pyplot as plt
                plt.imsave(os.path.join(self.debug_save_dir, "cjitter_input.png"), img_np.transpose(1,2,0))
        # torchvision ColorJitter expects (C, H, W) float in [0,1]
        image = torch.clamp(image, 0, 1)
        image = self.jitter(image)
        # 값 범위/타입 체크 (후)
        print(f"[CJITTER] after: dtype={image.dtype}, min={image.min().item():.4f}, max={image.max().item():.4f}, nan={torch.isnan(image).any().item()}, inf={torch.isinf(image).any().item()}")
        if self.debug_save_dir is not None:
            img_np = image.detach().cpu().numpy()
            if img_np.shape[0] == 3:
                from matplotlib import pyplot as plt
                plt.imsave(os.path.join(self.debug_save_dir, "cjitter_output.png"), img_np.transpose(1,2,0))
        return {"image": image, "label": sample["label"]}
import os
import random
import h5py
import itertools
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                          1. Samplers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class TwoStreamBatchSampler(Sampler): #
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        2. Generators
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # Handle both RGB and grayscale images
        if len(image.shape) == 2:  # grayscale
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:  # RGB
            x, y, c = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        
        x, y = label.shape
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample


class WeakStrongAugment(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if hasattr(sample["image"], "clone"):
            image_org = sample["image"].clone()
        else:
            image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # 디버깅: image shape/dtype 출력
        print(f"[WeakStrongAugment] image shape: {getattr(image, 'shape', None)}, dtype: {getattr(image, 'dtype', None)}")

        # geometry
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        

        # --- 패치: 이미 (256,256)로 resize된 경우 resize 생략 ---
        def to_hw_c(img):
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    # (C, H, W) → (H, W, C)
                    return img.permute(1, 2, 0).cpu().numpy()
                elif img.dim() == 2:
                    return img.cpu().numpy()
                else:
                    raise ValueError(f"Unsupported tensor shape for resize: {img.shape}")
            elif isinstance(img, np.ndarray):
                if img.ndim == 3:
                    axes = list(img.shape)
                    if 3 in axes:
                        c_idx = axes.index(3)
                        if c_idx == 0:
                            return np.transpose(img, (1, 2, 0))
                        elif c_idx == 1:
                            return np.transpose(img, (0, 2, 1))
                        elif c_idx == 2:
                            return img
                        else:
                            raise ValueError(f"Unexpected 3D shape for image: {img.shape}")
                    else:
                        raise ValueError(f"No channel dim==3 in ndarray shape for resize: {img.shape}")
                elif img.ndim == 2:
                    return img
                else:
                    raise ValueError(f"Unsupported ndarray shape for resize: {img.shape}")
            else:
                raise ValueError(f"Unsupported image type for resize: {type(img)}")

        def to_hw(img):
            if isinstance(img, torch.Tensor):
                if img.dim() == 2:
                    return img.cpu().numpy()
                elif img.dim() == 3 and img.shape[0] == 1:
                    return img.squeeze(0).cpu().numpy()
                else:
                    raise ValueError(f"Unsupported label tensor shape for resize: {img.shape}")
            return img

        # (C,H,W) or (H,W,C) or (B,C,H,W)
        # image_org, image가 이미 (256,256)로 맞춰져 있으면 resize 생략
        skip_resize = False
        if isinstance(image, torch.Tensor) and image.dim() == 3 and image.shape[1:3] == tuple(self.output_size):
            skip_resize = True
        elif isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0:2] == tuple(self.output_size):
            skip_resize = True

        if skip_resize:
            # 이미 (256,256)이면 바로 컬러지터만 적용
            image_org_np = to_hw_c(image_org)
            image_np = to_hw_c(image)
            label_np = to_hw(label)
        else:
            image_org_np = self.resize(to_hw_c(image_org))
            image_np = self.resize(to_hw_c(image))
            label_np = self.resize(to_hw(label))
        print(f"[WeakStrongAugment][after resize] image shape: {getattr(image_np, 'shape', None)}, dtype: {getattr(image_np, 'dtype', None)}")

        # strong augmentation is color jitter
        image_strong = func_strong_augs(image_np, p_color=0.8, p_blur=0.2)

        # fix dimensions - handle both RGB (H,W,C) and grayscale (H,W)
        if len(image_org_np.shape) == 3:  # RGB image
            image_org = torch.from_numpy(image_org_np.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image = torch.from_numpy(image_np.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:  # grayscale image
            image_org = torch.from_numpy(image_org_np.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image = torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
        label = torch.from_numpy(label_np.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        # torch.Tensor 방어
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 2:  # grayscale image (H, W)
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(image.shape) == 3:  # RGB image (H, W, C)
            if image.shape[2] not in [1, 3]:
                raise ValueError(f"resize: 3rd dim must be 1 or 3, got {image.shape}")
            x, y, c = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    

class WeakOnlyAugment(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if hasattr(sample["image"], "clone"):
            image_org = sample["image"].clone()
        else:
            image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # geometry - weak augmentation only (기존 WeakStrongAugment에서 사용하던 것과 동일)
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # resize
        image_org = self.resize(image_org)
        image = self.resize(image)
        label = self.resize(label)
        
        # strong augmentation 제거 - weak augmentation만 사용 (resize된 image를 그대로 사용)
        image_strong = image.copy()  # weak augmentation과 동일하게 설정

        # fix dimensions - handle both RGB (H,W,C) and grayscale (H,W)
        if len(image_org.shape) == 3:  # RGB image
            image_org = torch.from_numpy(image_org.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image_strong = torch.from_numpy(image_strong.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:  # grayscale image
            image_org = torch.from_numpy(image_org.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image_strong = torch.from_numpy(image_strong.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,  # weak와 동일
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        if len(image.shape) == 2:  # grayscale image (H, W)
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(image.shape) == 3:  # RGB image (H, W, C)
            x, y, c = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")


class WeakStrongAugmentMore(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if hasattr(sample["image"], "clone"):
            image_org = sample["image"].clone()
        else:
            image_org = sample["image"].copy()
        image, label = sample["image"], sample["label"]
        
        # geometry
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        # elif random.random() > 0.5:
        #     image, label = random_rotate(image, label)
        
        # resize
        image_org = self.resize(image_org)
        image = self.resize(image)
        label = self.resize(label)
        
        # strong augmentation is color jitter
        image_strong = func_strong_augs(image, p_color=0.5, p_blur=0.2)
        image_strong_more = func_strong_augs(image, p_color=1.0, p_blur=0.2)

        # fix dimensions - handle both RGB (H,W,C) and grayscale (H,W)
        if len(image_org.shape) == 3:  # RGB image
            image_org = torch.from_numpy(image_org.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        else:  # grayscale image
            image_org = torch.from_numpy(image_org.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (H,W) -> (1,H,W)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image_org,
            "image_weak": image,
            "image_strong": image_strong,
            "image_strong_more": image_strong_more,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        if len(image.shape) == 2:  # grayscale image (H, W)
            x, y = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(image.shape) == 3:  # RGB image (H, W, C)
            x, y, c = image.shape
            return zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                         3. Augmentations
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image, p=1.0):
    # if not torch.is_tensor(image):
    #     np_to_tensor = transforms.ToTensor()
    #     image = np_to_tensor(image)
    # s is the strength of color distortion.
    # s = 1.0
    # jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    jitter = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
    if np.random.random() < p:
        image = jitter(image)
    return image


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def func_strong_augs(image, p_color=0.8, p_blur=0.5):
    # shape 보정: torch.Tensor면 (C, H, W) → (H, W, C)로 변환
    if isinstance(image, torch.Tensor):
        # torch.Tensor: (C, H, W)만 허용
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            img_np = image.cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # (C, H, W) → (H, W, C)
        elif image.dim() == 2:
            img_np = image.cpu().numpy()
        else:
            print("[func_strong_augs] Debug info (torch.Tensor):")
            print(f"  shape: {image.shape}")
            print(f"  dtype: {image.dtype}")
            raise ValueError(f"Unsupported torch.Tensor shape for PIL conversion: {image.shape}")
    else:
        # numpy array: (H, W, C) 또는 (H, W)만 허용
        img_np = image
        if img_np.ndim == 3:
            if img_np.shape[2] not in [1, 3]:
                print("[func_strong_augs] Debug info (np.ndarray):")
                print(f"  shape: {img_np.shape}")
                print(f"  dtype: {img_np.dtype}")
                print(f"  min: {img_np.min()}, max: {img_np.max()}")
                raise ValueError(f"Unsupported np.ndarray shape for PIL conversion: {img_np.shape}")
        elif img_np.ndim == 2:
            pass
        else:
            print("[func_strong_augs] Debug info (np.ndarray):")
            print(f"  shape: {img_np.shape}")
            print(f"  dtype: {img_np.dtype}")
            raise ValueError(f"Unsupported np.ndarray ndim for PIL conversion: {img_np.shape}")

    # 최종 shape에 따라 PIL 변환
    if img_np.ndim == 3:
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        img = color_jitter(img, p_color)
        img = blur(img, p_blur)
        # (H,W,C) -> (C,H,W)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    elif img_np.ndim == 2:
        img = Image.fromarray((img_np * 255).astype(np.uint8))
        img = color_jitter(img, p_color)
        img = blur(img, p_blur)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
    else:
        raise ValueError(f"Final image shape not supported for PIL conversion: {img_np.shape}")

    return img


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                        Kvasir Dataset
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
class KvasirDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.list", "r", encoding="utf-8") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/test.list", "r", encoding="utf-8") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            with open(self._base_dir + "/test.list", "r", encoding="utf-8") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        
        if self.split == "train":
            image_path = self._base_dir + "/train/images/{}.png".format(case)
            mask_path = self._base_dir + "/train/masks/{}.png".format(case) 
        elif self.split == "val":
            image_path = self._base_dir + "/test/images/{}.png".format(case)
            mask_path = self._base_dir + "/test/masks/{}.png".format(case)
        else:  # test
            image_path = self._base_dir + "/test/images/{}.png".format(case)
            mask_path = self._base_dir + "/test/masks/{}.png".format(case)
        
        # Load image and mask using PIL
        image = Image.open(image_path).convert('RGB')  # Keep RGB for color information
        mask = Image.open(mask_path).convert('L')  # grayscale
        
        # Convert to numpy arrays
        image = np.array(image).astype(np.float32) / 255.0  # normalize to [0,1], shape: (H, W, 3)
        mask = np.array(mask).astype(np.float32) / 255.0    # normalize to [0,1], shape: (H, W)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0.5).astype(np.uint8)
        
        sample = {"image": image, "label": mask}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        else:
            # For validation/test, convert to tensor
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            mask = torch.from_numpy(mask.astype(np.uint8))
            sample = {"image": image, "label": mask}
        sample["idx"] = idx
        return sample
