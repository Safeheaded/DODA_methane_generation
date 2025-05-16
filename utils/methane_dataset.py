import cv2
import numpy as np
import os
import random

from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from pathlib import Path
import imageio.v3 as iio


class MethaneUnconditionaltBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,
        size=256,
        flip_p=0.5,
        ag_rate=0.8,
        crop_size=(256, 256),
    ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths],
        }

        self.size = size

        self.flip = flip_p
        self.ag_rate = ag_rate
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        self.crop_size = crop_size

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        path = Path(example["file_path_"])

        possible_labels = ["labelbinary.tif", "label_rgba.tif"]

        excluded_inputs = possible_labels
        allowed_inputs = [
            "TOA_AVIRIS_460nm.tif",
            "TOA_AVIRIS_550nm.tif",
            "TOA_AVIRIS_640nm.tif",
        ]

        inputs_paths = [
            f
            for f in path.iterdir()
            if f.is_file()
            and f.name not in excluded_inputs
            and f.name in allowed_inputs
        ]
        inputs_paths.sort()
        inputs = np.array([iio.imread(s) for s in inputs_paths], dtype=np.float32)

        image = inputs
        image = np.transpose(inputs, (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image_cp = image.copy()
        print(f"image.shape: {image.shape}")
        h, w, c = image.shape
        assert h == w, (
            "The images are not equal in length and width, please resize the images in the dataset to the same length and width first."
        )

        y_ref_start = np.random.randint(0, h - self.crop_size[0] + 1)
        x_ref_start = np.random.randint(0, w - self.crop_size[1] + 1)
        reference = image[
            y_ref_start : y_ref_start + self.crop_size[0],
            x_ref_start : x_ref_start + self.crop_size[1],
        ]
        reference = self.image_processor(images=reference)["pixel_values"][0]

        if random.random() < self.ag_rate:
            # 创建正方形的四个顶点坐标和中心坐标
            square_size = h
            center = (h // 2, w // 2)
            square_points = np.array(
                [
                    (0, 0),
                    (0, square_size),
                    (square_size, 0),
                    (square_size, square_size),
                ],
                dtype=np.float32,
            )

            # 随机生成旋转角度
            angle = random.uniform(-90, 90)

            # 创建顶点旋转矩阵并应用旋转
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_points = cv2.transform(
                square_points.reshape(1, -1, 2), rotation_matrix
            )

            y_min = int(np.min(rotated_points[0, :, 1]))
            y_max = int(np.max(rotated_points[0, :, 1]))
            max_scale = square_size / (y_max - y_min)

            # 将缩放因子设置为（0.5，max_scale)
            scale_factor = random.uniform(0.6, max_scale)

            # 创建缩放矩阵
            scaling_matrix = np.array(
                [
                    [scale_factor, 0, (1 - scale_factor) * center[0]],
                    [0, scale_factor, (1 - scale_factor) * center[1]],
                ],
                dtype=np.float32,
            )

            # 应用缩放
            scaled_points = cv2.transform(rotated_points, scaling_matrix)

            # 将坐标四舍五入为整数
            scaled_points = np.int32(scaled_points)

            transformed_x_min = int(np.min(scaled_points[:, :, 0]))
            transformed_y_min = int(np.min(scaled_points[:, :, 1]))

            # 随机生成平移量，并平移
            shift_x = random.randint(-transformed_x_min, transformed_x_min)
            shift_y = random.randint(-transformed_y_min, transformed_y_min)

            scaled_points[:, :, 0] += shift_x
            scaled_points[:, :, 1] += shift_y

            back_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

            # 执行旋转
            rotated_image = cv2.warpAffine(image, back_rotation_matrix, (h, w))

            scaled_points = cv2.transform(
                scaled_points.reshape(1, -1, 2), back_rotation_matrix
            )
            scaled_points = np.int32(scaled_points)

            x_max = np.max(scaled_points[:, :, 0])
            x_min = np.min(scaled_points[:, :, 0])
            y_max = np.max(scaled_points[:, :, 1])
            y_min = np.min(scaled_points[:, :, 1])

            image = rotated_image[y_min:y_max, x_min:x_max]

        try:
            resized_channels = [
                cv2.resize(
                    image[:, :, i],
                    (self.size, self.size),
                    interpolation=cv2.INTER_LINEAR,
                )
                for i in range(image.shape[2])
            ]
            image = np.stack(resized_channels, axis=-1)
        except:
            resized_channels = [
                cv2.resize(
                    image_cp[:, :, i],
                    (self.size, self.size),
                    interpolation=cv2.INTER_LINEAR,
                )
                for i in range(image_cp.shape[2])
            ]
            image = np.stack(resized_channels, axis=-1)
            print(example["file_path_"])
            print(x_max, x_min, y_max, y_min)

        if random.random() < self.flip:
            image = cv2.flip(image, 0)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["reference"] = reference
        print(f"final_image.shape: {example['image'].shape}")
        return example


class methaneUnconditionalTrain(MethaneUnconditionaltBase):
    def __init__(
        self,
        txt_file="datasets/methane/train_ldm.txt",
        data_root="datasets/methane/data",
        **kwargs,
    ):
        super().__init__(txt_file, data_root, **kwargs)


class methaneUnconditionalValidation(MethaneUnconditionaltBase):
    def __init__(
        self,
        txt_file="datasets/methane/val_ldm.txt",
        data_root="datasets/methane/data",
        flip_p=0.0,
        ag_rate=0.0,
        **kwargs,
    ):
        super().__init__(txt_file, data_root, flip_p=flip_p, ag_rate=ag_rate, **kwargs)


class MethaneConditionaltBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,
        size=256,
        flip_p=0.5,
        ag_rate=0.8,
        shuffle_channel=True,
        crop_size=(256, 256),
    ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "rt") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.source_folder_path = os.path.join(self.data_root, "source")
        self.target_folder_path = os.path.join(self.data_root, "target")
        # source == mask label
        self.labels = {
            "source_file_path_": [
                os.path.join(self.source_folder_path, l) for l in self.image_paths
            ],
            "target_file_path_": [
                os.path.join(self.target_folder_path, l) for l in self.image_paths
            ],
        }

        self.size = size
        self.flip = flip_p
        self.ag_rate = ag_rate
        self.shuffle_c = shuffle_channel
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        self.crop_size = crop_size

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # source == mask label
        example = dict((k, self.labels[k][idx]) for k in self.labels)

        target_path = Path(example["target_file_path_"])
        source_path = example["source_file_path_"]

        possible_labels = ["labelbinary.tif", "label_rgba.tif"]

        excluded_inputs = possible_labels
        allowed_inputs = [
            "TOA_AVIRIS_460nm.tif",
            "TOA_AVIRIS_550nm.tif",
            "TOA_AVIRIS_640nm.tif",
        ]

        inputs_paths = [
            f
            for f in target_path.iterdir()
            if f.is_file()
            and f.name not in excluded_inputs
            and f.name in allowed_inputs
        ]
        inputs_paths.sort()
        target = np.array([iio.imread(s) for s in inputs_paths], dtype=np.float32)

        source = np.array(iio.imread(source_path))

        if source.ndim != 2:
            raise ValueError("Obraz powinien być w skali szarości (2D)")

        # Dodaj dwie warstwy wypełnione zerami
        h, w = source.shape
        source = np.stack([source, np.zeros((h, w), dtype=source.dtype), np.zeros((h, w), dtype=source.dtype)], axis=-1)

        if self.shuffle_c:
            # 生成一个随机排列的索引
            random_indices = np.random.permutation(3)
            # 使用随机索引重新排列第三维
            source = source[:, :, random_indices]

        source_cp = source.copy()
        target_cp = target.copy()

        h, w, c = target.shape
        assert h == w, (
            "The images are not equal in length and width, please resize the images in the dataset to the same length and width first."
        )

        y_ref_start = np.random.randint(0, h - self.crop_size[0] + 1)
        x_ref_start = np.random.randint(0, w - self.crop_size[1] + 1)
        reference = target[
            y_ref_start : y_ref_start + self.crop_size[0],
            x_ref_start : x_ref_start + self.crop_size[1],
        ]
        reference = self.image_processor(images=reference)["pixel_values"][0]

        if random.random() < self.ag_rate:
            # 创建正方形的四个顶点坐标和中心坐标
            square_size = h
            center = (h // 2, h // 2)
            square_points = np.array(
                [
                    (0, 0),
                    (0, square_size),
                    (square_size, 0),
                    (square_size, square_size),
                ],
                dtype=np.float32,
            )

            # 随机生成旋转角度
            angle = random.uniform(-90, 90)

            # 创建顶点旋转矩阵并应用旋转
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_points = cv2.transform(
                square_points.reshape(1, -1, 2), rotation_matrix
            )

            y_min = int(np.min(rotated_points[0, :, 1]))
            y_max = int(np.max(rotated_points[0, :, 1]))
            max_scale = square_size / (y_max - y_min)

            # 将缩放因子设置为（0.5，max_scale)
            scale_factor = random.uniform(0.6, max_scale)

            # 创建缩放矩阵
            scaling_matrix = np.array(
                [
                    [scale_factor, 0, (1 - scale_factor) * center[0]],
                    [0, scale_factor, (1 - scale_factor) * center[1]],
                ],
                dtype=np.float32,
            )

            # 应用缩放
            scaled_points = cv2.transform(rotated_points, scaling_matrix)

            # 将坐标四舍五入为整数
            scaled_points = np.int32(scaled_points)

            transformed_x_min = int(np.min(scaled_points[:, :, 0]))
            transformed_y_min = int(np.min(scaled_points[:, :, 1]))

            # 随机生成平移量，并平移
            shift_x = random.randint(-transformed_x_min, transformed_x_min)
            shift_y = random.randint(-transformed_y_min, transformed_y_min)

            scaled_points[:, :, 0] += shift_x
            scaled_points[:, :, 1] += shift_y

            back_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

            # 执行旋转
            rotated_source = cv2.warpAffine(source, back_rotation_matrix, (h, w))
            rotated_target = cv2.warpAffine(target, back_rotation_matrix, (h, w))

            scaled_points = cv2.transform(
                scaled_points.reshape(1, -1, 2), back_rotation_matrix
            )
            scaled_points = np.int32(scaled_points)

            x_max = np.max(scaled_points[:, :, 0])
            x_min = np.min(scaled_points[:, :, 0])
            y_max = np.max(scaled_points[:, :, 1])
            y_min = np.min(scaled_points[:, :, 1])

            source = rotated_source[y_min:y_max, x_min:x_max]
            target = rotated_target[y_min:y_max, x_min:x_max]

        try:
            source = cv2.resize(source, (self.size, self.size))
            target = cv2.resize(target, (self.size, self.size))
        except:
            source = cv2.resize(source_cp, (self.size, self.size))
            target = cv2.resize(target_cp, (self.size, self.size))
            print("file_path_")
            print(x_max, x_min, y_max, y_min)

        if random.random() < self.flip:
            source = cv2.flip(source, 0)
            target = cv2.flip(target, 0)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, hint=source, reference=reference)


class methaneConditionalTrain(MethaneConditionaltBase):
    def __init__(
        self,
        txt_file="datasets/wheat/train_cldm.txt",
        data_root="datasets/wheat",
        **kwargs,
    ):
        super().__init__(txt_file=txt_file, data_root=data_root, **kwargs)


class methaneConditionalValidation(MethaneConditionaltBase):
    def __init__(
        self,
        txt_file="datasets/wheat/val_cldm.txt",
        data_root="datasets/wheat",
        **kwargs,
    ):
        super().__init__(txt_file=txt_file, data_root=data_root, **kwargs)
