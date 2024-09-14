import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image


def load_data(image_folder_path):

    train_dir = image_folder_path + "/train"
    valid_dir = image_folder_path + "/valid"

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    valid_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64)

    print("Successfully Loaded Data")
    print("-" * 75)

    return train_dataloader, valid_dataloader, train_dataset


def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """

    with Image.open(image_path) as img:
        # Resize with aspect ratio
        init_width, init_height = img.size
        if init_width > init_height:
            size = (256, int(256 * init_height / init_width))
        else:
            size = (int(256 * init_width / init_height), 256)

        img.thumbnail(size)

        # Crop
        width, height = img.size
        new_width, new_height = (224, 224)

        left = (width - new_width) / 2
        upper = (height - new_height) / 2
        right = (width + new_width) / 2
        lower = (height + new_height) / 2

        croped_img = img.crop((left, upper, right, lower))
        # Normalize
        np_image = np.array(croped_img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        norm_np_image = (np_image - mean) / std
        norm_np_image = norm_np_image.transpose((2, 0, 1))
        return torch.from_numpy(norm_np_image)
