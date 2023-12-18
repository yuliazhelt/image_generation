from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(self, dataset_path = '.', image_size=64, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        self.mean = mean
        self.std = std
        print(dataset_path)
        self.dataset = ImageFolder(
            dataset_path,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ]
            )
        )

        self.inverse_transform = transforms.Compose(
            [ 
                transforms.Normalize(mean=[ 0., 0., 0. ], std=list(1 / np.array(self.std))),
                transforms.Normalize(mean=list(-1 * np.array(self.mean)), std=[ 1., 1., 1. ]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ind):
        return self.dataset[ind]
    
    def denorm(self, img_tensors):
        return self.inverse_transform(img_tensors)
    
    def save_images(self, images, save_dir, save_name, nrow=8):
        save_image(self.denorm(images), os.path.join(save_dir, save_name), nrow=nrow)
        return os.path.join(save_dir, save_name)


class GeneratedImagesDataset(Dataset):
    def __init__(self, generated_images):
        """
        Args:
            generated_images (list or array): List or array of generated images.
        """
        self.generated_images = generated_images

    def __len__(self):
        return len(self.generated_images)

    def __getitem__(self, ind):
        image = self.generated_images[ind]
        return image, 0