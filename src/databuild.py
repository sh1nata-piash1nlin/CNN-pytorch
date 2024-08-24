# -*- coding: utf-8 -*-
"""
    @author: Nguyen "sh1nata" Duc Tri <tri14102004@gmail.com>
"""
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import cv2

class AnimalDataset(Dataset):
    def __init__(self, root="", train=True, transform=None):

        if train:
            datafile_path = os.path.join(root, "animals_train")
        else:
            datafile_path = os.path.join(root, "animals_test")

        self.animalsList = []
        self.image_paths = []
        self.labels = []

        for filename in os.listdir(datafile_path):
            if os.path.isdir(os.path.join(datafile_path, filename)):
                self.animalsList.append(filename)

        for category in self.animalsList:
            category_path = os.path.join(datafile_path, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(self.animalsList.index(category))

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label


if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    dataset = AnimalDataset(root="./data/animals", train=True, transform=transform)
    image, label = dataset.__getitem__(123)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=4)
    for images, labels in dataloader:
        print(images.shape)
        print(labels)
        #break
