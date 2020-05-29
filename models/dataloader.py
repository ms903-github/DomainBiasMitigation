import torch
import pickle
import numpy as np
from PIL import Image
import utils

class CifarDataset(torch.utils.data.Dataset):
    """Cifar dataloader, output image and target"""
    
    def __init__(self, image_path, target_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.targets = pickle.load(f)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)
    
class CifarDatasetWithWeight(torch.utils.data.Dataset):
    """Cifar dataloader, output image, target and the weight for this sample"""
    
    def __init__(self, image_path, target_path, weight_list, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.targets = pickle.load(f)
        self.transform = transform
        self.weight_list = weight_list

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        weight = self.weight_list[index]
        return img, target, weight

    def __len__(self):
        return len(self.targets)
    
class CifarDatasetWithDomain(torch.utils.data.Dataset):
    """Cifar dataloader, output image, class target and domain for this sample"""
    
    def __init__(self, image_path, class_label_path, domain_label_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(class_label_path, 'rb') as f:
            self.class_label = pickle.load(f)
        with open(domain_label_path, 'rb') as f:
            self.domain_label = pickle.load(f)    
        self.transform = transform

    def __getitem__(self, index):
        img, class_label, domain_label = \
            self.images[index], self.class_label[index], self.domain_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, class_label, domain_label

    def __len__(self):
        return len(self.class_label)

class ImdbDataset(torch.utils.data.Dataset):
    """Imdb dataloader, output image and target"""
    
    def __init__(self, data_path, transform=None):
        with open(data_path, "rb") as f:
            lines = f.readlines()
            pathlist, labellist = [], []
            for line in lines:
                path, label, gen = line.split()
                pathlist.append(path)
                labellist.append(label)
        self.pathlist = pathlist
        self.labellist = labellist
        self.transform = transform

    def __getitem__(self, index):
        path = self.pathlist[index]
        img = Image.open(path).convert("RGB")
        label = int(self.labellist[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labellist)
    
class ImdbDatasetWithDomain(torch.utils.data.Dataset):
    """Imdb dataloader, output image, class target and domain for this sample"""
    
    def __init__(self, image_path, transform=None):
        with open(data_path, "rb") as f:
            lines = f.readlines()
            pathlist, labellist, genlist = [], [], []
            for line in lines:
                path, label, gen = line.split()
                pathlist.append(path)
                labellist.append(label)
                genlist.append(gen)
        self.pathlist = pathlist
        self.labellist = labellist
        self.genlist = genlist
        self.transform = transform

    def __getitem__(self, index):
        ipath = self.pathlist[index]
        img = Image.open(path).convert("RGB")
        label = int(self.labellist[index])
        gen = int(self.genlist[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, label, gen

    def __len__(self):
        return len(self.class_label)

class ImdbDatasetFor16cls(torch.utils.data.Dataset):
    """Imdb dataloader, output image and target"""
    
    def __init__(self, data_path, transform=None):
        with open(data_path, "rb") as f:
            lines = f.readlines()
            pathlist, labellist = [], []
            for line in lines:
                path, label, gen = line.split()
                pathlist.append(path)
                if int(gen) == 1:
                    labellist.append(int(label))
                elif int(gen) == 0:
                    labellist.append(int(label) + 7)
        self.pathlist = pathlist
        self.labellist = labellist
        self.transform = transform

    def __getitem__(self, index):
        path = self.pathlist[index]
        img = Image.open(path).convert("RGB")
        label = self.labellist[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labellist)

class CelebaDataset(torch.utils.data.Dataset):
    """Celeba dataloader, output image and target"""
    
    def __init__(self, key_list, image_feature, target_dict, transform=None):
        self.key_list = key_list
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.transform = transform

    def __getitem__(self, index):
        key = self.key_list[index]
        img, target = Image.fromarray(self.image_feature[key][()]), self.target_dict[key]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(target)

    def __len__(self):
        return len(self.key_list)

class CelebaDatasetWithWeight(torch.utils.data.Dataset):
    """Celeba dataloader, output image, target and weight for this sample"""
    
    def __init__(self, key_list, image_feature, target_dict, transform=None):
        self.key_list = key_list
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.transform = transform
        target = np.array([target_dict[key] for key in key_list])
        self.per_class_weight = utils.compute_class_weight(target)

    def __getitem__(self, index):
        key = self.key_list[index]
        img, target = Image.fromarray(self.image_feature[key][()]), self.target_dict[key]
        weight = [class_weight[index] for class_weight in self.per_class_weight]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(target), torch.FloatTensor(weight)

    def __len__(self):
        return len(self.key_list)




