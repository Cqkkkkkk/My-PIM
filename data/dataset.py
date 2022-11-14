import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from config import cfg


def build_loader():
    train_set, train_loader = None, None

    if cfg.datasets.train_root is not None:
        train_set = ImageDataset(istrain=True,
                                 root=cfg.datasets.train_root,
                                 data_size=cfg.datasets.data_size,
                                 return_index=True)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   num_workers=cfg.datasets.num_workers,
                                                   shuffle=True,
                                                   batch_size=cfg.datasets.batch_size)

    val_set, val_loader = None, None
    if cfg.datasets.val_root is not None:
        val_set = ImageDataset(istrain=False,
                               root=cfg.datasets.val_root,
                               data_size=cfg.datasets.data_size,
                               return_index=True)
        val_loader = torch.utils.data.DataLoader(val_set,
                                                 num_workers=1,
                                                 shuffle=True,
                                                 batch_size=cfg.datasets.batch_size)

    return train_loader, val_loader




class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )


        if istrain:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label
