import torchvision.transforms as transforms
from PIL import Image


class ImageTransform(object):
    def __init__(self, batchsize, s=1.0, size=224, crop_rate=0.8):
        self.s = s
        self.transform = {
            "augment0": transforms.Compose(
                [
                    # transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                    transforms.RandomResizedCrop(size=224, scale=(crop_rate, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(size),
                    transforms.ToTensor(),
                ]
            ),
            "augment1": transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
                        p=0.8,
                    ),
                    transforms.Resize(size),
                    transforms.ToTensor(),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        size, scale=(1.0, 1.0), ratio=(1.0, 1.0)
                    ),
                    transforms.Resize(size),
                    transforms.ToTensor(),
                ]
            ),
        }

    def __call__(self, image, mode):
        img_transformed = self.transform[mode](image)
        return img_transformed


class ImageDataset(object):
    def __init__(self, file_list, label_list, transform, phase):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = self.file_list[index]

        img = Image.open(image)
        if self.phase == "train":
            img0 = self.transform(img, "augment0")
            img1 = self.transform(img, "augment1")
            return img0, img1, int(self.label_list[index])
        elif self.phase == "val":
            img = self.transform(img, "val")

            return img, int(self.label_list[index])
