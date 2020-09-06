import torchvision.transforms as transforms
from PIL import Image


class ImageTransform(object):
    def __init__(self, batchsize, s=1.0, size=224):
        self.s = s
        self.transform = {
            "augment0": transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=200, scale=(0.8, 1.0)),
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
            "test": transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                ]
            ),
        }

    def __call__(self, image, mode):
        img_transformed = self.transform[mode](image)
        return img_transformed


class ImageDataset(object):
    def __init__(self, file_list, label_list, transform):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image = self.file_list[index]
        img = Image.open(image)
        img0 = self.transform(img, "augment0")
        img1 = self.transform(img, "augment1")

        return img0, img1, int(self.label_list[index])
