# Henry
import os
import torch
import torchvision
from tqdm import tqdm


from config import Config


class MINSTDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = torchvision.datasets.MNIST(
            root=Config.DATA_PATH, train=split == "train", download=True
        )

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((81, 81)),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        index = 0 # Just for testing TODO: Remove this line
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label


def load_MINST_dataloader(split="train", batch_size=1, num_workers=0):
    dataset = MINSTDataset(split)
    if split == "train":
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    return dataloader


if __name__ == "__main__":
    dataset = MINSTDataset("train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    for index, label in tqdm(dataloader):
        # print(index, label)
        pass
