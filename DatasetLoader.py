import torchvision
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.transforms import v2


class CustomCIFAR10(Dataset):
    def __init__(self, root, num_samples_per_class, num_class_to_use,istrain):
        self.transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dataset = torchvision.datasets.CIFAR10(root=root, train=istrain, download=True, transform=self.transform)
        self.classes = [i for i in range(num_class_to_use)]
        self.num_samples_per_class = num_samples_per_class

        self.data, self.targets = self._filter_classes()

    def _filter_classes(self):
        datas = []
        targets = []

        pool = np.zeros(10)
        for i in self.classes:
            pool[i] = self.num_samples_per_class

        for data, target in zip(self.dataset.data, self.dataset.targets):
            if pool[target] > 0:
                data = self.transform(torch.tensor(data.transpose(2, 0, 1),dtype=torch.float) / 255)

                datas.append(data)
                targets.append(target)
                pool[target] = pool[target] - 1

            if sum(pool) == 0:
                break

        self.dataset = None
        return datas, targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


        def __init__(self, root, num_samples_per_class, num_class_to_use):
            self.transform = v2.Compose([
                v2.ToTensor(),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=self.transform)
            self.classes = [i for i in range(num_class_to_use)]
            self.num_samples_per_class = num_samples_per_class

            self.data, self.targets = self._filter_classes()

        def _filter_classes(self):
            datas = []
            targets = []

            pool = np.zeros(10)
            for i in self.classes:
                pool[i] = self.num_samples_per_class

            for data, target in zip(self.dataset.data, self.dataset.targets):
                if pool[target] > 0:
                    data = self.transform(torch.tensor(data.transpose(2, 0, 1), dtype=torch.float) / 255)

                    datas.append(data)
                    targets.append(target)
                    pool[target] = pool[target] - 1

                if sum(pool) == 0:
                    break

            self.dataset = None
            return datas, targets

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]