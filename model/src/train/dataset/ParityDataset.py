import torch.utils.data as data
from typing import Any, Callable, Optional, Tuple
import os
from torchvision.datasets.utils import check_integrity
import pickle
import numpy as np
import torch

class ParityDataset(data.Dataset):
    base_folder = "parity-4-batches-py"
    train_list = [
        ["data_batch", "fbf3f23fb44248e6247ad21d0e5c846f"],
    ]
    test_list = [
        ["test_batch", "da4ac9e3640e6b8bd8804dee74f996d5"],
    ]
    k = 4
    def __init__(
        self,
        root: str,
        train: bool = True
        ) -> None:
        super().__init__()
        
        self.root = root
        self.train = train
        
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        
        self.data: Any = []
        self.targets = []
        
        # now load the picked numpy arrays
        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f)
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])
        
        self.data = np.vstack(np.array(self.data)).reshape(-1, 512 * self.k, 8, 8)
        self.targets = np.vstack(np.array(self.targets)).reshape(-1, 10)
    
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = torch.tensor(img)
        target = torch.tensor(target)
        
        return img, target
    
    def __len__(self):
        return len(self.data)
        
    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True