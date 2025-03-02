from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename, "rb") as img_file:
            num_magics, num_images , num_rows, num_cols = struct.unpack(">4I", img_file.read(16))
            self.images = np.frombuffer(img_file.read(num_images * num_rows * num_cols), dtype=np.uint8)
            self.images = np.array(self.images.reshape(num_images, num_rows * num_cols).astype(np.float32))
            self.images -= np.min(self.images)
            self.images /= np.max(self.images)
        with gzip.open(label_filename, "rb") as lbl_file:
            num_magics, num_labels = struct.unpack(">2I", lbl_file.read(8))
            self.labels = np.frombuffer(lbl_file.read(num_labels), dtype=np.uint8)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        images = self.apply_transforms(self.images[index].reshape(28, 28, -1))
        return images.reshape(-1, 784), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION