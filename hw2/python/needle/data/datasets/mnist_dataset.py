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
        super().__init__(transforms)
        with gzip.open(image_filename, 'rb') as f:
            file_content = f.read()
            # use big-endian!
            num = struct.unpack('>I', file_content[4:8])[0]
            X = np.array(struct.unpack(
                'B' * 784 * num, file_content[16:16 + 784 * num]
            ), dtype=np.float32)
            X.resize((num, 784))
        with gzip.open(label_filename, 'rb') as f:
            file_content = f.read()
            num = struct.unpack('>I', file_content[4: 8])[0]
            y = np.array([struct.unpack('B', file_content[8 + i:9 + i])[0] for i in range(num)], dtype=np.uint8)

        # THE MAX VALUE IS 255, NOT 256!
        X = X / 255.0
        self.images = X
        self.labels = y


        #raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            X, y = self.images[index], self.labels[index]
            X_in = X.reshape((28, 28, -1))
            X_out = self.apply_transforms(X_in)
            X_ret = X_out.reshape(-1, 28 * 28)
            return X_ret, y
        return self.images[index], self.labels[index]
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        raise NotImplementedError()
        ### END YOUR SOLUTION