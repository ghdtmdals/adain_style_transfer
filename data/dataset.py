import os
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_type, transform, root = "./dataset"):
        self.data_path = os.path.join(root, data_type)
        self.transform = transform
        self.images = self.load_data()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image

    def load_data(self):
        ### 폴더 내 파일 경로를 읽어옴
        all_files = os.listdir(self.data_path)
        all_paths = []
        for path in all_files:
            all_paths.append(os.path.join(self.data_path, path))
        
        return all_paths