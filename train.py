import torch
from torch import nn

from models.adain_sf import AdaINTransfer

from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import ImageDataset
from utils.sampler import InfiniteSamplerWrapper
from utils.utils import get_mean_std

from tqdm import tqdm

### 아래 주소의 코드를 참고하였음
### https://github.com/naoto0804/pytorch-AdaIN

class Train:
    def __init__(self, style_weight, batch_size, max_iter, learning_rate, n_workers = 0):
        self.style_weight = style_weight
        
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_data(self, batch_size, n_workers):
        transform = transforms.Compose([
            transforms.Resize((256, 256)), ### Image Size from Paper
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ### ImageNet Mean / Std
        ])

        content_dataset = ImageDataset(data_type = "train_content_images", transform = transform)
        style_dataset = ImageDataset(data_type = "train_style_images", transform = transform)

        print("Total Number of Images: %d | Content Images: %d | Style Images: %d" % (len(content_dataset) + len(style_dataset), len(content_dataset), len(style_dataset)))

        content_dataloader = DataLoader(content_dataset, batch_size = batch_size, sampler = InfiniteSamplerWrapper(content_dataset), num_workers = n_workers)
        style_dataloader = DataLoader(style_dataset, batch_size = batch_size, sampler = InfiniteSamplerWrapper(style_dataset), num_workers = n_workers)

        return content_dataloader, style_dataloader

    def training_setup(self):
        model = AdaINTransfer(add_bn = False).to(self.device)

        criterion = nn.MSELoss()

        ### Only Decoder Needs Update
        optimizer = torch.optim.Adam(params = model.decoder.parameters(), lr = self.learning_rate)

        return model, criterion, optimizer
    
    def content_criterion(self, output, target, criterion):
        content_loss = criterion(output, target)
        return content_loss
    
    def style_criterion(self, multiscale_outputs, multiscale_style, criterion):
        style_mean, style_std = get_mean_std(multiscale_outputs[0])
        target_mean, target_std = get_mean_std(multiscale_style[0])
        style_loss = criterion(style_mean, target_mean) + criterion(style_std, target_std)

        for output, target in zip(multiscale_outputs[1:], multiscale_style[1:]):
            style_mean, style_std = get_mean_std(output)
            target_mean, target_std = get_mean_std(target)
            style_loss += criterion(style_mean, target_mean) + criterion(style_std, target_std)
        
        return style_loss

    def training_loop(self):
        content_dataloader, style_dataloader = self.load_data(self.batch_size, self.n_workers)
        content_dataloader = iter(content_dataloader)
        style_dataloader = iter(style_dataloader)

        model, criterion, optimizer = self.training_setup()

        n_iter = 0
        running_loss = 0
        running_content_loss = 0
        running_style_loss = 0
        model.train()
        print("Start Training for %d Iterations" % self.max_iter)
        for i in range(self.max_iter):
            content_images = next(content_dataloader).to(self.device)
            style_images = next(style_dataloader).to(self.device)

            multiscale_outputs, t, multiscale_style = model(content_images, style_images)

            content_loss = self.content_criterion(multiscale_outputs[-1], t, criterion)
            style_loss = self.style_criterion(multiscale_outputs, multiscale_style, criterion)

            loss = content_loss + (self.style_weight * style_loss)

            running_loss += loss.item()
            running_content_loss += content_loss.item()
            running_style_loss += style_loss.item()
            n_iter += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if i % 100 == 0:
                avg_loss = running_loss / n_iter
                avg_content_loss = running_content_loss / n_iter
                avg_style_loss = running_style_loss / n_iter
                
                content = "Iter: %d | Avg Loss: %.4f | Avg Content Loss: %.4f | Avg Style Loss: %.4f" % (i, avg_loss, avg_content_loss, avg_style_loss)
                print("\r{}".format(content), end = "")

                running_loss = 0
                running_content_loss = 0
                running_style_loss = 0
                n_iter = 0

                torch.save({"model_state_dict": model.state_dict()},
                        "./checkpoints/last_model.pt")

if __name__ == "__main__":
    train = Train(style_weight = 0.7, batch_size = 8, max_iter = 20000, learning_rate = 1e-4)
    train.training_loop()