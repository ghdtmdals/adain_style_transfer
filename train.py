import torch
from torch import nn

from models.adain_sf import AdaINTransfer

from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import ImageDataset

from tqdm import tqdm

class Train:
    def __init__(self, style_weight, batch_size, epochs, learning_rate, n_workers = 0):
        self.style_weight = style_weight
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.batch_size = batch_size
        self.n_workers = n_workers
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

    def load_data(self, batch_size, n_workers):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        content_dataset = ImageDataset(data_type = "content", transform = transform)
        style_dataset = ImageDataset(data_type = "style", transform = transform)

        print("Total Number of Images: %d | Content Images: %d | Style Images: %d" % (len(content_dataset) + len(style_dataset), len(content_dataset), len(style_dataset)))

        if len(content_dataset) >= batch_size:
            content_dataloader = DataLoader(content_dataset, batch_size = batch_size, shuffle = True, num_workers = n_workers)
        else:
            content_dataloader = DataLoader(content_dataset, batch_size = len(content_dataset), shuffle = True, num_workers = n_workers)

        if len(style_dataset) >= batch_size:
            style_dataloader = DataLoader(style_dataset, batch_size = batch_size, num_workers = n_workers)
        else:
            style_dataloader = DataLoader(style_dataset, batch_size = len(style_dataset), num_workers = n_workers)

        return content_dataloader, style_dataloader

    def training_setup(self):
        model = AdaINTransfer(add_bn = False).to(self.device)

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(params = model.parameters(), lr = self.learning_rate)

        return model, criterion, optimizer
    
    def content_criterion(self, output, t, criterion):
        content_loss = criterion(output, t)
        return content_loss

    def style_criterion(self, multiscale_outputs, style_features, criterion):
        style_mean = torch.mean(style_features.reshape(style_features.size(0), -1), dim = 1, keepdim = True)
        style_std = torch.std(style_features.reshape(style_features.size(0), -1), dim = 1, keepdim = True)

        if style_mean.size(0) < multiscale_outputs[0].size(0):
            style_mean = style_mean.repeat(multiscale_outputs[0].size(0), style_mean.size(1))
            style_std = style_std.repeat(multiscale_outputs[0].size(0), style_std.size(1))

        multiscale_mean_losses = []
        multiscale_std_losses = []
        for output in multiscale_outputs:
            output = output.reshape((output.size(0), -1))
            mean = torch.mean(output, dim = 1, keepdim = True)
            std = torch.std(output, dim = 1, keepdim = True)
            mean_loss = criterion(mean, style_mean)
            std_loss = criterion(std, style_std)

            multiscale_mean_losses.append(mean_loss)
            multiscale_std_losses.append(std_loss)

        style_mean_loss = multiscale_mean_losses[0] + multiscale_mean_losses[1] + multiscale_mean_losses[2] + multiscale_mean_losses[3]
        style_std_loss = multiscale_std_losses[0] + multiscale_std_losses[1] + multiscale_std_losses[2] + multiscale_std_losses[3]

        return style_mean_loss + style_std_loss

    def training_loop(self):
        content_dataloader, style_dataloader = self.load_data(self.batch_size, self.n_workers)

        model, criterion, optimizer = self.training_setup()

        n_iter = 0
        running_loss = 0
        running_content_loss = 0
        running_style_loss = 0
        model.train()
        for epoch in range(self.epochs):
            for content_images in content_dataloader:
                content_images = content_images.to(self.device)

                for style_images in style_dataloader:
                    style_images = style_images.to(self.device)

                    multiscale_outputs, t, style_features = model(content_images, style_images)

                    content_loss = self.content_criterion(multiscale_outputs[-1], t, criterion)
                    style_loss = self.style_criterion(multiscale_outputs, style_features, criterion)

                    loss = content_loss + (self.style_weight * style_loss)

                    running_loss += loss.item()
                    running_content_loss += content_loss.item()
                    running_style_loss += style_loss.item()
                    n_iter += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    if n_iter % 3 == 0:
                        avg_loss = running_loss / n_iter
                        avg_content_loss = running_content_loss / n_iter
                        avg_style_loss = running_style_loss / n_iter
                        
                        content = "Epoch: %d | Iter: %d | Avg Loss: %.4f | Avg Content Loss: %.4f | Avg Style Loss: %.4f" % (epoch, n_iter, avg_loss, avg_content_loss, avg_style_loss)
                        print("\r{}".format(content), end = "")

                        running_loss = 0
                        running_content_loss = 0
                        running_style_loss = 0
                        n_iter = 0
            print("")
            torch.save({"model_state_dict": model.state_dict()},
                    "./checkpoints/last_model.pt")

if __name__ == "__main__":
    train = Train(style_weight = 0.5, batch_size = 8, epochs = 10, learning_rate = 1e-3)
    train.training_loop()