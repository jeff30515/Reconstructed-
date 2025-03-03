import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.down5 = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512,
                                      kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512 + 512, 512)
        self.up2 = nn.ConvTranspose2d(512, 256,
                                      kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128,
                                      kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128 + 128, 128)
        self.up4 = nn.ConvTranspose2d(128, 64,
                                      kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64 + 64, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        d5 = self.down5(p4)

        u1 = self.up1(d5)
        u1 = self.center_crop(u1, d4.size()[2:])
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = self.center_crop(u2, d3.size()[2:])
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = self.center_crop(u3, d2.size()[2:])
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = self.center_crop(u4, d1.size()[2:])
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv4(u4)

        return self.out(u4)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:diff_y + target_size[0], diff_x:diff_x + target_size[1]]

class DRIVE_Dataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.images = sorted(os.listdir(images_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name

transform = transforms.Compose([transforms.Resize((512, 512)),
                                transforms.ToTensor(),])

train_images_path = "C:\\Users\\guanju\\Desktop\\深度學習\\HW2\\archive\\DRIVE\\training\\images"
test_images_path = "C:\\Users\\guanju\\Desktop\\深度學習\\HW2\\DIRVE_TestingSet"

train_dataset = DRIVE_Dataset(train_images_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = DRIVE_Dataset(test_images_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 200

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, _ in train_loader:
        images = images.cuda()

        outputs = model(images)

        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

def compute_psnr(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

save_folder = 'reconstructed_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

psnr_scores = []

model.eval()
with torch.no_grad():
    for images, img_names in test_loader:
        images = images.cuda()

        outputs = model(images)

        psnr = compute_psnr(outputs, images)
        psnr_scores.append(psnr)

        original = images[0].cpu().numpy().transpose(1, 2, 0)
        reconstructed = outputs[0].cpu().numpy().transpose(1, 2, 0)
        original = np.clip(original, 0, 1)
        reconstructed = np.clip(reconstructed, 0, 1)

        reconstructed_img = Image.fromarray((reconstructed * 255).astype(np.uint8))
        save_path = os.path.join(save_folder, f"{img_names[0]}_reconstructed.png")
        reconstructed_img.save(save_path)

        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        # axes[0].imshow(original)
        # axes[0].set_title("Original Image")
        # axes[0].axis('off')
        # axes[1].imshow(reconstructed)
        # axes[1].set_title(f"Reconstructed Image\nPSNR: {psnr:.2f} dB")
        # axes[1].axis('off')
        # plt.show()

mean_psnr = np.mean(psnr_scores)
print(f"Mean PSNR on Test Set: {mean_psnr:.2f} dB")

# 將PSNR值存入CSV檔
data = {"Image Name": [name[0] for name in test_loader.dataset],
        "PSNR": psnr_scores}
df = pd.DataFrame(data)
df.to_csv("reconstruction_psnr.csv", index=False)
