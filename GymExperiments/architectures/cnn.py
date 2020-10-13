import torch
import torch.nn as nn


class SimpleCNNEncoder(nn.Module):
    def __init__(self, out_dim, image_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=3, stride=2)
        self.linear = nn.Linear(in_features=200, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class SimpleCNNDecoder(nn.Module):
    def __init__(self, representation_size, image_channels=3):
        super().__init__()
        
        self.fcT1 = torch.nn.Linear(representation_size, 64)

        self.convT1 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1)
        self.convT2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.convT3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.convT4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.convT5 = nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.fcT1(x)
        x = self.relu(x)

        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = self.convT1(x)
        x = self.relu(x)

        x = self.convT2(x)
        x = self.relu(x)

        x = self.convT3(x)
        x = self.relu(x)

        x = self.convT4(x)
        x = self.relu(x)

        x = self.convT5(x)
        x = self.sigmoid(x)

        return x



if __name__ == '__main__':
    encoder = SimpleCNNEncoder(8, 3)
    decoder = SimpleCNNDecoder(8, 3)

    input_tensor = torch.empty(size=(13, 3, 96, 96), requires_grad=True)

    print(input_tensor.size())
    rep_tensor = encoder(input_tensor)
    print(rep_tensor.size())
    out_tensor = decoder(rep_tensor)
    print(out_tensor.size())