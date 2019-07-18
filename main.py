import torch
import torch.nn as nn
from torchvision import transforms, datasets
from model import PixelCNN

trainset = datasets.MNIST('content',train=True, download=True,
                          transform=tansforms.ToTensor())

trainloader = torch.utils.data.DataLoader(datasets, batch_size=64, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
model = PixelCNN().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate=lr, betas=(0.5, 0.999))

def train(trainloader):
    model.train()
    for idx, data in enumerate(trainloader):
        img, _ = data
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    return loss

num_epochs = 5
for epoch in range(num_epochs):
    train_loss = train()
    print('Epoch: [{}/{}]   loss: {}'.format(epoch+1, num_epochs, train_loss))
