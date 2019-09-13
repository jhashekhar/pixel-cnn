import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import PixelCNN

BATCH_SIZE = 32
dataset = datasets.CIFAR10(root='cifar10', train=True,
                           transform=transforms.ToTensor(), download=True)

trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


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


softmax = nn.Softmax(dim=1)
def sample(sample, epoch):
  with torch.no_grad():
    for i in range(32):
      for j in range(32):
        out = model(sample)
        probs = softmax(out[:, :, i, j]).data
        for k in range(3):
          pixel = torch.multinomial(probs[:, k], 1).float() / 255.
          #print(pixel.shape)
          #print(sample[:, k, i, j].shape)
          #print(pixel.view(-1).shape, pixel)
          sample[:, k, i, j] = pixel.view(-1)
  torchvision.utils.save_image(sample, 'sample_{}.png'.format(epoch), nrow=8, padding=0)

num_epochs = 50
for epoch in range(num_epochs):
    train_loss = train(trainloader)
    print('[{}/{}]  loss: {}'.format(epoch+1, num_epochs, train_loss))
    sample(fixed_input, epoch)
