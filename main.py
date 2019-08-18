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
    model.train()
    train_loss = train(trainloader)
    print('Epoch: [{}/{}]   loss: {}'.format(epoch+1, num_epochs, train_loss))
    with torch.no_grad():
      for i in range(32):
        for j in range(32):
          out = model(fixed_input)
          print(out.shape)
          out = out.view(144, 3, 256, 32, 32).permute(0, 1, 3, 4, 2)
          softmax = F.Softmax()
          out = softmax(out)
          print(out.shape)
    torchvision.utils.save_image(out, 'content', nrow=12, padding=0)
