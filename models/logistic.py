import torch
import torch.optim as optim
import torch.nn as nn
from functools import reduce

class Logistic(nn.Module):
    def __init__(self, volume, num_classes):
        super(Logistic, self).__init__()
        self.fc = nn.Linear(reduce(lambda a, b: a * b, volume),
                            num_classes)

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    batch_size, n_channels, height, width, depth = 5, 4, 160, 160, 144
    num_classes = 2
    inputs = torch.randn(batch_size, n_channels, height, width, depth)
    model = Logistic(volume=tuple(inputs.size())[1:],
                    num_classes=num_classes)
    inputs = inputs.view(batch_size, -1)
    outputs = model(inputs)
    assert outputs.size() == torch.Size([batch_size, num_classes])
    print('pass the dimension check')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.randint(low=0, high=2, size=(batch_size,))
    
    for iter in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        print('iter{}, loss{}'.format(iter, loss))
        optimizer.step()