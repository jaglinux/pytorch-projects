import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traceback as t

input_size = 3
batch_size = 5
output_size = 2

input_tensors = torch.rand(batch_size, input_size, requires_grad=True)
labels = torch.rand(batch_size*output_size)
print('input is ',input_tensors)
print('labels are', labels)


hidden_layer_size = 4
epoch = 50

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x= F.softmax(x, dim=1)
        return x

my_model = model()
my_optim = optim.Adam(my_model.parameters(), lr=0.01)

for i in range(epoch):
    out = my_model(input_tensors)

    num=0
    for i,j in my_model.named_parameters():
        #print(i,j)
        num+=1
    print('number of params ', num)

    print('out is ', out)
    print('out after view is', out.view(1, -1)[0])

    loss = F.mse_loss(out.view(1, -1)[0], labels)
    print('Loss is ', loss)
    my_model.zero_grad()
    loss.backward()
    my_optim.step()
