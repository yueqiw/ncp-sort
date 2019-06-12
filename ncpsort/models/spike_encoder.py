
import torch
import torch.nn as nn
import torch.nn.functional as F

class NCP_SpikeEncoder():
    def __init__(self, model):
        model.eval()
        self.model = model
        self.h_dim = model.params['h_dim']
        self.device = model.params['device']

    def encode(self, data):
        self.batch_size = data.shape[0]
        self.N = data.shape[1]

        data = data.to(self.device)    
        data = data.view([self.batch_size*self.N, data.shape[2], data.shape[3]])
        
        hs = self.model.h(data).view([self.batch_size, self.N, self.h_dim])   
        return hs 


class BasicBlock(nn.Module):
    """
    based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
    """

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet1DEncoder(nn.Module):
    def __init__(self, block, num_blocks, out_size=256, input_dim=7, planes=32):
        super(ResNet1DEncoder, self).__init__()
        self.in_planes = planes

        self.conv1 = nn.Conv1d(input_dim, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.layer1 = self._make_layer(block, planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, planes*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(planes*8, out_size)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    spike_encoder = ResNet1DEncoder(block=BasicBlock, 
                            num_blocks=[1,1,1,1],
                            out_size=256, 
                            input_dim=7, 
                            planes=32)
    n_channels = 7
    n_timesteps = 32
    input_data = torch.rand(100, n_channels, n_timesteps)
    # input shape: [batch_size * N, n_channels=7, n_timesteps=32]
    output = spike_encoder(input_data)
    print("input_data", input_data.shape)
    print("output", output.shape)
    


