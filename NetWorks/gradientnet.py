import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientNet(nn.Module):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(GradientNet, self).__init__()
        kernel_x = [[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]]
        kernel_x = torch.FloatTensor(kernel_x).to(device)

        kernel_y = [[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]]
        kernel_y = torch.FloatTensor(kernel_y).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient
        
    def inference(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return grad_x, grad_y