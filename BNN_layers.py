from torch_geometric.nn.conv import MessagePassing, SAGEConv, GraphConv
from torch_geometric.utils import add_self_loops, degree
import torch
import math
from torch.nn import Linear
import torch.nn.functional as F
import torch_sparse
from torch_scatter import scatter_add
from torch.autograd import Function
from torch.optim import Adam

# ------------- Binary_Layer ------------#
class BinLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        # size of input: [n, in_channels]
        # size of weight: [out_channels, in_channels]
        # size of bias: [out_channels]

        s = weight.size()
        weight_hat = weight.sign().relu() # NOTE: motified
        output = input.mm(weight_hat.t()) # NOTE: motified

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_bias = None

        grad_weight = grad_output.t().mm(input)

        s = weight.size()
        n = s[1]
        m = weight.norm(1, dim=1, keepdim=True).div(n).expand(s)
        # print(m.shape, m)
        m[weight.lt(-1.0)] = 0
        m[weight.gt(1.0)] = 0
        m = m.mul(grad_weight)

        m_add = weight.sign().mul(grad_weight)
        m_add = m_add.sum(dim=1, keepdim=True).expand(s)
        m_add = m_add.mul(weight.sign()).div(n)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.sign())
        if ctx.needs_input_grad[1]:
            grad_weight = m.add(m_add)
            # grad_weight[weight.lt(-1.0)] = 0
            # grad_weight[weight.gt(1.0)] = 0
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias

BinLinearFun = BinLinearFunction.apply

class BinLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (math.sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        output = BinLinearFun(input, self.weight, self.bias)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# Feat_Gated_Layer
class Feat_Gate(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Feat_Gate, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.lin = BinLinear(in_channels, out_channels, bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        x = self.lin(x)
        return x