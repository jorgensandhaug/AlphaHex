import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = self.relu(x)
        return x

class ConvBlockWithoutRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlockWithoutRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
        
    

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = ConvBlockWithoutRelu(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = F.relu(x)
        return x



# This is the final best perfomring CNN architecture used in the OHT. It is largelyl based on https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0
class HexNet(nn.Module):
    def __init__(self, size, in_channels, num_residual_blocks, num_filters, policy_output_dim, kernel_size=3, stride=1, padding=1, device="cpu"):
        super(HexNet, self).__init__()
        self.size = size
        self.conv = ConvBlock(in_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding)
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_filters, kernel_size, stride, padding) for _ in range(num_residual_blocks)])

        self.policy_head = nn.Sequential(
            ConvBlock(num_filters, 2, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(2 * size**2, policy_output_dim),
        )
        
        self.value_head = nn.Sequential(
            ConvBlock(num_filters, 1, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(size**2, 1),
            nn.Tanh()
        )

        # save all the other parameters
        self.in_channels = in_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_filters = num_filters
        self.policy_output_dim = policy_output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        self.to(device)

    def forward(self, x):
        x = self.conv(x)
        for i in range(self.num_residual_blocks):
            x = self.residual_blocks[i](x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def __getstate__(self):
        state = self.__dict__.copy()
        return state



# This is a simpler feedforward neural network
class SimpleHexNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers, activation_functions, device="cpu"):
        super(SimpleHexNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Constructing the network layers
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(nn.Linear(state_dim, hidden_layers[i]))
            else:
                self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            
            # Add activation function if specified
            if activation_functions[i] is not None:
                self.layers.append(activation_functions[i]())
        
        # Output layer for policy
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], action_dim),
            nn.Softmax(dim=1)
        )
        # Output activation function for policy and loss function for policy
        self.policy_loss_function = nn.CrossEntropyLoss()

        # Output layer for value
        self.value_head = nn.Sequential(
                nn.Linear(hidden_layers[-1], 1), # now the output is a single value, we want it between 0 and 1
                nn.Sigmoid()
                )

        # Loss function for value
        self.value_loss_function = nn.MSELoss()

        # save all the other parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions
        self.device = device

        self.to(device)

        

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        action = self.policy_head(x) # shape (batch_size, action_dim)
        value = self.value_head(x) # shape (batch_size, 1)
        return action, value

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

