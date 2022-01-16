import torch
import torch.nn as nn


class ConvolutionalNet(nn.Module):
    """Simple conv. net. Convolves the input channels but retains input image width."""
    def __init__(self, num_channels: int, cnn_kernel_size: int, num_conv_channels: int, dropout_probability: float,
                 stride=1):
        super(ConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=1,
                                padding=0, stride=stride)
        self.conv_2 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=stride, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=cnn_kernel_size,
                                stride=stride, padding=cnn_kernel_size // 2)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, image_width * image_width, num_conv_channels]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        conved_1 = self.conv_1(input_images)
        conved_2 = self.conv_2(input_images)
        conved_3 = self.conv_3(input_images)
        images_features = torch.cat([conved_1, conved_2, conved_3], dim=1)
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        images_features = self.layers(images_features)
        return images_features.reshape(batch_size, image_dimension * image_dimension, num_channels)


class DeepConvolutionalNet(nn.Module):
    """Simple conv. net. Convolves the input channels but retains input image width."""
    def __init__(self, num_channels: int, num_conv_channels: int, kernel_size: int, dropout_probability: float,
                 stride=1):
        super(DeepConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=1,
                                padding=0, stride=stride)
        self.conv_2 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=stride, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=stride, padding=2)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, image_width * image_width, num_conv_channels]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        conved_1 = self.conv_1(input_images)
        conved_2 = self.conv_2(input_images)
        conved_3 = self.conv_3(input_images)
        images_features = self.layers(torch.cat([conved_1, conved_2, conved_3], dim=1))
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        return images_features.reshape(batch_size, image_dimension * image_dimension, num_channels)


class DownSamplingConvolutionalNet(nn.Module):
    """TODO: make more general and describe"""
    def __init__(self, num_channels: int, num_conv_channels: int, dropout_probability: float):
        super(DownSamplingConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=5)
        self.conv_2 = nn.Conv2d(in_channels=num_conv_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=3, padding=0)
        self.conv_3 = nn.Conv2d(in_channels=num_conv_channels, out_channels=num_conv_channels, kernel_size=3,
                                stride=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.conv_1, self.relu, self.dropout, self.conv_2, self.relu, self.dropout, self.conv_3,
                  self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.output_dimension = num_conv_channels * 3

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, 6 * 6, output_dim]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        images_features = self.layers(input_images)
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        return images_features.reshape(batch_size, image_dimension, image_dimension, num_channels)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualConvNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvNet, self).__init__()
        self.resblock1 = ResidualBlock(in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 3, 3, stride = 1, padding = 1, bias=False)
        self.relu = nn.ReLU()
        self.resblock2 = ResidualBlock(out_channels // 3, out_channels // 3)
        self.conv2 = nn.Conv2d(out_channels // 3, out_channels, 3, stride = 1, padding = 1, bias=False)

    def forward(self, input_images):
        input_images = input_images.transpose(1, 3)
        out = self.resblock1(input_images)
        out = self.relu(self.conv1(out))
        out = self.resblock2(out)
        out = self.relu(self.conv2(out))
        batch_size, num_channels, _, image_dimension = out.size()
        out = out.transpose(1, 3)
        return out.reshape(batch_size, image_dimension * image_dimension, num_channels)

