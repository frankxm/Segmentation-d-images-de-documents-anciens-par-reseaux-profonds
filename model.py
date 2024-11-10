# -*- coding: utf-8 -*-

import copy
import logging
import os
import sys
import time

import torch
from torch.cuda.amp import autocast
from torch.nn import Module as NNModule

logger = logging.getLogger(__name__)


class DocUFCNModel(NNModule):
    """
    The DocUFCNModel class is used to generate the Doc-UFCN network.
    The class initializes different useful layers and defines
    the sequencing of the defined layers/blocks.
    """

    def __init__(self, no_of_classes, use_amp=False):
        """
        Constructor of the DocUFCNModel class.
        :param no_of_classes: The number of classes wanted at the
                              output of the network.
        :param use_amp: Whether to use Automatic Mixed Precision.
                        Disabled by default
        """
        super(DocUFCNModel, self).__init__()
        self.amp = use_amp
        self.dilated_block1 = self.dilated_block(3, 32)
        self.dilated_block2 = self.dilated_block(32, 64)
        self.dilated_block3 = self.dilated_block(64, 128)
        self.dilated_block4 = self.dilated_block(128, 256)
        # 池化下采样保留局部区域中的最大值，2*2的核，stride为2，降低一半分辨率
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv_block1 = self.conv_block(256, 128)
        self.conv_block2 = self.conv_block(256, 64)
        self.conv_block3 = self.conv_block(128, 32)
        # 没有批归一化层，所以 bias=True（默认为True），意味着卷积层的输出会包含偏置项。这是必要的，因为该层没有批归一化层去替代偏置的作用。
        self.last_conv = torch.nn.Conv2d(64, no_of_classes, 3, stride=1, padding=1)
        self.softmax = torch.nn.Softmax(dim=1)

    @staticmethod
    def dilated_block(input_size, output_size):
        """
        Define a dilated block.
        It consists in 6 successive convolutions with the dilations
        rates [1, 2, 4, 8, 16].
        :param input_size: The size of the input tensor.
        :param output_size: The size of the output tensor.
        :return: The sequence of the convolutions.
        """
        modules = []
        modules.append(
            torch.nn.Conv2d(
                input_size, output_size, 3, stride=1, dilation=1, padding=1, bias=False
            )
        )
        modules.append(torch.nn.BatchNorm2d(output_size, track_running_stats=False))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.Dropout(p=0.4))
        for i in [2, 4, 8, 16]:
            modules.append(
                torch.nn.Conv2d(
                    output_size,
                    output_size,
                    3,
                    stride=1,
                    dilation=i,
                    padding=i,
                    bias=False,
                )
            )
            modules.append(torch.nn.BatchNorm2d(output_size, track_running_stats=False))
            modules.append(torch.nn.ReLU(inplace=True))
            modules.append(torch.nn.Dropout(p=0.4))
        return torch.nn.Sequential(*modules)

    @staticmethod
    def conv_block(input_size, output_size):
        """
        Define a convolutional block.
        It consists in a convolution followed by an upsampling layer.
        :param input_size: The size of the input tensor.
        :param output_size: The size of the output tensor.
        :return: The sequence of the convolutions.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                input_size, output_size, 3, stride=1, padding=1, bias=False
            ),
            # 批归一化层的作用已经类似于偏置项，因为批归一化层会对每个通道做归一化，并且通过可学习的参数对其重新进行缩放和偏移
            torch.nn.BatchNorm2d(output_size, track_running_stats=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.4),
            # 上采样，分辨率提高为两倍 2为核大小，stride为步长 Hout=(Hin-1)*s-2p+k 反卷积具有可学习的权重能更好恢复特征细节。
            # 反卷积通过学习权重能够在上采样的过程中提取新的、有用的特征，从而增强输出的表现力。而插值方法只能基于现有的像素值，不能引入新的特征信息
            torch.nn.ConvTranspose2d(output_size, output_size, 2, stride=2, bias=False),
            torch.nn.BatchNorm2d(output_size, track_running_stats=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.4),
        )

    def forward(self, input_tensor,step='Validation'):
        """
        Define the forward step of the network.
        It consists in 4 successive dilated blocks followed by 3
        convolutional blocks, a final convolution and a softmax layer.
        :param input_tensor: The input tensor.
        :return: The output tensor.
        """
        with autocast(enabled=self.amp):
            tensor = self.dilated_block1(input_tensor)
            if torch.isnan(tensor).any():
                print("NaN detected after dilated_block1")
            out_block1 = tensor
            tensor = self.dilated_block2(self.pool(tensor))
            if torch.isnan(tensor).any():
                print("NaN detected after dilated_block2")
            out_block2 = tensor
            tensor = self.dilated_block3(self.pool(tensor))
            if torch.isnan(tensor).any():
                print("NaN detected after dilated_block3")
            out_block3 = tensor
            tensor = self.dilated_block4(self.pool(tensor))
            if torch.isnan(tensor).any():
                print("NaN detected after dilated_block4")
            tensor = self.conv_block1(tensor)
            if torch.isnan(tensor).any():
                print("NaN detected after conv_block1")
            tensor = torch.cat([tensor, out_block3], dim=1)
            tensor = self.conv_block2(tensor)
            if torch.isnan(tensor).any():
                print("NaN detected after conv_block2")
            tensor = torch.cat([tensor, out_block2], dim=1)
            tensor = self.conv_block3(tensor)
            if torch.isnan(tensor).any():
                print("NaN detected after conv_block3")
            tensor = torch.cat([tensor, out_block1], dim=1)
            output_tensor = self.last_conv(tensor)
            if torch.isnan(output_tensor).any():
                print("NaN detected after last_conv")
            if step=='Training':
                return output_tensor
            else:
                return self.softmax(output_tensor)




def weights_init(model):
    """
    Initialize the model weights.
    :param model: The model. 相比于上面的方法，考虑了偏置项和各种层
    """
    if isinstance(model, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        torch.nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            # 如果存在偏置项（bias），则将其初始化为零（nn.init.constant_）
            torch.nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight.data)
        if model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, torch.nn.BatchNorm2d):
        # 初始化批归一化层（BatchNorm2d）：将权重初始化为1 将偏置初始化为零
        torch.nn.init.constant_(model.weight.data, 1)
        torch.nn.init.constant_(model.bias.data, 0)

def load_network(no_of_classes: int, use_amp: bool):

    # Define the network.
    net = DocUFCNModel(no_of_classes, use_amp)
    # Allow parallel running if more than 1 gpu available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Running on %s", device)
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs", torch.cuda.device_count())
        net = torch.nn.DataParallel(net)
    return net.to(device)


def restore_model(
    net, optimizer, scaler, model_path: str, keep_last: bool = True
):

    starting_time = time.time()
    if not os.path.isfile(model_path):
        logger.error("No model found at %s",  model_path)
        sys.exit()
    else:
        if torch.cuda.is_available():
            checkpoint = torch.load( model_path)
        else:
            checkpoint = torch.load( model_path, map_location=torch.device("cpu") )
        loaded_checkpoint = {}

        if torch.cuda.device_count() > 1:
            for key in checkpoint["state_dict"].keys():
                if "module" not in key:
                    loaded_checkpoint["module." + key] = checkpoint["state_dict"][key]
                else:
                    loaded_checkpoint = checkpoint["state_dict"]
        else:
            for key in checkpoint["state_dict"].keys():
                loaded_checkpoint[key.replace("module.", "")] = checkpoint[
                    "state_dict"
                ][key]

        if keep_last:
            net.load_state_dict(loaded_checkpoint)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None:
                scaler.load_state_dict(checkpoint["scaler"])
        # 前后使用的类别数不同，所以迁移学习中把最后一层权重偏移去掉
        else:
            loaded_checkpoint.pop("last_conv.weight")
            loaded_checkpoint.pop("last_conv.bias")
            # 当调整模型架构（例如移除或替换某些层）时，可以使用 strict=False 以加载与新的模型架构兼容的部分参数
            net.load_state_dict(loaded_checkpoint, strict=False)

        logger.info(
            "Loaded checkpoint %s (epoch %d) in %1.5fs",
            model_path,
            checkpoint["epoch"],
            (time.time() - starting_time),
        )
        return checkpoint, net, optimizer, scaler


def save_model(epoch: int, model, loss: float, optimizer, scaler, filename: str):

    model_params = {
        "epoch": epoch,
        "state_dict": copy.deepcopy(model),
        "best_loss": loss,
        "optimizer": copy.deepcopy(optimizer),
        "scaler": scaler,
    }
    torch.save(model_params, filename)
