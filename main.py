import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from d2l import torch as d2l



class BasicBlock(nn.Module):
    """
        适用于 ResNet-18 和 ResNet-34 的基础残差块。
        """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 残差函数 F(x)
        self.residual_function = nn.Sequential(
            # 提取特征
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # 批量归一化，加速收敛并提高稳定性
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        # 快捷连接
        self.shortcut = nn.Sequential()

        # 如果输入和输出的维度不同（通道数或尺寸），则需要对快捷连接进行变换以匹配维度
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        # 输出是 F(x) + x
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    适用于 ResNet-50, ResNet-101, ResNet-152 的瓶颈残差块。
    """
    expansion = 4  # 输出通道数是 out_channels 的4倍

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 残差函数 F(x)，采用 "降维 -> 卷积 -> 升维" 的瓶颈设计
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        # 快捷连接 (Shortcut / Identity)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        # 1. 网络的输入部分 (Stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 2. 网络的主体，由4个残差阶段构成
        self.layer1 = self._make_layer(block, 64, num_block[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_block[3], stride=2)

        # 3. 网络的输出部分 (Classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 使用 .view 或 .flatten()
        x = self.fc(x)

        return x

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=1000):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=1000):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)


# --- 1. 定义数据预处理的步骤 ---
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. 定义数据集的路径 ---
base_path = r"C:\Users\Administrator\Desktop\test_for_pneumonia\ChestXRay2017\chest_xray"
train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")

# --- 3. 使用 ImageFolder 加载数据集 ---
train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transforms)

# 打印类别和索引，这句可以保留在全局
print("类别及其对应的索引:", train_dataset.class_to_idx)

# --- 4. 定义超参数并创建 DataLoader ---
BATCH_SIZE = 64
NUM_WORKERS = 4

# 创建数据迭代器
train_iter = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=True)

test_iter = DataLoader(dataset=test_dataset,
                       batch_size=BATCH_SIZE,
                       shuffle=False,
                       num_workers=NUM_WORKERS,
                       pin_memory=True)


if __name__ == "__main__":
    print("\n--- 验证 DataLoader ---")

    one_batch_images, one_batch_labels = next(iter(train_iter))

    print("成功创建 train_iter 和 test_iter !")
    print(f"一个训练批次中图像的形状: {one_batch_images.shape}")
    print(f"一个训练批次中标签的形状: {one_batch_labels.shape}")
    print("图像形状解读: [批次大小, 颜色通道数, 高度, 宽度]")

    net = resnet18()


    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 2)

    lr, num_epochs = 0.01, 10

    print("\n--- 开始训练 ---")
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

    print("\n--- 训练完成 ---")

    plt.show()
