# https://github.com/JeffersonQin/yolo-v1-pytorch

import torch
import torchvision
import torch.nn as nn
from config import YoloConfig

class YoloBackbone(nn.Module):
	def __init__(self):
		super(YoloBackbone, self).__init__()
		conv1 = nn.Sequential(
			# [#, 448, 448, 3] => [#, 224, 224, 64]
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 224, 224, 64] => [#, 112, 112, 64]
		pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv2 = nn.Sequential(
			# [#, 112, 112, 64] => [#, 112, 112, 192]
			nn.Conv2d(64, 192, kernel_size=3, padding=1),
			nn.BatchNorm2d(192),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 112, 112, 192] => [#, 56, 56, 192]
		pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		conv3 = nn.Sequential(
			# [#, 56, 56, 192] => [#, 56, 56, 128]
			nn.Conv2d(192, 128, kernel_size=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 56, 56, 128] => [#, 56, 56, 256]
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 56, 56, 256] => [#, 56, 56, 256]
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 56, 56, 256] => [#, 56, 56, 512]
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 56, 56, 512] => [#, 28, 28, 512]
		pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		
		conv4_part = nn.Sequential(
			# [#, 28, 28, 512] => [#, 28, 28, 256]
			nn.Conv2d(512, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 28, 28, 256] => [#, 28, 28, 512]
			nn.Conv2d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True)
		)
		conv4_modules = []
		for _ in range(4):
			conv4_modules.append(conv4_part)
		conv4 = nn.Sequential(
			*conv4_modules,
			# [#, 28, 28, 512] => [#, 28, 28, 512]
			nn.Conv2d(512, 512, kernel_size=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 28, 28, 512] => [#, 28, 28, 1024]
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True)
		)
		# [#, 28, 28, 1024] => [#, 14, 14, 1024]
		pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
		# [#, 14, 14, 1024] => [#, 14, 14, 1024]
		conv5 = nn.Sequential(
			# [#, 14, 14, 1024] => [#, 14, 14, 512]
			nn.Conv2d(1024, 512, kernel_size=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 512] => [#, 14, 14, 1024]
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 1024] => [#, 14, 14, 512]
			nn.Conv2d(1024, 512, kernel_size=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 512] => [#, 14, 14, 1024]
			nn.Conv2d(512, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True)
		)
		self.net = nn.Sequential(conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5)
	
	def forward(self, X):
		return self.net(X)

class Yolo(nn.Module):
	def __init__(self, backbone: YoloBackbone, config: YoloConfig, backbone_out_channels=1024):
		super(Yolo, self).__init__()
		self.backbone = backbone
		self.head = nn.Sequential(
			# [#, 14, 14, ?] => [#, 14, 14, 1024]
			nn.Conv2d(backbone_out_channels, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 14, 14, 1024] => [#, 7, 7, 1024]
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 7, 7, 1024] => [#, 7, 7, 1024]
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 7, 7, 1024] => [#, 7, 7, 1024]
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 7, 7, 1024] => [#, 7*7*1024]
			nn.Flatten(),
			# [#, 7*7*1024] => [#, 4096]
			nn.Linear(7*7*1024, 4096),
			# nn.Dropout(0.5),
			nn.LeakyReLU(0.1, inplace=True),
			# [#, 4096] => [#, 7*7*30]
			nn.Linear(4096, config.cellsPerAxis*config.cellsPerAxis*config.cellSize),
			nn.Sigmoid(), #  normalize to [0, 1]
			# [#, 7*7*30] => [#, 7, 7, 30]
			nn.Unflatten(1, (config.cellsPerAxis, config.cellsPerAxis, config.cellSize))
		)
		self.net = nn.Sequential(self.backbone, self.head)

	def forward(self, X):
		return self.net(X)