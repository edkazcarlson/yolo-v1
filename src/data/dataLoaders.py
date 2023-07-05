# https://github.com/JeffersonQin/yolo-v1-pytorch
import sys

sys.path.append('..')
from config import YoloConfig

import torch
import torchvision
from torch.utils import data
import math
from targetTransform import defaultTargetTransform

VOC_Detection_Categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


class VOCDataset(data.Dataset):
	def __init__(self, dataset, config: YoloConfig) -> None:
		super().__init__()
		self.dataset = dataset
		self.cellsPerAxis = config.cellsPerAxis
		self.boxesPerCell = config.boxesPerCell
		self.cellPredictionSize = self.boxesPerCell * 5 + len(VOC_Detection_Categories) # 5 is the x,y,h,w, predicted IoU

	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, index):
		"""
		input: int index into the dataset\n
		output: (image, label) tuple\n
		Image is a 488 x 488 pixel image\n
		Label is a tuple of shape (cellsPerAxes, cellsPerAxes, cellPredictionSize)\n
		The coordinates are relative to the cell 
		The canonical cellsPerAxes = 7 cellPredictionSize = 30 (2 boxes of 5 predictions + 20 long one hot prediction for class)
		"""
		img, label = self.dataset.__getitem__(index)
		imgWidth = int(label['annotation']['size']['width'])
		cellWidth = imgWidth / self.cellsPerAxis
		imgHeight = int(label['annotation']['size']['height'])
		cellHeight = imgHeight / self.cellsPerAxis

		outputLabel = torch.zeros((self.cellsPerAxis, self.cellsPerAxis, self.cellPredictionSize))
		print(f'found {len(label["annotation"]["object"])} objects')
		for obj in label['annotation']['object']:
			xmin = float(obj['bndbox']['xmin']) #absolute coordinates
			ymin = float(obj['bndbox']['ymin'])
			xmax = float(obj['bndbox']['xmax'])
			ymax = float(obj['bndbox']['ymax'])
			name = obj['name']

			print(obj['bndbox'])

			if xmin == xmax or ymin == ymax:
				continue
			
			xCenter = (xmin + xmax) / 2.0 # center of the bounding box
			yCenter = (ymin + ymax) / 2.0

			width = xmax - xmin
			height = ymax - ymin

			xidx = int(xCenter // cellWidth)
			xPosInCell = (xCenter % cellWidth) / cellWidth
			assert xidx >= 0
			assert xidx < self.cellsPerAxis
			assert xPosInCell >= 0 
			assert xPosInCell <= 1 

			yidx = int(yCenter // cellWidth)
			yPosInCell = (yCenter % cellHeight) / cellHeight
			assert xidx >= 0
			assert xidx < self.cellsPerAxis
			assert yPosInCell >= 0 
			assert yPosInCell <= 1 

			# According to the paper
			# if multiple objects exist in the same cell
			# pick the one with the largest area
			if outputLabel[yidx][xidx][4] == 1: # already have object
				if outputLabel[yidx][xidx][2] * outputLabel[yidx][xidx][3] < width * height:
					use_data = True
				else: use_data = False
			else: use_data = True

			possibleOffsets = [x * 5 for x in range(self.boxesPerCell)]
			if use_data:
				for offset in possibleOffsets:
					# Transforming image relative coordinates to cell relative coordinates:
					# x - idx / 7.0 = x_cell / cell_count (7.0)
					# => x_cell = x * cell_count - idx = x * 7.0 - idx
					# y is the same
					outputLabel[yidx][xidx][0 + offset] = xPosInCell
					outputLabel[yidx][xidx][1 + offset] = yPosInCell
					outputLabel[yidx][xidx][2 + offset] = width
					outputLabel[yidx][xidx][3 + offset] = height
					outputLabel[yidx][xidx][4 + offset] = 1 #target predicted IoU will be 1 since every boxes IoU with itself is 1.
				outputLabel[yidx][xidx][(5 * self.boxesPerCell) + VOC_Detection_Categories.index(name)] = 1

		return img, outputLabel
	

# Sample output from an iteration of the VOCDetection Dataset
# (<PIL.Image.Image image mode=RGB size=500x375 at 0x7FC05FFCBB20>, {'annotation': {'folder': 'VOC2007', 'filename': '000005.jpg', 'source': {'database': 'The VOC2007 Database', 'annotation': 'PASCAL VOC2007', 'image': 'flickr', 'flickrid': '325991873'}, 'owner': {'flickrid': 'archintent louisville', 'name': '?'}, 'size': {'width': '500', 'height': '375', 'depth': '3'}, 'segmented': '0', 'object': [{'name': 'chair', 'pose': 'Rear', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '263', 'ymin': '211', 'xmax': '324', 'ymax': '339'}}, {'name': 'chair', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '165', 'ymin': '264', 'xmax': '253', 'ymax': '372'}}, {'name': 'chair', 'pose': 'Unspecified', 'truncated': '1', 'difficult': '1', 'bndbox': {'xmin': '5', 'ymin': '244', 'xmax': '67', 'ymax': '374'}}, {'name': 'chair', 'pose': 'Unspecified', 'truncated': '0', 'difficult': '0', 'bndbox': {'xmin': '241', 'ymin': '194', 'xmax': '295', 'ymax': '299'}}, {'name': 'chair', 'pose': 'Unspecified', 'truncated': '1', 'difficult': '1', 'bndbox': {'xmin': '277', 'ymin': '186', 'xmax': '312', 'ymax': '220'}}]}})


def load_data_voc(batch_size, download=False, test_shuffle=True, trans = None, targetTrans = None):
	"""
	Loads the Pascal VOC dataset.
	:return: train_iter, test_iter, test_raw_iter
	"""
	# Load the dataset
	if trans == None:
		trans = [
			torchvision.transforms.Resize(448,448),
			torchvision.transforms.ToTensor(),
		]
	if targetTrans == None:
		targetTrans = [
			defaultTargetTransform(448, 448),
		]
	trans = torchvision.transforms.Compose(trans)
	voc2007_trainval = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='trainval', download=download, transform=trans)
	voc2007_test = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='test', download=download, transform=trans)
	voc2012_train = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='train', download=download, transform=trans)
	voc2012_val = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='val', download=download, transform=trans)
	return (
		data.DataLoader(VOCDataset(data.ConcatDataset([voc2007_trainval, voc2007_test, voc2012_train]), train=True), 
			batch_size, shuffle=True), 
		data.DataLoader(VOCDataset(voc2012_val, train=False), 
			batch_size, shuffle=test_shuffle),
		data.DataLoader(VOCDataset(voc2012_val), 
			batch_size, shuffle=test_shuffle)
	)
