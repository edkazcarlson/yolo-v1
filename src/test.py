import torch
import torchvision

def defaultTargetTransform(target, desiredWidth = 448, desiredHeight = 448):
	"""
    Target that is being transformed
    desiredWidth is the desired width
    desiredHeight is the desired height
    """
	xTransform = desiredWidth / int(target['annotation']['size']['width'])
	target['annotation']['size']['width'] = desiredWidth
	yTransform = desiredHeight / int(target['annotation']['size']['height'])
	target['annotation']['size']['height'] = desiredHeight
	for obj in target['annotation']['object']:
		obj['bndbox']['xmin'] = float(obj['bndbox']['xmin']) * xTransform
		obj['bndbox']['xmax'] = float(obj['bndbox']['xmax']) * xTransform
		obj['bndbox']['ymin'] = float(obj['bndbox']['ymin']) * yTransform
		obj['bndbox']['ymax'] = float(obj['bndbox']['ymax']) * yTransform
	return target

data = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='trainval', download=False, target_transform=defaultTargetTransform)
for x in data:
	print('-')
	break