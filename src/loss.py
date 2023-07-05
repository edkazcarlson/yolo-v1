import torch
import torch.nn as nn
import torchvision
from config import YoloConfig

boxPredictionSize = 5
lambda_coord = 5.
lambda_noobj = .5

def yolo_loss(yhat, y, config: YoloConfig):
	"""
	Args:
        canonical shape:
		yhat: [#, 7, 7, 30]
		y: [#, 7, 7, 30]
	Returns:
		loss: [#]
	"""
	cellsPerAxis = config.cellsPerAxis
	boxesPerCell = config.boxesPerCell
	with torch.no_grad():
		# arrange cell xidx, yidx
		# [7, 7]
		cell_xidx = torch.arange(cellsPerAxis**2).reshape(cellsPerAxis, cellsPerAxis)
		cell_yidx = torch.arange(cellsPerAxis**2).reshape(cellsPerAxis, cellsPerAxis)
		# transform to [7, 7, 2]
		cell_xidx.unsqueeze_(-1)
		cell_yidx.unsqueeze_(-1)
		cell_xidx.expand(cellsPerAxis, cellsPerAxis, boxesPerCell)
		cell_yidx.expand(cellsPerAxis, cellsPerAxis, boxesPerCell)
		# move to device
		cell_xidx = cell_xidx.to(yhat.device)
		cell_yidx = cell_yidx.to(yhat.device)

	def calc_coord(val):
		"""
		transform cell relative coordinates to image relative coordinates
		"""
		with torch.no_grad():
			x = (val[..., 0] + cell_xidx) / cellsPerAxis
			y = (val[..., 1] + cell_yidx) / cellsPerAxis

			return (x - val[..., 2] / 2.0,
				x + val[..., 2] / 2.0,
				y - val[..., 3] / 2.0,
				y + val[..., 3] / 2.0)

	y_area = y[..., :boxPredictionSize * boxesPerCell].reshape(-1, cellsPerAxis, cellsPerAxis, boxesPerCell, boxPredictionSize)
	yhat_area = yhat[..., :boxPredictionSize * boxesPerCell].reshape(-1, cellsPerAxis, cellsPerAxis, boxesPerCell, boxPredictionSize)

	y_class = y[..., boxPredictionSize * boxesPerCell:].reshape(-1, cellsPerAxis, cellsPerAxis, config.categoriesCount)
	yhat_class = yhat[..., boxPredictionSize * boxesPerCell:].reshape(-1, cellsPerAxis, cellsPerAxis, config.categoriesCount)

	with torch.no_grad():
		# calculate IoU
		x_min, x_max, y_min, y_max = calc_coord(y_area)
		x_min_hat, x_max_hat, y_min_hat, y_max_hat = calc_coord(yhat_area)

		#get the smallest edges of the box
		wi = torch.min(x_max, x_max_hat) - torch.max(x_min, x_min_hat)
		wi = torch.max(wi, torch.zeros_like(wi)) #inner width
		hi = torch.min(y_max, y_max_hat) - torch.max(y_min, y_min_hat)
		hi = torch.max(hi, torch.zeros_like(hi)) #inner height

		intersection = wi * hi
		union = (x_max - x_min) * (y_max - y_min) + (x_max_hat - x_min_hat) * (y_max_hat - y_min_hat) - intersection
		iou = intersection / (union + 1e-6) # add epsilon to avoid nan
		
		_, res = iou.max(dim=3, keepdim=True)
	
	# [#, 7, 7, 5]
	# responsible bounding box (having higher IoU)
	yhat_res = torch.take_along_dim(yhat_area, res.unsqueeze(3), 3).squeeze_(3)
	y_res = y_area[..., 0, :5]

	with torch.no_grad():
		# calculate indicator matrix
		have_obj = y_res[..., 4] > 0
		no_obj = ~have_obj
	
	return ((lambda_coord * ( # coordinate loss
		  (y_res[..., 0] - yhat_res[..., 0]) ** 2 # X
		+ (y_res[..., 1] - yhat_res[..., 1]) ** 2 # Y
		+ (torch.sqrt(y_res[..., 2]) - torch.sqrt(yhat_res[..., 2])) ** 2  # W
		+ (torch.sqrt(y_res[..., 3]) - torch.sqrt(yhat_res[..., 3])) ** 2) # H
		# confidence
		+ (y_res[..., 4] - yhat_res[..., 4]) ** 2
		# class
		+ ((y_class - yhat_class) ** 2).sum(dim=3)) * have_obj
		# noobj
		+ ((y_area[..., 0, 4] - yhat_area[..., 0, 4]) ** 2 + \
		(y_area[..., 1, 4] - yhat_area[..., 1, 4]) ** 2) * no_obj * lambda_noobj).sum(dim=(1, 2))

