VOC_Detection_Categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class YoloConfig:
    def __init__(self, cellsPerAxis = 7,  boxesPerCell = 2, categoriesCount = 20, categories = VOC_Detection_Categories, imageSize: int = 448):
        self.cellsPerAxis = cellsPerAxis
        self.boxesPerCell = boxesPerCell
        self.categoriesCount = categoriesCount
        self.categories = categories
        self.cellTensorSize = boxesPerCell * 5 + categoriesCount
        self.categoriesStartInd = boxesPerCell * 5
        self.imageSize = imageSize
        self.cellPixelLength = imageSize // self.cellsPerAxis
        self.indexOfIoU = 4