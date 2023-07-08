from data.dataLoaders import VOC_Detection_Categories

class YoloConfig:
    def __init__(self, cellsPerAxis = 7,  boxesPerCell = 2, categoriesCount = 20, categories = VOC_Detection_Categories):
        self.cellsPerAxis = cellsPerAxis
        self.boxesPerCell = boxesPerCell
        self.categoriesCount = categoriesCount
        self.categories = categories
        self.cellSize = boxesPerCell * 5 + categoriesCount
        self.categoriesStartInd = boxesPerCell * 5

        self.indexOfIoU = 4