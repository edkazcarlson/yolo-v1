import argparse
from config import YoloConfig


VocName = 'Voc'

dataSetToClasses = [VocName, 20]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cellsPerAxis', type = int, default=7)
    parser.add_argument('-boxesPerCell', type = int, default=2)
    parser.add_argument('-dataSet', type = str, choices=(VocName))

    args = parser.parse_args()
    yoloConfig = YoloConfig(cellsPerAxis = args.cellsPerAxis, boxesPerCell=args.boxesPerCell, categoriesCount=dataSetToClasses[args.dataSet])
    