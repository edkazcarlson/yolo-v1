from dataLoaders import VOC_Detection_Categories

def getBasicAnnotation():
    imageWidth = 448 * 2
    imageHeight = 448 * 3
    target = {}
    target['annotation'] = {}
    target['annotation']['size'] = {}
    target['annotation']['size']['width'] = str(imageWidth)
    target['annotation']['size']['height'] = str(imageHeight)
    target['annotation']['object'] = []

    firstObject = {}
    firstObject['bndbox'] = {}
    firstObject['bndbox']['xmin'] = '100'
    firstObject['bndbox']['xmax'] = '200'
    firstObject['bndbox']['ymin'] = '99'
    firstObject['bndbox']['ymax'] = '300'
    firstObject['name'] = VOC_Detection_Categories[0]

    secondObject = {}
    secondObject['bndbox'] = {}
    secondObject['bndbox']['xmin'] = '200'
    secondObject['bndbox']['xmax'] = '300'
    secondObject['bndbox']['ymin'] = '210'
    secondObject['bndbox']['ymax'] = '330'
    secondObject['name'] = VOC_Detection_Categories[1]

    target['annotation']['object'].append(firstObject)
    target['annotation']['object'].append(secondObject)
    return target