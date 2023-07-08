TODO: 
- see why his example uses math floor x * 7 instead of // 7 



format of a cell is 
0        (x coord of the middle of the bounding box relative to the cell, 
1        y coord of the middle of the bounding box relative to the cell,
2        width in pixels, 
3        height in pixels, 
4        IoU (target is 1 since Iou(x, x) = 1))

to run all tests in dir py -m unittest discover



todo:
- reorg metrics to be a class to stop passing around the config and the other things like the sampling points.