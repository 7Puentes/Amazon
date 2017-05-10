
import gdal
import os
import numpy as np

train_dir = '/Users/Charly/Downloads/train-tif-sample/'

output = open("tati", "wb")

count = 0
for f in os.listdir(train_dir):
    if f.endswith('tif'):

        try:
            ds = gdal.Open(train_dir + f)

            output.write(ds.GetRasterBand(1).ReadAsArray())
            output.write(ds.GetRasterBand(2).ReadAsArray())
            output.write(ds.GetRasterBand(3).ReadAsArray())
            output.write(ds.GetRasterBand(4).ReadAsArray())

            labels = 1234

            output.write(np.array(labels,dtype=np.uint16))

            print "Converted %s" % f
            count +=1
        except Exception as e:
            print e
output.close()
print count