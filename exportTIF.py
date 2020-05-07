import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.metrics import r2_score
from PIL import Image
from osgeo import gdal,osr

# read the form from the existing data
geotransform = gdal.Open("data/water_distance.tif").GetGeoTransform()
band=gdal.Open("data/water_distance.tif").GetRasterBand(1)
wkt = gdal.Open("data/water_distance.tif").GetProjection()
driver = gdal.GetDriverByName("GTiff")
# output address
output_file = "pop_2018_0_somalia.tif"

dst_ds = driver.Create(output_file,
                       band.XSize,
                       band.YSize,
                       1,
                       gdal.GDT_Float32)
# read from the previous output
raw_data=np.loadtxt('test5.out',delimiter=",")
# print(np.shape(raw_data))
new_data=np.exp(raw_data)*0.01
dst_ds.GetRasterBand(1).WriteArray(new_data)
#setting nodata value
dst_ds.GetRasterBand(1).SetNoDataValue(-999)
#setting extension of output raster
# top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
dst_ds.SetGeoTransform(geotransform)
# setting spatial reference of output raster
srs = osr.SpatialReference()
srs.ImportFromWkt(wkt)
dst_ds.SetProjection(srs.ExportToWkt())

dst_ds = None

