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
from osgeo import gdal

# # score function (show all score methods)
# print(sorted(sklearn.metrics.SCORERS.keys()))

#the participated covariate in the random forest
# classes=['built','DEM','health_facility','landcover','nightlight','Precipitation','water','temperature','school']
classes=['built','health_facility','nightlight','Precipitation','water','temperature','school','road','river','DEM','slope','landcover']



# Read CSV file
RawData=pd.read_csv('pop_2018_0.csv')
# convert train x to numpy array
pop_data=RawData[classes].to_numpy()
# pop_target=RawData[['pop_density']].to_numpy()
# convert train y to numpy array
pop_target=np.log(RawData[['pop_density_2018']].to_numpy())

# if you want to split data to train and test, in this project, oob are used for test
# Xtrain, Xtest, Ytrain, Ytest =train_test_split(pop_data,pop_target,test_size=0.3)


# initiate the regressor
regressor = RandomForestRegressor(n_estimators=1000, random_state=19, oob_score=True, criterion="mse")

# if you assign test and train, use this.
# regressor.fit(Xtrain,Ytrain)
# score_r = regressor.score(Xtest,Ytest)
# print(score_r)
# print(r2_score(Ytest,regressor.predict(Xtest)))

# fit the model
regressor.fit(pop_data,pop_target)


# this session is to test which random state will result in a better test score
# max=0
# maxnum=0
# for i in range(100):
#     regressor = RandomForestRegressor(n_estimators=1000, random_state=i, oob_score=True, criterion="mse")
#     # regressor.fit(pop_data, pop_target)
#     # if(r2_score(pop_target,regressor.predict(pop_data))>max):
#     #     max=r2_score(pop_target,regressor.predict(pop_data))
#     #     maxnum=i
#     regressor.fit(Xtrain, Ytrain)
#     if (r2_score(Ytest, regressor.predict(Xtest)) > max):
#         max = r2_score(Ytest, regressor.predict(Xtest))
#         maxnum = i
#
# print(max)
# print(maxnum)


# this session in to read tif and convert to numpy array
# numcount=0
#
# # Read TIF
# imWater = gdal.Open("data/water_distance.tif").ReadAsArray()
# imBuilt = gdal.Open("data/built.tif").ReadAsArray()
# imEdu = gdal.Open("data/education_distance.tif").ReadAsArray()
# imHealth = gdal.Open("data/health_distance.tif").ReadAsArray()
# imNight = gdal.Open("data/nightlight.tif").ReadAsArray()
# imPer = gdal.Open("data/percipitation.tif").ReadAsArray()
# imTem = gdal.Open("data/temperature.tif").ReadAsArray()
# imRiver = gdal.Open("data/river_distance.tif").ReadAsArray()
# imRoad = gdal.Open("data/road_distance.tif").ReadAsArray()
# imDEM = gdal.Open("data/DEM2.tif").ReadAsArray()
# imlandcover = gdal.Open("data/landcover.tif").ReadAsArray()
# imslope = gdal.Open("data/slope.tif").ReadAsArray()
# Tifshape=np.shape(imWater)
# print(Tifshape)
# result = np.copy(imWater)
#
# # This part is to write the result
# for i in range(Tifshape[0]):
#     numcount+=1
#     dataPre = np.transpose([imBuilt[i, :], imHealth[i,:], imNight[i,:], imPer[i,:], imWater[i,:], imTem[i,:], imEdu[i,:],imRoad[i,:],imRiver[i,:],imDEM[i,:],imslope[i,:],imlandcover[i,:]])
#     prediction=regressor.predict(dataPre)
#     result[i, :]=prediction
#     print(numcount)
#
#
#
# # save in case
# np.savetxt('test5.out', result, delimiter=',')

# print the feature importance and sort from high to low
print(regressor.feature_importances_)
var_importance=regressor.feature_importances_
var_importance_sort=np.sort(var_importance)[::-1]
for i in range(len(classes)):
    # print(np.where(var_importance == var_importance_sort[i])[0][0])
    print(classes[np.where(var_importance == var_importance_sort[i])[0][0]]+": "+ str(var_importance_sort[i]))

# two metric oob score and r2
print(regressor.oob_score_)
print(r2_score(pop_target,regressor.predict(pop_data)))

# draw by matplotlib
m, b = np.polyfit(regressor.predict(pop_data),pop_target,1)
print(m,b)
plt.plot(regressor.predict(pop_data),pop_target,'o')
plt.plot(regressor.predict(pop_data), m*regressor.predict(pop_data) + b)
plt.show()





# rfr_s = cross_val_score(regressor,pop_data,pop_target,cv=10,scoring="neg_mean_absolute_error")
# print(rfr_s)