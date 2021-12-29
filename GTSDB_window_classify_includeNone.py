import GTSDB_SlidingWindow as sw
import numpy as np
import cv2
import csv
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import time
from PIL import Image
from sklearn.svm import LinearSVC
from PIL import Image
from google.colab.patches import cv2_imshow
from GTSRB_lesson_functions import *


# input으로 주어진 image의 window를 만들어서, 각 window를 test img로 삼아 classification을 진행한다.
# 각기 다른 size와 overlap을 가진 window를 만든다. 한 4세트 정도?
# input: GTSDB 데이터 중 하나. 모든 GTSDB에 대해 수행하고 accuracy 구하면 물론 좋겠으나, 계산량이 버티지 못할 듯. 이래서 batch size를 지정하나보다.
# output: input image에 포함된 모든 교통표지의 class를 원소로 가지는 list. 먄약 교통표지가 없으면, None.
# accuracy: label과 output(predict)를 비교. => 현재는 일단 구하지 말자. ram과 batch size에 대해 해결하기 전까지는.


# step 0. Load the classifier and parameters
data_file = 'ClassifierData_includeNone.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)
    
svc = data['svc'] 
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data ['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data ['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']


# step 1. 구글 drive에서 unzip 되어있는 GTSDB 이미지를 읽어 준다.
def readRoadViews(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    num_of_image = 0
    gtFile = open(rootpath + '/gt.txt')
    gtReader = csv.reader(gtFile, delimiter=';')
    beforeImgName = 'init'
    for row in gtReader:
        if row[0] != beforeImgName:
            img = cv2.imread(rootpath+'/'+row[0],  cv2.IMREAD_COLOR)
            img = np.array(img)
            images.append(img)      # images는 list, 그 원소들은 array
            labels.append(np.array([row[5]]))       # labels는 list, 그 원소들은 array
            num_of_image += 1
        else:
            labels[-1] = np.append(labels[-1], np.array([row[5]]))
        beforeImgName = row[0]
        
    return images, labels, num_of_image

rootpath = '/content/drive/MyDrive/20-w IAB 실무응용/1-german-traffic-sign-detection/FullIJCNN2013'
print('Start collecting...')
roadImages, roadLabels, imageNum = readRoadViews(rootpath)
print('finish!')
if len(roadImages)==len(roadLabels) and len(roadImages)==imageNum:
    print(imageNum, ' road images are loaded')    # 741
else:
    print('ERROR while reading files from rootpath')
    
for i in range(100,110):
  # print('shape of ', i, 'th image: ', roadImages[i].shape)    # (800, 1360, 3)
  print('class of traffic signs that are included in ', i, 'th image: ', roadLabels[i])

# 확인해 보니 일부 이미지에 대한 정보가 유실되었다! 예를 들어, '00108.ppm' 이미지는 실제로 존재하나 그에 대한 데이터가 gt.txt에서는 감쪽같이 빠져 있다.
# 그러나, 어차피 상관 없다. 이미지를 읽을 때 순서대로 다 읽은게 아니라, gt.txt에 기술된 이미지에 대해서만 읽었다.
# 즉, 신경쓰지 말고 900개가 아닌 741개 데이터에 대해서만 수행하도록 한다.
# 다만 코드에서 저장된 index와 실제 이미지 파일의 이름이 다름을 주의한다.


# step 2. 900개의 image 각각에 대해 window를 만들고, (32*32*3) size로 만들어 준다.

# 하나의 이미지에 대해, window를 X_val과 같은 형식으로 모아야 한다.
# numpy array, (window 개수, feature 개수)      # feature 개수는 아마도 1836

roadWindows = []
# for k in range(imageNum):   # 0~899. 900번 도는 loop
for k in range(5):
    roadWindows.append([])      # 앞으로 roadWindows[k]에는 k번째 이미지의 다양한 size의 window들이 모두 저장되게 된다.
    windows1 = sw.generate(roadImages[k], sw.DimOrder.HeightWidthChannel, 128, 0.3)      # 일단 sign이 여러 크기의 window에서 중복 인식되는 경우는 생각하지 말자.
    windows2 = sw.generate(roadImages[k], sw.DimOrder.HeightWidthChannel, 64, 0.3)
    windows3 = sw.generate(roadImages[k], sw.DimOrder.HeightWidthChannel, 32, 0.3)
    # windows4 = sw.generate(roadImages[k], sw.DimOrder.HeightWidthChannel, 16, 0.3)
    for window in windows1:
        subset = roadImages[k][ window.indices() ]      # subset, roadImages[k]는 numpy array
        changeSize = cv2.resize(subset, (32,32))
        roadWindows[k].append(changeSize)
        # sub = Image.fromarray(subset.astype(np.uint8))
        # new_sub = sub.resize((32,32))
        # array_sub = np.array(new_sub)
        # roadWindows[k].append(array_sub)   # roadWindows[k]는 list. roadWindows도 list였음.
    for window in windows2:
        subset = roadImages[k][ window.indices() ]
        changeSize = cv2.resize(subset, (32,32))
        roadWindows[k].append(changeSize)
        # sub = Image.fromarray(subset.astype(np.uint8))
        # new_sub = sub.resize((32,32))
        # array_sub = np.array(new_sub)
        # roadWindows[k].append(array_sub)
    for window in windows3:
        subset = roadImages[k][ window.indices() ]
        changeSize = cv2.resize(subset, (32,32))
        roadWindows[k].append(changeSize)
        # sub = Image.fromarray(subset.astype(np.uint8))
        # new_sub = sub.resize((32,32))
        # array_sub = np.array(new_sub)
        # roadWindows[k].append(array_sub)
    # for window in windows4:
    #     subset = roadImages[k][ window.indices() ]
    #     sub = Image.fromarray(subset.astype(np.uint8))
    #     new_sub = sub.resize((32,32))
    #     array_sub = np.array(new_sub)
    #     roadWindows[k].append(array_sub)
    samplePrint = roadWindows[k][0]
    samplePrint = cv2.resize(samplePrint, (128, 128))     # 잘 안 보여서. 조금 키웠음
    cv2_imshow(samplePrint)   # 확인해 보니 window는 잘 나오고 있다. 색도 맞음.
        
print('number of road images: ', len(roadWindows))    # 741
print('number of windows of zeroth road image: ', len(roadWindows[0]))    # 22107(window 4개, 0.5)     # 2740(window 3개, 0.3)


# step 3. 각 window의 feature를 추출한다.
def get_features(files, color_space='RGB', spatial_size=(32, 32),   # files에는 window들을 원소로 갖는 k번째 이미지에 대한 list가 들어감. 즉 roadWindows[k]
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in files:    # file은 3차원 array
        img = Image.fromarray(file.astype(np.uint8))    # 이미지로 바꿔줌
        img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
        
        features.append(img_features)
        # features = np.array(features)   => 의심. 없어야 하는것 같기도함
    return features   # list를 return. 이 list의 원소는 files 내 file(이미지)의 single img features

color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True

# 돌려 보니 gpu ram 또 다 찬다! 741개 이미지 전부 하지 말고, 앞의 10개에 대해서만 해보자.
t=time.time()
window_feat = []
# for k in range(imageNum):
#     windowFeat = get_features(roadWindows[k], color_space, spatial_size,hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
#     window_feat.append(windowFeat)
for k in range(5):
    windowFeat = get_features(roadWindows[k], color_space, spatial_size,hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
    window_feat.append(windowFeat)
t2 = time.time()

print(round(t2-t, 2), 'seconds to extract HOG, spatial and color features...')    # 26.25 sec
print(len(window_feat))    # 741을 기대.
print(len(window_feat[0]))  # 0번째 image의 window 개수를 기대. 즉 22107    # 2740
print(len(window_feat[0][0]))    # feature vector의 길이를 기대. 즉 1836    # 돌려보니 25164 나온다. 무엇인가 잘못되었다.   # (32,32,3) 크기조절 후 1836.

# step 4. 각 image에 대해, 22107개 window에 대해 각각 prediction 진행하고 class 내놓는다. class는 -1~42.
# 각 image 당 길이가 22107

# predict_vec = []
# class_in_img = []
# for k in range(imageNum):   # k번째 road image에 대해
#     class_in_img.append([])
#     pred = svc.predict(window_feat[k])    # k번째 road image에 해당하는 window 22107개에 대한 예측값
#     predict_vec.append(pred)     # k번째 road image에 포함된 window 들에 대한 prediction. 길이 22107인 1차원 array일 것.
#     for c in pred:      # 예측한 하나하나의 class들. 아마도 22107번 도는 loop가 될 것.
#         if c != '-1':       # c는 아마도 int가 아니라 string일 것.
#             class_in_img[k].append(c)

#######(수정) window_feat[k]를 그대로 가지고 prediction 하는게 아니라, vstack 해줘야 함!
feat_len = len(window_feat[0][0])
X_window =[]
for k in range(5):
  X_window.append(np.array([0]*feat_len))
  for j in range(len(window_feat[k])):    # j는 한 이미지당 window 개수
    X_window[k] = np.vstack((X_window[k], window_feat[k][j]))
  X_window[k] = np.delete(X_window[k], 0, axis=0)
  X_window[k] = X_window[k].astype(np.float64)
  print(X_window[k].shape)
print(len(X_window))

predict_vec = []
class_in_img = []
for k in range(5):   # k번째 road image에 대해
    class_in_img.append([])
    pred = svc.predict(X_window[k])    # k번째 road image에 해당하는 window 22107개에 대한 예측값
    predict_vec.append(pred)     # k번째 road image에 포함된 window 들에 대한 prediction. 길이 22107인 1차원 array일 것.
    for c in pred:      # 예측한 하나하나의 class들. 아마도 22107번 도는 loop가 될 것.
        if c != '-1':       # c는 아마도 int가 아니라 string일 것.
            class_in_img[k].append(c)

# step 5. 전체 road image에 대한 accuracy를 계산한다. 점수 계산은 all or nothing, 즉 해당 img에 포함된 class vector를 하나라도 맞추지 못한다면 틀린 것으로 한다.
correct = 0
# for k in range(imageNum):
#     groundTruth = np.array(roadLabels[k]).astype(uint8)
#     prediction = np.array(class_in_img[k]).asatype(uint8)
#     if groundTruth == prediction:
#         correct += 1
# acc = round(correct/imageNum, 4)

for k in range(5):
    groundTruth = np.array(roadLabels[k]).astype(np.uint8)
    prediction = np.array(class_in_img[k]).astype(np.uint8)
    print('grouhdTruth: ', groundTruth)
    print('prediction: ', prediction)
    print('prediction length: ', len(prediction))
    if str(groundTruth) == str(prediction):
        correct += 1
acc = round(correct/5, 4)

print('accuracy in search & classify: ', acc)

