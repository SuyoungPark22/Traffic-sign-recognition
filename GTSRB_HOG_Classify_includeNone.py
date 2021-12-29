import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import sklearn.svm as svm
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib
from GTSRB_lesson_functions import *

print ("start running code...")

# Load the training validation and test data
data_file = 'data_includeNone.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)

train_img = data['train_img']
val_img = data['val_img']
test_img = data['test_img']

train_lab = data['train_lab']
val_lab = data['val_lab']
test_lab = data['test_lab']

classForUse = data['classForUse']
classNum = data['classNum']

print("retrieved data file.")


# print('classForUse: ', classForUse)   # 여기에 -1 넣는거 까먹었음. 근데어차피 앞으로 classForUse 안쓴다
classForUse = np.append(classForUse, -1)    # 기워붙이기..ㅎㅎ
print('clasForUse: ', classForUse)
print('classNum: ', classNum)
print('classNum again: ', len(train_img))

# helper function to extract features from files

def get_features(files, color_space='RGB', spatial_size=(32, 32),
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
    return features


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

t=time.time()

train_feat = []
val_feat = []
test_feat = []

for k in range(classNum):                                             # k번째 class에 담긴 image들의 feature
    trainFeat = get_features(train_img[k],color_space, spatial_size,hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
    train_feat.append(trainFeat)
    valFeat = get_features(val_img[k],color_space, spatial_size,hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
    val_feat.append(valFeat)
    testFeat = get_features(test_img[k],color_space, spatial_size,hist_bins, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
    test_feat.append(testFeat)

t2 = time.time()
print(round(t2-t, 2), 'seconds to extract HOG, spatial and color features...')
print(len(train_feat))    # 44를 기대.
print(len(train_feat[0]))  # 0번째 class의 train image 개수를 기대. 즉 147
print(len(train_feat[0][0]))    # feature vector의 길이를 기대. 즉 1836



feat_len = len(train_feat[0][0])        # vstack 연산을 위해 길이가 1836인 임의의 벡터 만들어줌.
X = np.array([0]*feat_len)
for k in range(classNum):
    X = np.vstack((X, train_feat[k]))
    X = np.vstack((X, val_feat[k]))
    X = np.vstack((X, test_feat[k]))
    # print(X.shape)
X = X.astype(np.float64)
X = np.delete(X, 0, axis=0)             # 임의로 만든 벡터 삭제.
print(X.shape)      # (41209, 1836)을 기대.

# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)  # 이 자체가 뭔가 array를 나타내는 것이 아니라, 후의 계산을 위해 std, mean 등을 계산한 정보를 담고 있음.
# Apply the scaler to X
scaled_X = X_scaler.transform(X)    # scaler에 따라 표준화(standardization) 진행. scaled_X는 X와 형태 똑같음, 값만 표준화됨.



startIndex = 0
for k in range(classNum):
    train_feat[k] = scaled_X[startIndex : startIndex + len(train_feat[k])]
    startIndex += len(train_feat[k])
    val_feat[k] = scaled_X[startIndex : startIndex + len(val_feat[k])]
    startIndex += len(val_feat[k])
    test_feat[k] = scaled_X[startIndex : startIndex + len(test_feat[k])]
    startIndex += len(test_feat[k])


y_train = np.array([])
y_val = np.array([])
y_test = np.array([])
X_train = np.array([0]*feat_len)
X_val = np.array([0]*feat_len)
X_test = np.array([0]*feat_len)

for k in range(classNum):
    y_train = np.hstack((y_train, train_lab[k]))        # hstack은 []뒤에 이어붙여도 문제없음. 차원 문제가 없어서그럼.
    y_val = np.hstack((y_val, val_lab[k]))
    y_test = np.hstack((y_test, test_lab[k]))
    X_train = np.vstack((X_train, train_feat[k]))
    X_val = np.vstack((X_val, val_feat[k]))
    X_test = np.vstack((X_test, test_feat[k]))
X_train = np.delete(X_train, 0, axis=0)
X_val = np.delete(X_val, 0, axis=0)
X_test = np.delete(X_test, 0, axis=0)

print(len(X_train), len(y_train))   # 28846 28846 기대. 플마 44까지 오차가능. # 실제로는 28839 28839 나옴. ok  
print(len(X_val), len(y_val))       # 8249 8249
print(len(X_test), len(y_test))     # 4121 4121
X_train,y_train = shuffle(X_train,y_train,random_state=42)  # sklearn.utils.shuffle의 random_state 파라미터의 의미: 정수를 입력해라. 여기에 입력한 정수는 seed가 된다. 같은 seed를 넣으면 항상 똑같이 shuffle 해 준다.
X_val,y_val = shuffle(X_val,y_val,random_state=42)          # 나란히 섞인다. 즉 label이 유지된 채로 섞인다. 왜? 각각이 아니라, 동시에 shuffle 해주니까..
X_test,y_test = shuffle(X_test,y_test,random_state=42)

print('Using:',orient,'orientations',pix_per_cell,'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', feat_len)   # 1836


# Use a linear SVC 
svc = LinearSVC()

# use of the rbf kernel improves the accuracy by about another percent, 
# but increases the prediction time up to 1.7s(!) for 100 labels. Too slow.
# svc = svm.SVC(kernel='rbf')

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)   # .fit; 이걸로 LinearSVC 모델 학습시켜라
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Validation Accuracy of SVC = ', round(svc.score(X_val, y_val), 4))   #.score; 이걸로 test 하고 accuracy 내놔라.
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 100
print('My SVC predicts: ', svc.predict(X_val[0:n_predict]))   #.predict; 이걸로 test해서 예측한 class 내놔라.
print('For these',n_predict, 'labels: ', y_val[0:n_predict])  # 실제 class(label). 위의 prediction과 비교해 봐라.
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# 시각화는 생략. colab에서 왠지 출력 안 됨.


# wrongImg = []
# wrongImgNum = []
# for k in range(classNum):
#     preds = svc.predict(val_feat[k])
#     inds = np.where(preds != val_lab[k])
#     inds = np.ravel(inds)
#     misclassifieds = [ val_img[k][i] for i in inds ]
#     wrongImg.append(misclassifieds)
#     wrongImgNum.append(len(misclassifieds))
#     print('number of misclassified sign', classForUse[k], ' images: ', len(misclassifieds))
lab0 = np.array([-1]*400)
pred0 = svc.predict(val_feat[43])
inds = np.where(pred0 != lab0)
inds = np.ravel(inds)
print(inds)

# Save the data for easy access
pickle_file = 'ProcessedData_includeNone.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test                
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise
    
print('Data cached in pickle file.')
print('File name is ', pickle_file)

pickle_file = 'ClassifierData_includeNone.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {   'svc':svc, 
                'X_scaler': X_scaler,
                'color_space': color_space,
                'spatial_size': spatial_size,
                'hist_bins': hist_bins,
                'orient': orient,
                'pix_per_cell': pix_per_cell,
                'cell_per_block': cell_per_block,
                'hog_channel': hog_channel,
                'spatial_feat': spatial_feat,
                'hist_feat': hist_feat,
                'hog_feat':hog_feat
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Data cached in pickle file.')
print('File name is ', pickle_file)