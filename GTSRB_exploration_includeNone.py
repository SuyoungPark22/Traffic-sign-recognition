import matplotlib.pyplot as plt
import csv
import numpy as np
import pickle
import matplotlib.image as mpimg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from PIL import Image



# The German Traffic Sign Recognition Benchmark
# readTrafficSigns.py
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 


# step 1. 구글 drive에서 unzip 되어있는 이미지를 읽어 준다.
def readTrafficSigns(rootpath, classForUse):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    num_of_class = 0
    for c in classForUse:
        images.append([])
        labels.append([])
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            # row += 1
            images[num_of_class].append(plt.imread(prefix + row[0])) # the 1th column is the filename
            labels[num_of_class].append(row[7]) # the 8th column is the label
        gtFile.close()
        num_of_class += 1
    return images, labels, num_of_class
    
    
# step 2. 사용할 class의 traffic sign만 불러 온다.
classForUse = np.arange(43)    # 수요에 따라 수정하는 자리
rootpath = '/content/drive/MyDrive/20-w IAB 실무응용/1-german-traffic-sign-detection/GTSRB/Final_Training/Images'
print('Start collecting...')
trafficImages, trafficLabels, classNum = readTrafficSigns(rootpath, classForUse)
print('finish!')
print(len(trafficImages), len(trafficLabels))
print(len(trafficImages[0]))

if classNum == len(classForUse):
    print(classNum, 'types of traffic signs...')
else:
    print('ERROR while reading files from rootpath')


# step 3. image size를 (32,32)로 resize 한다.

# 가로, 세로 중 더 긴변에 맞추어 정사각형으로 만들어 준다.
squareImage = []

squareImage = []
for k in range(classNum):
    squareImage.append([])
    for img in trafficImages[k]:
        a, b, c = img.shape
        if a == b:    # 정사각형
            pass
        elif a > b:   # 세로로 긴 직사각형
          num_append_pixel = a-b
          left_pixel = round(num_append_pixel / 2)
          right_pixel = num_append_pixel - left_pixel
          left_append = np.ones((a, left_pixel, c), np.int32)
          for i in np.arange(0,a):
            left_append[i,:,0] = img[i,0,0]
            left_append[i,:,1] = img[i,0,1]
            left_append[i,:,2] = img[i,0,2]
          right_append = np.ones((a, right_pixel, c), np.int32)
          for j in np.arange(0,a):
            right_append[j,:,0] = img[j,b-1,0]
            right_append[j,:,1] = img[j,b-1,1]
            right_append[j,:,2] = img[j,b-1,2]
          img = np.concatenate((left_append, img, right_append), axis = 1)
        elif a < b:   # 가로로 긴 직사각형
          num_append_pixel = b-a
          up_pixel = round(num_append_pixel / 2)
          down_pixel = num_append_pixel - up_pixel
          up_append = np.ones((up_pixel, b, c), np.int32)
          for i in np.arange(0,b):
            up_append[:,i,0] = img[0,i,0]
            up_append[:,i,1] = img[0,i,1]
            up_append[:,i,2] = img[0,i,2]
          down_append = np.ones((down_pixel, b, c), np.int32)
          for j in np.arange(0,b):
            down_append[:,j,0] = img[a-1,j,0]
            down_append[:,j,1] = img[a-1,j,1]
            down_append[:,j,2] = img[a-1,j,2]
          img = np.concatenate((up_append, img, down_append), axis = 0)
        squareImage[k].append(img)
    print("Numbers of images in class ", classForUse[k], ": ", len(squareImage[k]))
print("Check number of class again: ", len(squareImage))

# np.array를 Image로 변경해 (32,32)로 크기변경 
resized_img = []
for k in range(classNum):
    resized_img.append([])
    for img in squareImage[k]:
      im = Image.fromarray(img.astype(np.uint8))    # array를 Image 형태로,
      new_img = im.resize((32,32))
      array_img = np.array(new_img)    # 다시 array로 변환해서 저장
      resized_img[k].append(array_img)
      # image_sequence = new_img.getdata()
      # image_array = np.array(image_sequence)
      # resized_img[k].append(image_array)
      # resized_img[k].append(new_img)

print("shape of image in numpy array: ", resized_img[0][0].shape)
print(len(resized_img))

# step 4. random image로 구성된 none class를 만들어 준다. 
resized_img.append([])      # 맨 마지막에 -1 class 추가
trafficLabels.append([])
for i in range(2000):
    random_img = np.random.random((32,32,3)) * 255        # random image: traffic sign이 없는 None
    random_img = random_img.astype(np.uint8)
    resized_img[43].append(random_img)
    trafficLabels[43].append(-1)                        # None의 class: -1
classNum += 1                                           # None class까지 합해서 classNum 계산해줌.
print(len(resized_img))   # 44
print(len(trafficLabels))   # 44

print(len(resized_img[43])) # 2000
print(resized_img[43][0].shape)   # (32,32,3)
print(len(trafficLabels[43]))   # 2000
print(trafficLabels[43][:10])


# step 5. train, validation, test set으로 분리해서 pickle file에 저장한다.

# split 70% training 20% validation 10% test set
imageNum = []
for k in range(classNum):
    imageNum.append(len(resized_img[k]))
imageNum = np.array(imageNum)
frac1 = 0.7
L1 = (frac1*imageNum).astype('int')
frac2 = 0.9
L2 = (frac2*imageNum).astype('int')


train_img = []      # 각 class의 train image를 저장한다.
val_img = []
test_img = []

trainImgNum = []    # 각 class의 train image의 개수를 저장한다.
valImgNum = []
testImgNum = []

for k in range(classNum):
    trainVec = resized_img[k][:L1[k]]
    train_img.append(trainVec)
    trainImgNum.append(len(trainVec))
    valVec = resized_img[k][L1[k]:L2[k]]
    val_img.append(valVec)
    valImgNum.append(len(valVec))
    testVec = resized_img[k][L2[k]:]
    test_img.append(testVec)
    testImgNum.append(len(testVec))


print('Number of samples in training set of each class: ', trainImgNum)
print('Number of samples in validation set of each class: ', valImgNum)
print('Number of samples in test set of each class: ', testImgNum)


train_lab = []      # 각 class의 train label을 저장한다.
val_lab = []
test_lab = []

for k in range(classNum):           # 사실 이 정보는 필요없고 그냥 classForUse 만 보존해도 될 것 같긴 하다.. 일단 작성해보고 필요없으면 빼자. 저장하는정보 줄이기
    trainLabVec = trafficLabels[k][:L1[k]]
    train_lab.append(trainLabVec)
    valLabVec = trafficLabels[k][L1[k]:L2[k]]
    val_lab.append(valLabVec)
    testLabVec = trafficLabels[k][L2[k]:]
    test_lab.append(testLabVec)



# Save the data for easy access
pickle_file = 'data_includeNone.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'train_img': train_img,
                'val_img': val_img,
                'test_img': test_img,
                
                'train_lab': train_lab,
                'val_lab': val_lab,
                'test_lab': test_lab,
                
                'classForUse': classForUse,
                'classNum': classNum
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Data cached in pickle file.')
print('File name is ', pickle_file)

