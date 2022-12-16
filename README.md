# 오픈소스SW 기말 과제
## 20221996 김동건

###### Import

```
import os

import sklearn.datasets
import sklearn.linear_model
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics

import skimage.io
import skimage.transform
import skimage.color

import numpy as np

import matplotlib.pyplot as plt 
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
```
###### 데이터 불러오기
```
image_size = 64
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

images = []
y = []
for i in labels:
    folderPath = os.path.join('./tumor_dataset/Training',i)
    for j in os.listdir(folderPath):
        img = skimage.io.imread(os.path.join(folderPath,j),)
        img = skimage.transform.resize(img,(image_size,image_size))
        img = skimage.color.rgb2gray(img)
        images.append(img)
        y.append(i)
        
images = np.array(images)

X = images.reshape((-1, image_size**2))
y = np.array(y)

j = 0
for i in range(len(y)):
    if y[i] in labels[j]:
        plt.imshow(images[i])
        plt.title("[Index:{}] Label:{}".format(i, y[i]))
        plt.show()
        j += 1
    if j >= len(labels):
        break
```
###### 데이터를 학습용과 테스트용으로 분리

```
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
```

sklearn을 이용한 분류 모델들 중 KNN, Extratreesclassifier, SVC 등 여러가지를 이용하였지만 단일 모델로 높은 정확도를 보이지 못하는 것으로 보여 voting을 사용

```
clf1 = ExtraTreesClassifier(n_estimators=1000,
                             criterion = "entropy",
                             max_depth = 20,
                             n_jobs = -1,
                             random_state=10, min_samples_split = 2, min_samples_leaf = 1, max_leaf_nodes = 10)
clf2 = KNeighborsClassifier(n_neighbors = 1,metric = 'minkowski', p = 2, weights = 'uniform')
clf3 = svm.SVC(kernel = 'linear', gamma = 0.01, C = 100)
eclf1 = VotingClassifier(estimators=[('etc', clf1), ('knn', clf2), ('svc', clf3)], voting='hard')
eclf1 = eclf1.fit(X_train, y_train)
y_pred = eclf1.predict(X_test)

print('Accuracy: %.9f' % sklearn.metrics.accuracy_score(y_test, y_pred))
```

###### license
MIT license

###### Contact information

김동건 Kim Donggeon / Moungom / caukdk@gmail.com
