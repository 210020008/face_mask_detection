import numpy as np
import cv2
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

print(with_mask.shape)
print(without_mask.shape)

with_mask = with_mask.reshape(200, 50*50*3)
without_mask = without_mask.reshape(200, 50*50*3)

x = np.r_[with_mask, without_mask]

labels = np.zeros(x.shape[0])
# labels = np.zeros(400)
labels[200:] = 1.0

x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.30)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
svm = SVC()
svm.fit(x_train, y_train)

x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)

print(accuracy_score(y_test, y_pred))

names = {0: 'Mask', 1: 'No Mask'}
haar_data = cv2.CascadeClassifier('f2.xml')

capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            face = pca.transform(face)  # Apply PCA transformation
            predt = svm.predict(face)[0]
            n = names[int(predt)]
            cv2.putText(img, n, (x, y), font, 1, (244, 255, 255), 2)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27:
            break

capture.release()
cv2.destroyAllWindows()
