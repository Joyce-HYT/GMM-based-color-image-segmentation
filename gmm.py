import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

'''Scenario 1'''
img1 = cv2.imread("soccer1.jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = img1.reshape(225*400, 1)

M1 = GaussianMixture(n_components=2).fit(img1)
sce1 = M1.predict(img1)

# result1 = np.asarray(sce1)
# np.savetxt('sec1_RGBresult.csv', result1, delimiter=',')

# sce1 = sce1.reshape(225, 400) # 再reshape回來
# plt.imshow(sce1)
# plt.show()
#-----------------------------------------------------

'''Scenario 2'''
img2 = cv2.imread("soccer2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = img2.reshape(217*400, 1)

sce2 = M1.predict(img2)
# result2 = np.asarray(sce2)
# np.savetxt('sec2_GRAYresult.csv', result2, delimiter=',')

sce2 = sce2.reshape(217, 400) # 再reshape回來
plt.imshow(sce2)
plt.show()
#-----------------------------------------------------

'''Scenario 3'''
# mix_2_img = np.array(img1)
# mix_2_img = np.append(mix_2_img, img2, 0)
# M2 = GaussianMixture(n_components=2).fit(mix_2_img)
#
# sce3_1 = M2.predict(img1)
# sce3_2 = M2.predict(img2)
#
# result3_1 = np.asarray(sce3_1)
# np.savetxt('sec3_1_GRAYresult.csv', result3_1, delimiter=',')
# result3_2 = np.asarray(sce3_2)
# np.savetxt('sec3_2_GRAYresult.csv', result3_2, delimiter=',')
#
# sce3_1 = sce3_1.reshape(225, 400)
# sce3_2 = sce3_2.reshape(217, 400)
#
# plt.subplot(121) # 列 行 第幾張
# plt.imshow(sce3_1)
# plt.subplot(122)
# plt.imshow(sce3_2)
# plt.show()



