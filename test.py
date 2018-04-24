import face_recognition as fr
import api
from skimage import io as io
from matplotlib import pyplot as plt

img1_r = io.imread('test1.jpg')
img1_a = api.face_alignment(img1_r, scale = 1.05)
plt.imshow(img1_a)
plt.show()

img2_r = io.imread('test2.jpg')
img2_a = api.face_alignment(img2_r, scale = 1.05)
plt.imshow(img2_a)
plt.show()

img3_r = io.imread('test3.jpg')
img3_a = api.face_alignment(img3_r, scale = 1.05)
plt.imshow(img3_a)
plt.show()