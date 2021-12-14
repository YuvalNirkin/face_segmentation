import numpy as np
import cv2


x = np.load('output/scarlett-johansson_0.npy')
x = x * 255
cv2.imwrite('out.jpg', x)
