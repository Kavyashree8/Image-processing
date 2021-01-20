# Image-processing
import cv2
import numpy as np
image=cv2.imread("flower.jpg")
image=cv2.resize(image,(0,0),None,.95,.95)
grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
grey_3_channel=cv2.cvtColor(grey,cv2.COLOR_GRAY2BGR)
numpy_horizontal=np.hstack((image,grey_3_channel))
numpy_horizontal_concat=np.concatenate((image,grey_3_channel),axis=1)
cv2.imwrite("flower.jpg")
cv2.imshow("flower",numpy_horizontal_concat)
cv2.waitKey()

![image](https://user-images.githubusercontent.com/75052954/105162145-160eea00-5ac7-11eb-8a96-95b24179afdc.png)
