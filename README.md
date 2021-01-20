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

![prt](https://user-images.githubusercontent.com/75052954/105162953-207db380-5ac8-11eb-9c29-5372ebbadc21.png)

import cv2
import numpy as np 
img=cv2.imread("flower.jpg")
(height,width)=img.shape[:2]
res=cv2.resize(img,(int(width/2),int(height/2)),interpolation=cv2.INTER_CUBIC)
cv2.imwrite("result.jpg",res)
cv2.imshow("image",img)
cv2.imshow("result",res)
cv2.waitKey(0) 

![otpt](https://user-images.githubusercontent.com/75052954/105164668-4015db80-5aca-11eb-87b3-337449a1d05a.png)
![image otpt](https://user-images.githubusercontent.com/75052954/105165140-ccc09980-5aca-11eb-91e8-908819009f9a.png)


import cv2
import numpy as np   
img = cv2.imread("flower.jpg") 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow("result.jpg", res) 
cv2.waitKey(0)


![output](https://user-images.githubusercontent.com/75052954/105166357-3ab99080-5acc-11eb-9a7a-6b01517a9da0.PNG)

import cv2
img = cv2.imread('f2.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image',gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

![gray](https://user-images.githubusercontent.com/75052954/105166952-f8448380-5acc-11eb-9a95-3084923e3f8b.PNG)
![binary](https://user-images.githubusercontent.com/75052954/105166971-fd093780-5acc-11eb-94c9-8262a6ac51f4.PNG)

import cv2
img = cv2.imread(&#39;pet.jpg&#39;)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow(&#39;GRAY image&#39;,gray)
cv2.waitKey(0)
cv2.imshow(&#39;HSV image&#39;,hsv)
cv2.waitKey(0)
cv2.imshow(&#39;LAB image&#39;,lab)
cv2.waitKey(0)
cv2.imshow(&#39;HLS image&#39;,hls)
cv2.waitKey(0)
cv2.imshow(&#39;YUV image&#39;,yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.destroyAllWindows()
