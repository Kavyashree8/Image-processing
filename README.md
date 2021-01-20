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
