# Image-processing
**1. Develop a program to display grayscale image using read and write operation.**
**Description:**
        Grayscaling is the process of converting an image from other color spaces
e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and
complete white. 
imread() : is used for reading an image.
imwrite(): is used to write an image in memory to disk.
imshow() :to display an image.
waitKey(): The function waits for specified milliseconds for any keyboard event.
destroyAllWindows():function to close all the windows.
cv2. cvtColor() method is used to convert an image from one color space to another
    syntax is cv2.cvtColor(Input_image,flag)

**Program:**
```python
import cv2
import numpy as np
image=cv2.imread("flower.jpg")
cv2.imshow("Old",image)
cv2.imshow("Gray",grey)
grey=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray",grey)
cv2.imwrite("flower.jpg",grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
**OUTPUT**

![prt](https://user-images.githubusercontent.com/75052954/105162953-207db380-5ac8-11eb-9c29-5372ebbadc21.png)


**2. Develop a program to perform linear transformations on an image: Scaling and Rotation**
**Description:**
**A)Scaling:**
         Image resizing refers to the scaling of images. Scaling comes handy in many image processing as well as machinelearning applications. It helps in reducing the number of pixels from an image
cv2.resize() method refers to the scaling of images. Scaling comes handy in many image processing as well as machine learning applications. It helps in reducing the
number of pixels from an image 
imshow() function in pyplot module of matplotlib library is used to display data as an image

**program**
```python
import cv2
import numpy as np 
img=cv2.imread("flower.jpg")
(height,width)=img.shape[:2]
res=cv2.resize(img,(int(width/2),int(height/2)),interpolation=cv2.INTER_CUBIC)
cv2.imwrite("result.jpg",res)
cv2.imshow("image",img)
cv2.imshow("result",res)
cv2.waitKey(0) 
```
**OUTPUT**

![otpt](https://user-images.githubusercontent.com/75052954/105164668-4015db80-5aca-11eb-87b3-337449a1d05a.png)
![image otpt](https://user-images.githubusercontent.com/75052954/105165140-ccc09980-5aca-11eb-91e8-908819009f9a.png)

**B)Rotation:** 
         **Description:** 
                Image rotation is a common image processing routine used to rotate images at any desired angle. This helps in image reversal,flipping, and obtaining an intended view of the image. Image rotation has applications in matching, alignment, and other image-based algorithms. OpenCV is a well-known library used for image processing. cv2.getRotationMatrix2D Perform the counter clockwise rotation 
warpAffine() function is the size of the output image, which should be in the form of (width, height).
width = number of columns, and height = number of rows.

**Program**
```python
import cv2
import numpy as np   
img = cv2.imread("flower.jpg") 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow("result.jpg", res) 
cv2.waitKey(0)
```

**output**

![output](https://user-images.githubusercontent.com/75052954/105166357-3ab99080-5acc-11eb-9a7a-6b01517a9da0.PNG)


**4. Develop a program to convert the color image to gray scale and binary image.**
**Description:** 
        Grayscaling is the process of converting an image from other color spaces e.g RGB, CMYK, HSV, etc. to shades of gray. It varies between complete black and
complete white. A binary image is a monochromatic image that consists of pixels that can have one of exactly two colors, usually black and white. 
cv2.threshold works as, if pixel value is greater than a threshold value, it is assigned one value (may be white),else it is assigned another value (may be black). destroyAllWindows() simply destroys all the windows we created. To destroy any specific window, use the function cv2.destroyWindow() where you pass the exact window name.
**Program**
```python
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
```

**OUTPUT**

![gray](https://user-images.githubusercontent.com/75052954/105166952-f8448380-5acc-11eb-9a95-3084923e3f8b.PNG)
![binary](https://user-images.githubusercontent.com/75052954/105166971-fd093780-5acc-11eb-94c9-8262a6ac51f4.PNG)


**5. Develop a program to convert the given color image to different color spaces.**
**Description:**
        Color spaces are a way to represent the color channels present in the image that gives the image that particular hue 
BGR color space: OpenCV’s default color space is RGB. 
HSV color space: It stores color information in a cylindrical representation of RGB color points. It attempts to depict the colors as perceived by the human eye. Hue value varies from 0-179, Saturation value varies from 0-255 and Value value varies from 0-255. 
LAB color space :
        L – Represents Lightness.
        A – Color component ranging from Green to Magenta.
        B – Color component ranging from Blue to Yellow. 
The HSL color space, also called HLS or HSI, stands for:Hue : the color type Ranges from 0 to 360° in most applications Saturation : variation of the color depending
on the lightness. Lightness :(also Luminance or Luminosity or Intensity). Ranges from 0 to 100% (from black to white).
YUV:Y refers to the luminance or intensity, and U/V channels represent color information. This works well in many applications because the
human visual system perceives intensity information very differently from color information.

**Program**
```python
import cv2
img = cv2.imread('f3.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow('GRAY image',gray)
cv2.waitKey(0)
cv2.imshow('HSV image',hsv)
cv2.waitKey(0)
cv2.imshow('LAB image',lab)
cv2.waitKey(0)
cv2.imshow('HLS image',hls)
cv2.waitKey(0)
cv2.imshow('YUV image',yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.destroyAllWindows()
```


**OUTPUT**

![grayimage](https://user-images.githubusercontent.com/75052954/105169236-1bbcfd80-5ad0-11eb-8558-ac8e7a262733.PNG)
![hls](https://user-images.githubusercontent.com/75052954/105169262-25466580-5ad0-11eb-8add-99b779e56158.PNG)
![hsv](https://user-images.githubusercontent.com/75052954/105169304-32fbeb00-5ad0-11eb-8812-2c4a1fc15f12.PNG)
![lab](https://user-images.githubusercontent.com/75052954/105169321-398a6280-5ad0-11eb-8495-9a3d5a9d00de.PNG)
![yuv](https://user-images.githubusercontent.com/75052954/105169331-3e4f1680-5ad0-11eb-9d6d-7bf7f0b83229.PNG)


**6. Develop a program to create an image from 2D array (generate an array of random size).**
**Description**
        2D array can be defined as an array of arrays. The 2D array is organized as matrices which can be represented as the collection of rows and columns. However,
2D arrays are created to implement a relational database look alike data structure.
numpy.zeros() function returns a new array of given shape and type, with zeros.
Image.fromarray(array) is creating image object of above array
**Program**
```python
import numpy as np
from PIL import Image 
import cv2 as c 
array =np.zeros([100, 200, 3], dtype=np.uint8) 
array[:,:100] = [150, 128, 0] #Orange left side
array[:,100:] = [0, 0, 255] #Blue right side
img = Image.fromarray(array) 
img.save('f4.jpg') 
img.show() 
c.waitKey(0)
```

**OUTPUT**

![boutput](https://user-images.githubusercontent.com/75052954/105334646-f8f01f00-5b8b-11eb-820f-59c6b6a2a88c.PNG)


**3. Develop a program to find the sum and mean of a set of images.
        a. Create ‘n’ number of images and read them from the directory and perform the operations.**
 **Description**
 You can add two images with the OpenCV function, cv. add(), or simply by the numpy operation res = img1 + img2. The function mean calculates the mean value M of array elements,
independently for each channel, and return it:" This mean it should return you a scalar for each layer of you image The append() method in python adds a single item to the
existing list. 
listdir() method in python is used to get the list of all files and directories in the specified directory.
**Program**
```python
import cv2 
import os 
path = "D://ff"
imgs = []
files = os.listdir(path) 
for file in files:
    filepath=path+'\\'+file
imgs.append(cv2.imread(filepath)) 
i=0 
im = [] 
for im in imgs:
    cv2.imshow(files[i],imgs[i])
    im+=imgs[i]
    i=i+1 
cv2.imshow('sum of four pictures',im) 
meanImg = im/len(files)
cv2.imshow("mean of four picture",meanImg) 
cv2.waitKey(0)
```

**OUTPUT:**

![sum](https://user-images.githubusercontent.com/75052954/105335198-9d726100-5b8c-11eb-9dbc-f4d1aa82f829.PNG)
![mean](https://user-images.githubusercontent.com/75052954/105335220-a2371500-5b8c-11eb-8ebd-0baab9d922b1.PNG)


**7) Find the Neighbourhood Matrix**
     **Description**
                Description: A pixel's neighborhood is some set of pixels, defined by their locations relative to that pixel, which is called the center pixel. The neighborhood is a rectangular block, and as you move from one element to the next in an image matrix, the neighborhood block slides in the same direction.

**Program**
```python
import numpy as np
axis =3
x=np.empty((axis,axis))
y=np.empty((axis+2,axis+2))
s=np.empty((axis,axis))
print("Matrix\n")
x=np.array([[1,4,3],[2,8,5],[3,4,6]])
for i in range(0,axis):
    for j in range(0,axis):
        print(int(x[i,j]),end= '\t')
    print('\n')
print("\nTemp matrix\n")
for i in range(0,axis+2):
    for j in range(0,axis+2):
        if i==0 or i==axis+1 or j==0 or j==axis+1:
            y[i][j]=0
        else:
            y[i][j]=x[i-1][j-1]
for i in range(0,axis+2):
    for j in range(0,axis+2):
        print(int(y[i][j]),end='\t')
    print('\n')

 ``` 
    **OUTPUT:**
    Matrix
```python

1	4	3	

2	8	5	

3	4	6	


Temp matrix

0	0	0	0	0	

0	1	4	3	0	

0	2	8	5	0	

0	3	4	6	0	

0	0	0	0	0	

```
     
**8) Calculate the Neighbourhood of Matrix**
             **Description**
                Description: Given a M x N matrix, find sum of all K x K sub-matrix 2. Given a M x N matrix and a cell (i, j), find sum of all elements of the matrix in constant time except the elements present at row i & column j of the matrix. Given a M x N matrix, calculate maximum sum submatrix of size k x k in a given M x N matrix in O (M*N) time. Here, 0 < k < M, N.


**Program**
```python
import numpy as np
axis =3
x=np.empty((axis,axis))
r=np.empty((axis+2,axis+2))
x=np.empty((axis,axis))
print("Matrix\n")
x=np.array([[1,4,3],[2,8,5],[3,4,6]])
for i in range(0,axis):
    for j in range(0,axis):
        print(int(x[i,j]),end= '\t')
    print('\n')
print("\nTemp matrix\n")
for i in range(0,axis+2):
    for j in range(0,axis+2):
        if i==0 or i==axis+1 or j==0 or j==axis+1:
            y[i][j]=0
        else:
            y[i][j]=x[i-1][j-1]
for i in range(0,axis+2):
    for j in range(0,axis+2):
        print(int(y[i][j]),end='\t')
    print('\n')
print('Output calculated Neighbours of matrix\n')
print('sum of Neighbours of matrix\n')
for i in range(0,axis):
    for j in range(0,axis):
        r[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2]))
        print(r[i][j],end='\t')
    print('\n')
print('\n Average of Neighbours of matrix\n')
for i in range(0,axis):
    for j in range(0,axis):
        s[i][j]=((y[i][j]+y[i][j+1]+y[i][j+2]+y[i+1][j]+y[i+1][j+2]+y[i+2][j]+y[i+2][j+1]+y[i+2][j+2])/8)
        print(s[i][j],end = '\t')
    print('\n')
```
**OUTPUT**
```pyton
Matrix

1	4	3	

2	8	5	

3	4	6	


Temp matrix

0	0	0	0	0	

0	1	4	3	0	

0	2	8	5	0	

0	3	4	6	0	

0	0	0	0	0	

Output calculated Neighbours of matrix

sum of Neighbours of matrix

14.0	19.0	17.0	

20.0	28.0	25.0	

14.0	24.0	17.0	


 Average of Neighbours of matrix

1.75	2.375	2.125	

2.5	3.5	3.125	

1.75	3.0	2.125	
```

**9) Develop a program to implement Negative Transformation of a image**
**Description:**
  The second linear transformation is negative transformation, which is invert of identity transformation. In negative transformation, each value of the input image is subtracted from the L-1 and mapped onto the output image
**Program**
```python
import cv2 
import matplotlib.pyplot as plt
img_Original = cv2.imread('pic3.jpg', 1) 
plt.imshow(cv2.cvtColor(img_Original,cv2.COLOR_BGR2RGB))
plt.show() 
cv2.waitKey(0)
img_neg = 255 - img_Original 
plt.imshow(img_neg) 
plt.show() 
cv2.waitKey(0)
```
**OUTPUT**


![output](https://user-images.githubusercontent.com/75052954/107626524-c8435880-6c12-11eb-9b3a-5a87b71ab675.JPG)

**Contrast**
        **Description:**
                Description: Contrast can be simply explained as the difference between maximum and minimum pixel intensity in an image.
**Program**
```python
from PIL import Image, ImageEnhance
img = Image.open("pic1.jpeg")
img.show()
img=ImageEnhance.Color(img)
img.enhance(2.0).show()
```
**OUTPUT**

![output2](https://user-images.githubusercontent.com/75052954/107627419-1e64cb80-6c14-11eb-9e7e-ce8828f5eba9.JPG)
![output3](https://user-images.githubusercontent.com/75052954/107627437-245aac80-6c14-11eb-8191-ad4f7cda0629.JPG)


**Thresholding Brightness**
  **Description:**
  Brightness is a relative term. It depends on your visual perception. Since brightness is a relative term, so brightness can be defined as the amount of energy output by a source of light relative to the source we are comparing it to. In some cases we can easily say that the image is bright, and in some cases, its not easy to perceive.

a)**Program:**
 ```python
 import cv2  
import numpy as np  

image1 = cv2.imread('flower1.jpg')  

img = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
 
ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)

cv2.imshow('Binary Threshold', thresh1)
cv2.imshow('Binary Threshold Inverted', thresh2)
cv2.imshow('Truncated Threshold', thresh3)
cv2.imshow('Set to 0', thresh4)
cv2.imshow('Set to 0 Inverted', thresh5)

if cv2.waitKey(0) & 0xff == 27:  
   cv2.destroyAllWindows() 
```
**OUTPUT:**

![output4](https://user-images.githubusercontent.com/75052954/107628324-58829d00-6c15-11eb-8217-771df8536f75.jpg)
![output5](https://user-images.githubusercontent.com/75052954/107628328-5a4c6080-6c15-11eb-91e3-703883c83aad.JPG)
![output6](https://user-images.githubusercontent.com/75052954/107628332-5ae4f700-6c15-11eb-8207-f42e7e6da5ae.JPG)
![output7](https://user-images.githubusercontent.com/75052954/107628337-5b7d8d80-6c15-11eb-94e3-e2ff880c90f9.JPG)
![output8](https://user-images.githubusercontent.com/75052954/107628340-5caeba80-6c15-11eb-8cc3-9e3a814811d4.JPG)

b)import cv2
import numpy as np
import matplotlib.pyplot as plt
image =cv2.imread('flower1.jpg')
img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(img)
plt.show()
cv2.waitKey(0)
ret, thresh1=cv2.threshold(img,120,255,cv2.THRESH_BINARY)
ret, thresh2=cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
ret, thresh3=cv2.threshold(img,120,255,cv2.THRESH_TRUNC)
ret, thresh4=cv2.threshold(img,120,255,cv2.THRESH_TOZERO)
ret, thresh5=cv2.threshold(img,120,255,cv2.THRESH_TOZERO_INV)
plt.imshow(thresh1)
plt.show()
plt.imshow(thresh2)
plt.show()
plt.imshow(thresh3)
plt.show()
plt.imshow(thresh4)
plt.show()
plt.imshow(thresh5)
plt.show()


