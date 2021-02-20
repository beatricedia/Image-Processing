import cv2
import numpy as np
from matplotlib import pyplot as plt

def checkRGB1(px):
    R, G, B = px
    return ( ((R > 95) and (G > 40) and (B > 20))
        and ((max(px)-min(px))>15) and (abs(R - G) > 15) and
        (R > G) and (R > B))

def checkRGB2(px):
    R, G, B = px
    if G!= 0: 
        return ( (R/G > 1.185) and
        ((R*B)/np.square(R+G+B) > 0.107) and
        ((R*G)/np.square(R+G+B) > 0.112)
        )

def checkHSV1(px):
    H, S, V = px
   
    return ( ((V >= 0.4) and (S > 0.2) and (S < 0.6))
        and (((H > 0) and (H < 25)) or ((H > 335) and (H <= 360)))
    )

def checkHSV2(px):
    H, S, V = px
    return (
        ((H >= 0) and (H <= 50)) and
        ((S >= 0.23) and (S <= 0.68)) and
        ((V >= 0.35) and (V <= 1))
    )

def checkHSV3(px):
    H, S, V = px
    return (
        (((H >= 0) and (H <= 50)) and ((H >=340) and (H <= 360))) and
        (H >= 0.2) and (H >= 0.35)
    )

def checkYCbCr1(px):
    Y, Cb, Cr = px

    return ( 
        (Y  > 80) and (Cb > 85) and (Cb < 135) and (Cr > 135) and (Cr < 180)
    )

def checkYCbCr2(px):
    Y, Cb, Cr = px
 
    return (
        (Cr <= 1.5862*Cb + 20) and 
        (Cr >= 0.3448*Cb + 76.2069) and 
        (Cr >= -4.5652*Cb + 234.5652) and
        (Cr <= -1.15*Cb + 301.75) and 
        (Cr <= -2.2857*Cb + 432.85)
    )

def iterate_over_list(img,method):  
    img = img.tolist()
    skinmask = [[(1 if (method(px)) else 0) for px in row] for row in img]
    return skinmask

def skin_pixels_convert(skinmask):
    mask=np.array(skinmask, dtype = "uint8")
    skin = cv2.bitwise_and(img, img, mask = mask)
    grayImage = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage


img = cv2.imread("messi-body.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#1
skinmask = iterate_over_list(img,checkRGB1)
blackAndWhiteImage = skin_pixels_convert(skinmask)

#2
skinmask2 = iterate_over_list(img,checkRGB2)
blackAndWhiteImage2 = skin_pixels_convert(skinmask2)

# 3
imageHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
skinmask = iterate_over_list(imageHSV,checkHSV1)
hsv_img1 = skin_pixels_convert(skinmask)

# 4
imageHSV2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
skinmask2 = iterate_over_list(imageHSV2,checkHSV2)
hsv_img2 = skin_pixels_convert(skinmask2)

# 5
imageHSV3 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
skinmask3 = iterate_over_list(imageHSV3,checkHSV3)
hsv_img3 = skin_pixels_convert(skinmask3)

# 6
imageYCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
skinmask = iterate_over_list(imageYCrCb,checkYCbCr1)
ycc_img1 = skin_pixels_convert(skinmask)

# 7
imageYCrCb2 = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
skinmask2 = iterate_over_list(imageYCrCb2,checkYCbCr2)
ycc_img2 = skin_pixels_convert(skinmask2)


# face detection 
img = cv2.imread("messi-body.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
skinmask = iterate_over_list(img,checkRGB1)
mask=np.array(skinmask, dtype = "uint8")
skin = cv2.bitwise_and(img, img, mask = mask)
img = skin
method = eval("cv2.TM_CCOEFF")
template = cv2.imread('messi.JPG',0)

w, h = template.shape[::-1]
res = cv2.matchTemplate(blackAndWhiteImage,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img,top_left, bottom_right, 255, 2)

plt.imshow(blackAndWhiteImage, cmap='gray')
plt.savefig('lab1_rgb1')
plt.imshow(blackAndWhiteImage2, cmap='gray')
plt.savefig('lab1_rgb2')
plt.imshow(hsv_img1, cmap='gray')
plt.savefig('lab1_hsv1')
plt.imshow(hsv_img2, cmap='gray')
plt.savefig('lab1_hsv2')
plt.imshow(hsv_img3, cmap='gray')
plt.savefig('lab1_hsv3')
plt.imshow(ycc_img1, cmap='gray')
plt.savefig('lab1_ycc1')
plt.imshow(ycc_img1, cmap='gray')
plt.savefig('lab1_ycc2')


# cv2.imshow("rgb",blackAndWhiteImage)
# cv2.imshow("rgb2",blackAndWhiteImage2)
# cv2.imshow("hsv1", hsv_img1)
# cv2.imshow("hsv2", hsv_img2)
# cv2.imshow("hsv3", hsv_img2)
# cv2.imshow("ycc1", ycc_img1)
# cv2.imshow("ycc2", ycc_img2)


# plt.subplot(121),plt.imshow(res,cmap = 'gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle("cv.TM_CCOEFF")
plt.show()

# emoticon
img = np.zeros((500, 500, 3), dtype = "uint8") 
center_coordinates = (120, 100)
radius = 80
color = (46, 242, 255)
thickness = -1
image = cv2.circle(img, center_coordinates, radius, color, thickness)
elipse_coord = (120,120)
elipse_color = (255,255,255)
img = cv2.ellipse(img,elipse_coord,(50,30),0,0,180,elipse_color,-1)
first_eye_coord = (95,70)
second_eye_coord = (145,70)
eye_color = (0,0,0)
img = cv2.circle(img, first_eye_coord, 10, eye_color, thickness)
img = cv2.circle(img, second_eye_coord, 10, eye_color, thickness)

cv2.imshow("Emoticon", img) 

cv2.waitKey(0)
cv2.destroyAllWindows()  


#  Detect the “skin-pixels” in a color image. Create a new binary image, the same size as the input
# color image, in which the skin pixels are white (255) and all non-skin pixels are black (0).
# Implement all the below described methods.

# A color pixel (R,G,B) is classified as “skin” if: ...
# Use skin pixel classification to detect the face in a portrait image (find a minimal square that frames
# the human face).
# Create an emoticon image.
