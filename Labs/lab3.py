import cv2 
import pytesseract 
import numpy as np
from skimage.util import random_noise
from pytesseract import Output
import csv
import csv  
import enchant

# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\beatr\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def sp_noise(img):
    #salt&peper
    noise_img = random_noise(img, mode='s&p',amount=0.3)
    sp_noise = np.array(255*noise_img, dtype = 'uint8')
    return sp_noise

def gaussian_noise(img):
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    img_gauss = cv2.add(img,gauss) 
    return img_gauss

def speckle_noise(img):
    gauss = np.random.normal(0,1,img.size)
    gauss = gauss.reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8')
    speckle_noise = img + img * gauss
    return speckle_noise

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)

def rotation(img):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 10, 1.0)
    img_rotate = cv2.warpAffine(img, M, (w, h))
    return img_rotate, M

def shear(img,M):
    shear = 1
    type_border = cv2.BORDER_CONSTANT
    color_border = (255,255,255)
    rows,cols,ch = img.shape
    cos_part = np.abs(M[0, 0]); sin_part = np.abs(M[0, 1])
    new_cols = int((rows * sin_part) + (cols * cos_part))
    new_rows = int((rows * cos_part) + (cols * sin_part))
    #Second: Necessary space for the shear
    new_cols += (shear*new_cols)
    new_rows += (shear*new_rows)

    #Calculate the space to add with border
    up_down = int((new_rows-rows)/2); left_right = int((new_cols-cols)/2)

    final_image = cv2.copyMakeBorder(img, up_down, up_down,left_right,left_right,type_border, value = color_border)
    return final_image

def resize(img):
    scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    res = cv2.resize(img,dim, interpolation = cv2.INTER_CUBIC)
    return res

 

def ocr(img):
    sharpeningKernel = np.array(([0, -1, 0],[-1, 5, -1],[0, -1, 0]), dtype="int")

    # sharpening
    img = cv2.filter2D(img, -1, sharpeningKernel)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
    # morphological operations
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
    erosion = cv2.erode(dilation, rect_kernel, iterations = 1) 
    opening = cv2.morphologyEx( erosion , cv2.MORPH_OPEN,rect_kernel) 
    
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,  
                                                    cv2.CHAIN_APPROX_NONE) 
    
    im2 = img.copy() 
    
    file = open("recognized.txt", "w+") 
    file.write("") 
    file.close() 
    
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cropped = im2[y:y + h, x:x + w] 
        file = open("recognized.txt", "a") 
        text = pytesseract.image_to_string(cropped)  
        file.write(text) 
        file.write("\n") 
        file.close 

def count_words():
    # file = open("recognized.txt", "rt")
    file = open("recognized.txt", "rt")
    data = file.read()
    words = data.split()
    remaining_words = []
    d = enchant.Dict("en_US") 
    for i in range(0,len(words)):
        if i != "" and d.check(words[i]) == True:
            remaining_words.append(words[i])

    return len(remaining_words)


# 0 simple image
img = cv2.imread("article.png") 
ocr(img )
print("simple image : ",count_words())

# 1 noise
sp_img = sp_noise(img)
gauss_img = gaussian_noise(sp_img)
speckle_img = speckle_noise(gauss_img)
ocr(speckle_img )
print("noised image : ",count_words())

noise_removed = remove_noise(speckle_img )
ocr(noise_removed )
print("unnoised image : ",count_words())

# 2 affine transformations
rotate_img, M = rotation(img)
shear_img = shear(rotate_img,M)
ocr(shear_img )
print("affine transformation : ",count_words())

# 3 resize
resize_img = resize(img)
ocr(resize_img  )
print("resized transformation : ",count_words())

# 4 blur
blur_img = cv2.medianBlur(img,5)
ocr(blur_img  )
print("blur transformation : ",count_words())



#  Install an OCR library (Tesseract, for example) and test its limitations:
# 1. Add different amounts of noise and different types of noise to the image to be processed;
# 2. Apply affine transformations: rotations, shear (vertical, horizontal), …;
# 3. Resize the image;
# 4. Blur the image with average filters of different sizes.
# Evaluate the results computing the number of well recognized words.
# Use image enhancement preprocessing techniques: image sharpening, morphological
# operations, thresholding techniques. Does preprocessing help?