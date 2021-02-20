import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread("original_image.png")
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

R, G, B = fix_img[:,:,0], fix_img[:,:,1],fix_img[:,:,2]

# 1.Simple averaging
gray_1 = (R + G + B)/3
plt.imshow(gray_1, cmap='gray')
plt.savefig('image_simple_average.png')

# 2.Weighted average
gray_2_1 = 0.3 * R + 0.59 * G + 0.11 * B
plt.imshow(gray_2_1, cmap='gray')
plt.savefig('image_weighted_average_1.png')

gray_2_2 = 0.2126 * R + 0.7152 * G + 0.0722 * B
plt.imshow(gray_2_2, cmap='gray')
plt.savefig('image_weighted_average_2.png')

gray_2_3 = 0.299 * R + 0.587 * G + 0.114 * B
plt.imshow(gray_2_3, cmap='gray')
plt.savefig('image_weighted_average_3.png')

# 3.Desaturation
gray_des = (np.minimum(np.minimum(R, G), B) + np.maximum(np.maximum(R, G), B)) / 2
plt.imshow(gray_des, cmap='gray')
plt.savefig('image_desaturation.png')

# 4.Descompotion
gray_max = np.maximum(np.maximum(R, G), B)
plt.imshow(gray_max, cmap='gray')
plt.savefig('image_decomp_max.png')
gray_min = np.minimum(np.minimum(R, G), B)
plt.imshow(gray_min, cmap='gray')
plt.savefig('image_decomp_min.png')

# 5.Single colour channel
gray_5_1 = R
plt.imshow(gray_5_1, cmap='gray')
plt.savefig('image_single_1.png')

gray_5_2 = G
plt.imshow(gray_5_2, cmap='gray')
plt.savefig('image_single_2.png')

gray_5_3 = B
plt.imshow(gray_5_3, cmap='gray')
plt.savefig('image_single_3.png')


# 6.Custom number of grey shades
gray_6= 0.3 * R + 0.59 * G + 0.11 * B
gray_shades = gray_6/20
plt.imshow(gray_shades, cmap='gray')
plt.savefig('image_gray_shades.png')


# 7.Custom number of grey shades with error-diffusion dithering
gray_image = (R + G + B)/3
error = 0
result_img = np.zeros((img.shape[0],img.shape[1]))
for i in range(0,len(gray_image)-1):
    for j in range(0,len(gray_image[0])-1):
        pixel = gray_image[i][j] + error
       
        if  pixel < 127:
            error = pixel
            gray_image[i][j] = 0
        else: 
            error = 255 - pixel
            gray_image[i][j] = 255
        if j == 0 and i == 0:
            gray_image[i][j+1] = gray_image[i][j+1] + error*(7/16)
            gray_image[i+1][j-1] = gray_image[i+1][j-1] + error*(3/16)
            gray_image[i+1][j] = gray_image[i+1][j] + error*(5/16)
            gray_image[i+1][j+1] = gray_image[i+1][j+1] + error*(7/16)

        
plt.imshow(gray_image, cmap='gray')
plt.savefig('image_error_diffusion.png')


# gray to rgb  
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
backtorgb = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)
plt.imshow(backtorgb, cmap='gray')
plt.savefig('image_back2rgb.png')


# Consider a color image, given by its red, green, blue components R, G, B. We present some
# methods for converting the color image to grayscale. Implement all these methods.