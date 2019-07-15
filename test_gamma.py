from load_rgb_cv import load_rgb, convert_rgb_to_cv2
import cv2
import matplotlib.pyplot as plt

image_file = "Chicago_albedo.png"
output_file = "Chicago_albedo_test.png"

image1 = load_rgb(image_file)

plt.imshow(image1)
plt.show()
print("image1 : {}".format(image1[150,150,1]))

cv2.imwrite(output_file, convert_rgb_to_cv2(image1))

image2 = load_rgb(output_file)

plt.imshow(image2)
plt.show()
print("image2 : {}".format(image2[150,150,1]))

