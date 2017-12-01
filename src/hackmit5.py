import base64
import json
import scipy.misc
import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

with open('captchas.json') as data_file:
    data1 = json.load(data_file)

images = data1["images"]

def get_image(index):
    image_binary = base64.decodebytes(images[index]["jpg_base64"].encode('utf-8'))
    name = "captchas/" + images[index]["name"] + ".jpg"
    with open(name,'wb') as f:
        f.write(image_binary)
    return name

#print(len(images))

def get_data():
    data = []
    for i in range(1000):
        name = get_image(i)
        im_array = scipy.misc.imread(name)
        data += [im_array]
        print(i)
    return np.array(data)

# data2 = get_data()
# np.save("data", data2)
#
# print(data2.shape)

def equal(val1, val2, threshold):
    return abs(val1 - val2) < threshold

def compare(im1, im2, x, y, thres):
    return (equal(int(im1[x][y][0]), int(im2[x][y][0]), thres)
        and equal(int(im1[x][y][1]), int(im2[x][y][1]), thres)
        and equal(int(im1[x][y][2]), int(im2[x][y][2]), thres))


data = np.load("data.npy")
print(data.dtype)

def is_white(pixel):
    return all(pixel[i] > 185 for i in range(3))

# def clean(index):
#     current_image = data[index]
#     sample_size = 60
#     thres = 40
#     sample_indices = np.random.choice(range(1000), size=sample_size, replace=False)
#     sample = [data[i] for i in sample_indices]
#
#     for i in range(50):
#         for j in range(100):
#             if sum([int(compare(im, current_image, i, j, thres)) for im in sample])/sample_size > 0.7\
#                     or not is_white(current_image[i][j]):
#                 data[index][i][j][0], data[index][i][j][1], data[index][i][j][2] = 0, 0, 0
#             else:
#                 data[index][i][j][0], data[index][i][j][1], data[index][i][j][2] = 255, 255, 255
#
#     return data[index]

def clean(index):
    thres = 20
    if index > 0:
        current_image = data[index]
        previous_image = data[index-1]
    else:
        current_image = data[index]
        previous_image = data[index + 1]

    for i in range(50):
        for j in range(100):
            if compare(current_image, previous_image, i, j, thres):
                data[index][i][j][0], data[index][i][j][1], data[index][i][j][2] = 0, 0, 0
            elif is_white(current_image[i][j]):
                data[index][i][j][0], data[index][i][j][1], data[index][i][j][2] = 255, 255, 255
            else:
                data[index][i][j][0], data[index][i][j][1], data[index][i][j][2] = 0, 0, 0

    return data[index]

cleaned = clean(1)
scipy.misc.imsave("check.jpg", cleaned)

cleaned = Image.open("thing.jpg")

#cleaned = cleaned.filter(ImageFilter.MedianFilter())
#enhancer = ImageEnhance.Contrast(cleaned)
#cleaned = enhancer.enhance(2)
#cleaned = cleaned.convert('1')
#cleaned.save("thing.jpg")
print(pytesseract.image_to_string(Image.open('thing.jpg')))
