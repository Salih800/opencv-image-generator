import datetime
import os
import glob
import sys
import threading

import cv2
import imutils
import random
import numpy as np
import shutil
import time
from PIL import ImageColor

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (219, 152, 52),
    (0, 255, 255),
    (0, 165, 255),
    (128, 0, 128),
    (63, 133, 205),
    (128, 128, 128),
    (0, 0, 0),
    (255, 255, 255)
]

colorsName = ["red", "green", "blue", "yellow", "orange", "purple", "brown", "gray", "black", "white"]

class_names = ["circle", "cross", "heptagon", "hexagon", "octagon", "pentagon",
               "quartercircle", "rect", "semicircle", "square", "star", "trapezoid", "triangle",
               "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
               "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
               ]


def make_random_uai(i, images_path):
    picture_name = f"{i}_UAI"
    background = cv2.imread(random.choice(glob.glob("background/*.png")))
    background = imutils.resize(background, width=background.shape[0] * 3)

    rotate_angle = random.randint(0, 359)

    shape_class = 0

    shape_path = f"shape/{class_names[shape_class]}.png"
    shape = cv2.imread(shape_path)
    char_path = f"hilal2.png"
    char = cv2.imread(char_path)

    gray_shape = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)
    _, shape_mask = cv2.threshold(gray_shape, 200, 255, cv2.THRESH_BINARY_INV)
    shape = cv2.cvtColor(shape_mask, cv2.COLOR_GRAY2BGR)

    char_shape = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
    _, char_mask = cv2.threshold(char_shape, 200, 255, cv2.THRESH_BINARY_INV)
    char = cv2.cvtColor(char_mask, cv2.COLOR_GRAY2BGR)

    resized_shape = imutils.resize(shape, width=random.randint(int(20 * background.shape[0] / 100),
                                                               int(30 * background.shape[0] / 100)))
    resized_char = imutils.resize(char, width=random.randint(int(4 * resized_shape.shape[0] / 10),
                                                             int(6 * resized_shape.shape[0] / 10)))

    shape_mask = imutils.rotate_bound(resized_shape, angle=rotate_angle)

    r_shape, g_shape, b_shape = ImageColor.getcolor("#ff0000", "RGB")
    shape_color = (b_shape, g_shape, r_shape)

    empty_shape = np.zeros((resized_shape.shape[0], resized_shape.shape[1], 3), np.uint8)
    empty_shape[:] = shape_color
    painted_shape = cv2.bitwise_and(resized_shape, empty_shape)

    r_char, g_char, b_char = ImageColor.getcolor("#ffffff", "RGB")
    char_color = (b_char, g_char, r_char)

    empty_char = np.zeros((resized_char.shape[0], resized_char.shape[1], 3), np.uint8)
    empty_char[:] = char_color
    painted_char = cv2.bitwise_and(resized_char, empty_char)

    shape_width = random.randint(0, max(0, int(background.shape[0] - painted_shape.shape[0])))
    shape_height = random.randint(0, max(0, int(background.shape[1] - painted_shape.shape[1])))

    char_width = int(painted_shape.shape[0] / 2)  # - int(painted_char.shape[0] / 2)
    char_height = int(painted_shape.shape[1] / 2)  # - int(painted_char.shape[1] / 2) - 3

    added_char = add_image(painted_shape, painted_char, resized_char, (char_width, char_height))

    rotated_shape = imutils.rotate_bound(added_char, angle=rotate_angle)

    added_shape = add_image(background, rotated_shape, shape_mask, (shape_width, shape_height))

    normalized = cv2.GaussianBlur(added_shape, (3, 3), cv2.BORDER_DEFAULT)

    cv2.imwrite(f"{images_path}/{picture_name}.jpg", normalized)


def make_random_uap(i, images_path):
    picture_name = f"{i}_UAP"
    background = cv2.imread(random.choice(glob.glob("background/*.png")))
    background = imutils.resize(background, width=background.shape[0] * 3)

    shape_class = 0
    char_classes = (43, 23, 38)

    characters = []
    for char_class in char_classes:
        char_path = f"alphanumeric/{class_names[char_class]}.png"
        char = cv2.imread(char_path)

        gray_char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
        _, char_mask = cv2.threshold(gray_char, 200, 255, cv2.THRESH_BINARY_INV)
        char = cv2.cvtColor(char_mask, cv2.COLOR_GRAY2BGR)

        char = char[:, random.randint(10, 15): random.randint(-15, -10)]
        characters.append(char)

    characters = np.concatenate(characters, axis=1)

    shape_path = f"shape/{class_names[shape_class]}.png"
    shape = cv2.imread(shape_path)

    gray_shape = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)
    _, shape_mask = cv2.threshold(gray_shape, 200, 255, cv2.THRESH_BINARY_INV)
    shape = cv2.cvtColor(shape_mask, cv2.COLOR_GRAY2BGR)

    rotate_angle = random.randint(0, 359)

    resized_shape = imutils.resize(shape, width=random.randint(int(20 * background.shape[0] / 100),
                                                               int(30 * background.shape[0] / 100)))
    resized_characters = imutils.resize(characters, width=random.randint(int(4 * resized_shape.shape[0] / 10),
                                                                         int(6 * resized_shape.shape[0] / 10)))

    shape_mask = imutils.rotate_bound(resized_shape, angle=rotate_angle)

    r_shape, g_shape, b_shape = ImageColor.getcolor("#89cff0", "RGB")
    shape_color = (b_shape, g_shape, r_shape)

    empty_shape = np.zeros((resized_shape.shape[0], resized_shape.shape[1], 3), np.uint8)
    empty_shape[:] = shape_color
    painted_shape = cv2.bitwise_and(resized_shape, empty_shape)

    r_char, g_char, b_char = ImageColor.getcolor("#ffffff", "RGB")
    char_color = (b_char, g_char, r_char)

    empty_char = np.zeros((resized_characters.shape[0], resized_characters.shape[1], 3), np.uint8)
    empty_char[:] = char_color
    painted_characters = cv2.bitwise_and(resized_characters, empty_char)

    shape_width = random.randint(0, max(0, int(background.shape[0] - painted_shape.shape[0])))
    shape_height = random.randint(0, max(0, int(background.shape[1] - painted_shape.shape[1])))

    char_width = int(painted_shape.shape[0]/2)  # - int(painted_characters.shape[0]/2)
    char_height = int(painted_shape.shape[1]/2)  # - int(painted_characters.shape[1]/2)

    added_characters = add_image(painted_shape, painted_characters, resized_characters, (char_width, char_height))

    rotated_shape = imutils.rotate_bound(added_characters, angle=rotate_angle)

    added_shape = add_image(background, rotated_shape, shape_mask, (shape_width, shape_height))

    normalized = cv2.GaussianBlur(added_shape, (3, 3), cv2.BORDER_DEFAULT)

    cv2.imwrite(f"{images_path}/{picture_name}.jpg", normalized)


def add_image(back, front, mask, pos):
    start_width, start_height = int(pos[0] - front.shape[0] / 2), int(pos[1] - front.shape[1] / 2)
    # start_width, start_height = pos[0], pos[1]
    if start_width < 0:
        front = front[-start_width:, :]
        mask = mask[-start_width:, :]
        start_width = 0

    if start_height < 0:
        front = front[:, -start_height:]
        mask = mask[:, -start_height:]
        start_height = 0

    if start_width + front.shape[0] > back.shape[0]:
        diff = back.shape[0] - start_width
        front = front[:diff, :]
        mask = mask[:diff, :]

    if start_height + front.shape[1] > back.shape[1]:
        diff = back.shape[1] - start_height
        front = front[:, :diff]
        mask = mask[:, :diff]

    rows, cols, channels = front.shape
    roi = back[start_width: start_width + rows, start_height: start_height + cols]

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(roi, roi, mask=mask)
    fg = cv2.bitwise_and(front, front, mask=mask_inv)

    dst = cv2.add(bg, fg)
    back[start_width:start_width + rows, start_height: start_height + cols] = dst

    return back


def clear_path(paths):
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)


uap_images_path = "uap_images"
uai_images_path = "uai_images"

clear_path([uap_images_path, uai_images_path])

uap_image_size = 1000
uai_image_size = 1000

picture_size = uap_image_size + uai_image_size

print(f"UAP Image Size: {uap_image_size}\tUAİ Image Size: {uai_image_size}\nTotal Picture Size: {picture_size}\n")
start_time = time.time()

for i in range(picture_size):
    if i < uap_image_size:
        # make_random_uap(i, uap_images_path)
        threading.Thread(target=make_random_uap, daemon=True, args=(i, uap_images_path)).start()

    else:
        # make_random_uai(i, uai_images_path)
        threading.Thread(target=make_random_uai, daemon=True, args=(i, uai_images_path)).start()
    if i > 0:
        ps = (time.time() - start_time) / i
    else:
        ps = 0
    eta = (picture_size - i) * ps
    eta = datetime.timedelta(seconds=eta)
    sys.stdout.write('\r' + f"random image number: {i + 1}\t\tRemaining Time: {eta}")

while True:
    thread_list = []
    for thread in threading.enumerate():
        thread_list.append(thread)
    if len(thread_list) != 1:
        print(len(thread_list))
        time.sleep(1)
        continue
    else:
        break
