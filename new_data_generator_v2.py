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

angles = [0, 45, 90, 135, 180, 225, 270, 315]
rotations = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

colorsName = ["red", "green", "blue", "yellow", "orange", "purple", "brown", "gray", "black", "white"]

class_names = ["circle", "cross", "heptagon", "hexagon", "octagon", "pentagon",
               "quartercircle", "rect", "semicircle", "square", "star", "trapezoid", "triangle",
               "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
               "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
               "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
               ]


def get_random_images(i, images_path, labels_path):
    background = cv2.imread(random.choice(glob.glob("background/*")))
    background = imutils.resize(background, height=960)
    background_random_width = random.randint(0, background.shape[1] - 960)
    background = background[:, background_random_width:background_random_width + 960]

    shape_class = random.randint(0, len(class_names) - 37)
    char_class = random.randint(13, len(class_names) - 1)

    shape_path = f"shape/{class_names[shape_class]}.png"
    char_path = f"alphanumeric/{class_names[char_class]}.png"

    shape_name = class_names[shape_class]
    char_name = class_names[char_class]

    shape = cv2.imread(shape_path)
    char = cv2.imread(char_path)

    gray_shape = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)
    _, shape_mask = cv2.threshold(gray_shape, 200, 255, cv2.THRESH_BINARY_INV)
    shape = cv2.cvtColor(shape_mask, cv2.COLOR_GRAY2BGR)

    gray_char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
    _, char_mask = cv2.threshold(gray_char, 200, 255, cv2.THRESH_BINARY_INV)
    char = cv2.cvtColor(char_mask, cv2.COLOR_GRAY2BGR)

    shape_angle = random.randint(0, len(angles) - 1)
    # shape_angle = angles[choose_angle]
    # shape_rotation = rotations[choose_angle]
    rotated_shape = imutils.rotate_bound(shape, angle=angles[shape_angle])

    char_angle = random.randint(0, len(angles) - 1)
    # char_angle =
    # char_rotation = rotations[choose_angle]
    rotated_char = imutils.rotate_bound(char, angle=angles[char_angle])

    resized_shape = imutils.resize(rotated_shape, width=random.randint(int(1.7 * background.shape[1] / 20),
                                                                       int(2.5 * background.shape[1] / 20)))

    # char_size_max = (resized_shape.shape[0] * resized_shape.shape[1] / 6) / \
    #                 (rotated_char.shape[0] * rotated_char.shape[1])
    # char_size_min = (resized_shape.shape[0] * resized_shape.shape[1] / 8) / \
    #                 (rotated_char.shape[0] * rotated_char.shape[1])
    # rotated_char.shape[0] * rotated_char.shape[1]
    # print("image number: ", i)
    # print(char_size_min, char_size_max)
    # print(rotated_char.shape)

    # random.randint(int(rotated_char.shape[0] * char_size_min),
    #                int(rotated_char.shape[0] * char_size_max))
    resized_char = imutils.resize(rotated_char, width=random.randint(int(7.9 * resized_shape.shape[1] / 24),
                                                                     int(8 * resized_shape.shape[1] / 24)))

    # resized_char = imutils.resize(rotated_char, width=random.randint(int(rotated_char.shape[0]/2),
    #                                                                  int(rotated_char.shape[0])))

    shape_color = random.randint(0, len(colors) - 1)
    # shape_color_name = colorsName[shape_color]
    # shape_color = 9
    if shape_color == 8:
        painted_shape = cv2.bitwise_not(resized_shape)
    else:
        empty_shape = np.zeros((resized_shape.shape[0], resized_shape.shape[1], 3), np.uint8)
        empty_shape[:] = colors[shape_color]
        painted_shape = cv2.bitwise_and(resized_shape, empty_shape)

    char_color = random.randint(0, len(colors) - 1)
    while char_color == shape_color:
        char_color = random.randint(0, len(colors) - 1)
    # char_color_name = colorsName[char_color]
    # char_color = 9
    if char_color == 8:
        painted_char = cv2.bitwise_not(resized_char)
    else:
        empty_char = np.zeros((resized_char.shape[0], resized_char.shape[1], 3), np.uint8)
        empty_char[:] = colors[char_color]
        painted_char = cv2.bitwise_and(resized_char, empty_char)

    # cv2.imshow("painted_char", painted_char)
    # cv2.waitKey(0)

    picture_name = f"{i}_{colorsName[shape_color]}-{shape_name}_{colorsName[char_color]}-{rotations[char_angle]}-{char_name}"

    shape_width = random.randint(0, max(0, int(background.shape[1] - painted_shape.shape[1])))
    shape_height = random.randint(0, max(0, int(background.shape[0] - painted_shape.shape[0])))

    # char_width = min(max(0, random.randint(int(shape_width + 2 * painted_shape.shape[1] / 4),
    #                                        int(shape_width + 3 * painted_shape.shape[1] / 4))),
    #                  background.shape[1] - painted_shape.shape[1])
    #
    # char_height = min(max(0, random.randint(int(shape_height + 2 * painted_shape.shape[0] / 4),
    #                                         int(shape_height + 3 * painted_shape.shape[0] / 4))),
    #                   background.shape[0] - painted_shape.shape[0])

    # char_width = random.randint(int(shape_width + 2 * painted_shape.shape[1] / 10), int(shape_width + 6 * painted_shape.shape[1] / 10))
    # char_height = random.randint(int(shape_height + 2 * painted_shape.shape[0] / 10), int(shape_height + 6 * painted_shape.shape[0] / 10))

    char_width = int(shape_width + painted_shape.shape[0]/2 - painted_char.shape[0]/2)
    char_height = int(shape_height + painted_shape.shape[0]/2 - painted_char.shape[0]/2)

    positions = (shape_width, shape_height, char_width, char_height)
    # char_width = random.randint(shape_width, shape_width + painted_shape.shape[0] - painted_char.shape[0])
    # char_height = random.randint(shape_height, shape_height + painted_shape.shape[1] - painted_char.shape[1])

    # print(shape_width, shape_height)
    # print(char_width, char_height)

    add_images(background, painted_shape, painted_char, resized_shape,
               resized_char, positions, picture_name, shape_class, char_class,
               labels_path, images_path, shape_color, char_color, char_angle)

    # added_shape = add_image(background, painted_shape, resized_shape, (shape_width, shape_height), picture_name,
    #                         shape_class, labels_path, images_path, color_id=shape_color)
    # added_char = add_image(added_shape, painted_char, resized_char, (char_width, char_height), picture_name, char_class,
    #                        labels_path, images_path, color_id=char_color, rotation_id=char_angle)

    # normalized = cv2.GaussianBlur(added_images, (3, 3), cv2.BORDER_DEFAULT)

    # print(f"Shape_path: {shape_path}\tChar_path: {char_path}")
    # print(f"Shape_color: {colorsName[shape_color]}\tChar_color: {colorsName[char_color]}")

    # cv2.imwrite(f"{data_paths[0]}/{images_path}/{picture_name}.jpg", normalized)
    # cv2.imshow("added_char", added_char)
    # cv2.imshow("normalized", normalized)
    # cv2.waitKey(0)


def add_images(back, shape, char, shape_mask, char_mask, positions, name, shape_class,
               char_class, label_path, image_path, shape_color, char_color, rotation):
    # start_width, start_height = int(pos[0] - front.shape[0] / 2), int(pos[1] - front.shape[1] / 2)
    shape_width, shape_height, char_width, char_height = positions
    # for shape
    if shape_width < 0:
        shape = shape[:, -shape_width:]
        shape_mask = shape_mask[:, -shape_width:]
        shape_width = 0

    if shape_height < 0:
        shape = shape[-shape_height:, :]
        shape_mask = shape_mask[-shape_height:, :]
        shape_height = 0

    if shape_width + shape.shape[1] > back.shape[1]:
        diff = back.shape[1] - shape_width
        shape = shape[:, :diff]
        shape_mask = shape_mask[:, :diff]

    if shape_height + shape.shape[0] > back.shape[0]:
        diff = back.shape[0] - shape_height
        shape = shape[:diff, :]
        shape_mask = shape_mask[:diff, :]

    # for char
    if char_width < 0:
        char = char[:, -char_width:]
        char_mask = char_mask[:, -char_width:]
        char_width = 0

    if char_height < 0:
        char = char[-char_height:, :]
        char_mask = char_mask[-char_height:, :]
        char_height = 0

    if char_width + char.shape[1] > back.shape[1]:
        diff = back.shape[1] - char_width
        char = char[:, :diff]
        char_mask = char_mask[:, :diff]

    if char_height + char.shape[0] > back.shape[0]:
        diff = back.shape[0] - char_height
        char = char[:diff, :]
        char_mask = char_mask[:diff, :]

    height, width, channels = shape.shape
    roi = back[shape_height: shape_height + height, shape_width: shape_width + width]
    # print(shape_height,height,shape_width,width)
    # print(shape_mask.shape, roi.shape)
    gray = cv2.cvtColor(shape_mask, cv2.COLOR_BGR2GRAY)
    _, shape_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    shape_mask_inv = cv2.bitwise_not(shape_mask)

    # print(shape.shape, shape_mask.shape, roi.shape, back.shape)
    bg = cv2.bitwise_and(roi, roi, mask=shape_mask)
    fg = cv2.bitwise_and(shape, shape, mask=shape_mask_inv)

    shape_roi = cv2.add(bg, fg)
    back[shape_height: shape_height + height, shape_width: shape_width + width] = shape_roi

    shape_x, shape_y, shape_w, shape_h = (shape_width + shape.shape[1] / 2) / back.shape[1], (
                shape_height + shape.shape[0] / 2) / back.shape[0], \
                                         shape.shape[1] / back.shape[1], shape.shape[0] / back.shape[0]

    shape_label = f"{shape_class} {shape_x} {shape_y} {shape_w} {shape_h}\n"
    shape_color_label = f"{shape_color} 0.5 0.5 1 1"

    height, width, channels = char.shape
    roi = back[char_height: char_height + height, char_width: char_width + width]

    gray = cv2.cvtColor(char_mask, cv2.COLOR_BGR2GRAY)
    _, char_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    char_mask_inv = cv2.bitwise_not(char_mask)

    bg = cv2.bitwise_and(roi, roi, mask=char_mask)
    fg = cv2.bitwise_and(char, char, mask=char_mask_inv)

    char_roi = cv2.add(bg, fg)
    back[char_height: char_height + height, char_width: char_width + width] = char_roi

    height, width, channels = shape.shape
    shape_roi = back[shape_height: shape_height + height, shape_width: shape_width + width]

    char_x, char_y, char_w, char_h = (char_width + char.shape[1] / 2) / back.shape[1], \
                                     (char_height + char.shape[0] / 2) / back.shape[0], char.shape[1] / back.shape[1], char.shape[0] / back.shape[0]

    char_label = f"{char_class} {char_x} {char_y} {char_w} {char_h}\n"
    char_color_label = f"{char_color} 0.5 0.5 1 1"
    char_rotation_label = f"{rotation} 0.5 0.5 1 1"

    with open(f"{data_paths[0]}/{label_path}/{name}.txt", "a+") as label_file:
        label_file.write(shape_label + char_label)
    with open(f"{data_paths[1]}/{label_path}/{name}_{shape_class}.txt", "a+") as label_file:
        label_file.write(shape_color_label)
    with open(f"{data_paths[1]}/{label_path}/{name}_{char_class}.txt", "a+") as label_file:
        label_file.write(char_color_label)
    with open(f"{data_paths[2]}/{label_path}/{name}.txt", "a+") as label_file:
        label_file.write(char_rotation_label)

    normalized_image = cv2.GaussianBlur(back, (3, 3), cv2.BORDER_DEFAULT)
    normalized_shape_roi = cv2.GaussianBlur(shape_roi, (3, 3), cv2.BORDER_DEFAULT)
    normalized_char_roi = cv2.GaussianBlur(char_roi, (3, 3), cv2.BORDER_DEFAULT)

    cv2.imwrite(f"{data_paths[0]}/{image_path}/{name}.jpg", normalized_image)
    cv2.imwrite(f"{data_paths[1]}/{image_path}/{name}_{shape_class}.jpg", normalized_shape_roi)
    cv2.imwrite(f"{data_paths[1]}/{image_path}/{name}_{char_class}.jpg", normalized_char_roi)
    cv2.imwrite(f"{data_paths[2]}/{image_path}/{name}.jpg", normalized_char_roi)

    # if x < 0 or y < 0 or w < 0 or h < 0:
    #     print(label + "\n")
    # if x > 1 or y > 1 or w > 1 or h > 1:
    #     print(label + "\n")

    # cv2.imshow("back", back)
    # cv2.waitKey(0)
    # return back


def clear_path(paths, parents):
    for parent in parents:
        for path in paths:
            if os.path.isdir(parent + "/" + path):
                shutil.rmtree(parent + "/" + path)
            print(f"Making {parent + '/' + path} folder")
            os.makedirs(parent + "/" + path)


data_paths = ["for_objects", "for_colors", "for_rotations"]

train_images_path = "train/images"
train_labels_path = "train/labels"

val_images_path = "val/images"
val_labels_path = "val/labels"

test_images_path = "test/images"
test_labels_path = "test/labels"

clear_path([train_images_path, train_labels_path, val_images_path, val_labels_path, test_images_path, test_labels_path],
           data_paths)

train_size = 8000
val_size = 2000
test_size = 1000

picture_size = train_size + val_size + test_size

print(f"Train Size: {train_size}\tVal Size: {val_size}\tTest Size: {test_size}\nTotal Picture Size: {picture_size}\n")
start_time = time.time()

for i in range(picture_size):
    if i < train_size:
        # get_random_images(i, train_images_path, train_labels_path)
        threading.Thread(target=get_random_images, daemon=True, args=(i, train_images_path, train_labels_path)).start()
    elif train_size <= i < train_size + val_size:
        # get_random_images(i, val_images_path, val_labels_path)
        threading.Thread(target=get_random_images, daemon=True, args=(i, val_images_path, val_labels_path)).start()
    else:
        # get_random_images(i, test_images_path, test_labels_path)
        threading.Thread(target=get_random_images, daemon=True, args=(i, test_images_path, test_labels_path)).start()

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
        print(f"\nCompleted time: {datetime.timedelta(seconds=(time.time() - start_time))}")
        break
