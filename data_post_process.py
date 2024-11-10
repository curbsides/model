import csv
import os
import random
import shutil

import cv2
import numpy as np

IMAGE_DIR = 'tmp_datasets/bdd100k/sf_images'
LABEL_FILE = 'tmp_datasets/bdd100k/sf_labels.csv'
NEW_IMAGE_DIR = 'tmp_datasets/bdd100k/sf_images_augmented'
NEW_LABEL_FILE = 'tmp_datasets/bdd100k/sf_labels_augmented.csv'

# Create necessary directories
os.makedirs(NEW_IMAGE_DIR, exist_ok=True)

def count_labels():
    true_count = 0
    false_count = 0
    with open(NEW_LABEL_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if there is one
        for line in reader:
            if line[1] == '1':
                true_count += 1
            elif line[1] == '0':
                false_count += 1
    return true_count, false_count, true_count + false_count

def adjust_brightness(image, brightness=0):
    # Ensure brightness does not go out of bounds
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

def augment_image(image_name, label):
    image = cv2.imread(os.path.join(IMAGE_DIR, image_name))
    if image is None:
        return  # Skip if image could not be loaded
    if random.random() < 0.7:
        image = cv2.flip(image, 1)
    if random.random() < 0.5:
        # Rotate randomly between -10 and 10 degrees
        angle = random.randint(-10, 10)
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        # Change brightness randomly between -50 and 50
        brightness = random.randint(-50, 50)
        image = adjust_brightness(image, brightness)
    new_image_name = f'{image_name.split(".")[0]}_augmented.jpg'
    new_image_path = os.path.join(NEW_IMAGE_DIR, new_image_name)
    cv2.imwrite(new_image_path, image)
    with open(NEW_LABEL_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([new_image_name, label])

def augment_loop():
    # First copy the existing labeled data
    shutil.copyfile(LABEL_FILE, NEW_LABEL_FILE)
    for image_file in os.listdir(IMAGE_DIR):
        shutil.copy(os.path.join(IMAGE_DIR, image_file), NEW_IMAGE_DIR)
    
    labeled_dict = {}
    with open(LABEL_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header if necessary
        for line in reader:
            label = line[1]
            if label not in labeled_dict:
                labeled_dict[label] = [line[0]]
            else:
                labeled_dict[label].append(line[0])
    
    while True:
        true_count, false_count, labeled = count_labels()
        if true_count < 1000:
            label = '1'
            image_name = random.choice(labeled_dict[label])
            augment_image(image_name, label)
        if false_count < 1000:
            label = '0'
            image_name = random.choice(labeled_dict[label])
            augment_image(image_name, label)
        if true_count >= 1000 and false_count >= 1000:
            break

augment_loop()
