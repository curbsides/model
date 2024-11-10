import csv
import os

IMAGE_DIR = 'tmp_datasets/bdd100k/sf_images'
LABEL_FILE = 'tmp_datasets/bdd100k/sf_labels.csv'

def count_labels():
    true_count = 0
    false_count = 0
    with open(LABEL_FILE, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            if line[1] == '1':
                true_count += 1
            elif line[1] == '0':
                false_count += 1
    return true_count, false_count, true_count + false_count

def count_total_images():
    return len(os.listdir(IMAGE_DIR))

true_count, false_count, labeled = count_labels()
total_count = count_total_images()
print(f'True: {true_count / labeled:.2%}')
print(f'False: {false_count / labeled:.2%}')
print(f'Total: {labeled}')
print(f'Labeled: {labeled}/{total_count}')
