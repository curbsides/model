import os
import cv2
import random
from tkinter import *
from PIL import Image, ImageTk
import csv

# Path definitions
IMAGE_DIR = 'tmp_datasets/bdd100k/sf_images'
LABEL_FILE = 'tmp_datasets/bdd100k/sf_labels.csv'


# Get the next image to label
def get_next_image():
    unlabeled_images = get_unlabeled_images()
    return random.choice(unlabeled_images) if unlabeled_images else None

# Get the list of images that have not been labeled
def get_unlabeled_images():
    labeled_images = set()
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r') as f:
            reader = csv.reader(f)
            labeled_images = {line[0] for line in reader}
    return [image for image in os.listdir(IMAGE_DIR) if image not in labeled_images]

# Count labeled images
def count_labeled_images():
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, 'r') as f:
            return sum(1 for _ in f)
    return 0

# Save the label
def save_label(image_name, label):
    with open(LABEL_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_name, label])

# Create the labeler
def labeler():
    # Initialize Tkinter window
    window = Tk()
    window.title('Labeler')
    window.configure(bg='black')

    image_label = Label(window)
    image_label.pack()

    image_name = None  # Initialize image_name in the outer scope
    img = None         # Initialize img in the outer scope
    labeled_count = count_labeled_images()  # Get initial labeled count

    # Label to display count
    count_label = Label(window, text=f"Labeled images: {labeled_count}", bg='black', fg='white')
    count_label.pack(pady=10)

    # Flash background to indicate feedback
    def flash_background(color):
        original_bg = window['bg']
        window['bg'] = color
        window.after(150, lambda: window.configure(bg=original_bg))

    # Display the next image
    def display_next_image():
        nonlocal image_name, img
        image_name = get_next_image()
        if not image_name:
            Label(window, text='All images have been labeled', bg='black', fg='white').pack()
            return
        
        # Load and display the image
        img_path = os.path.join(IMAGE_DIR, image_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        # Resize the window to match the image dimensions
        img_width, img_height = img.size
        window.geometry(f"{img_width}x{img_height + 100}")  # Extra height for buttons

        img = ImageTk.PhotoImage(img)
        
        image_label.configure(image=img)
        image_label.image = img

    # Handle labeling actions
    def label_true():
        nonlocal labeled_count
        save_label(image_name, '1')
        labeled_count += 1
        count_label.config(text=f"Labeled images: {labeled_count}")
        flash_background('green')
        display_next_image()
        
    def label_false():
        nonlocal labeled_count
        save_label(image_name, '0')
        labeled_count += 1
        count_label.config(text=f"Labeled images: {labeled_count}")
        flash_background('red')
        display_next_image()
        
    # Stop and save action
    def stop_and_save():
        window.destroy()

    # Set up key bindings for 'j' (True) and 'k' (False)
    window.bind('j', lambda event: label_true())
    window.bind('k', lambda event: label_false())

    # Add buttons with mouse click options as backup
    true_button = Button(window, text='True (J)', command=label_true)
    true_button.pack(side=LEFT, padx=20, pady=10)
    false_button = Button(window, text='False (K)', command=label_false)
    false_button.pack(side=RIGHT, padx=20, pady=10)
    stop_button = Button(window, text='Stop and Save', command=stop_and_save)
    stop_button.pack(pady=20)

    # Initialize with the first image
    display_next_image()
    window.mainloop()

labeler()
