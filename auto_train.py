import cv2
import numpy as np
import os
import shutil
from PIL import Image

def create_dataset():
    img_path = "image.png"
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image.png")
        return

    # Create directories
    if os.path.exists("my_dataset"):
        shutil.rmtree("my_dataset")
    os.makedirs("my_dataset/free", exist_ok=True)
    os.makedirs("my_dataset/occupied", exist_ok=True)

    print("Extracting occupied spots...")
    # The white car in the middle is roughly here
    car = img[155:245, 255:310]
    # Augment the car crop to make enough training data
    for i in range(20):
        # Add random noise and slight brightness shifts
        shift = np.random.randint(-5, 5, 2)
        y1 = max(0, 155 + shift[0])
        y2 = min(img.shape[0], 245 + shift[0])
        x1 = max(0, 255 + shift[1])
        x2 = min(img.shape[1], 310 + shift[1])
        c = img[y1:y2, x1:x2].copy()
        
        # adjust brightness
        c = cv2.convertScaleAbs(c, alpha=1.0, beta=np.random.randint(-20, 20))
        cv2.imwrite(f"my_dataset/occupied/car_{i}.jpg", c)
        
    # the black car top left
    car2 = img[30:100, 30:110]
    for i in range(20):
        c = car2.copy()
        c = cv2.convertScaleAbs(c, alpha=1.0, beta=np.random.randint(-20, 20))
        cv2.imwrite(f"my_dataset/occupied/car2_{i}.jpg", c)

    print("Extracting free spots...")
    # List of known free areas (approximate bounding boxes for empty spots)
    free_boxes = [
        (150, 100, 240, 160),
        (150, 180, 240, 240),
        (150, 350, 240, 410),
        (150, 450, 240, 510),
        (30, 200, 100, 260),
        (30, 300, 100, 360),
        (30, 400, 100, 460),
        (30, 500, 100, 560),
        (260, 100, 350, 160),
        (260, 200, 350, 260),
        (260, 400, 350, 460)
    ]

    count = 0
    for y1, x1, y2, x2 in free_boxes:
        for i in range(5): # Augment each free spot
            shift = np.random.randint(-10, 10, 2)
            yy1, yy2 = y1+shift[0], y2+shift[0]
            xx1, xx2 = x1+shift[1], x2+shift[1]
            if yy1>=0 and yy2<img.shape[0] and xx1>=0 and xx2<img.shape[1]:
                c = img[yy1:yy2, xx1:xx2].copy()
                c = cv2.convertScaleAbs(c, alpha=1.0, beta=np.random.randint(-20, 20))
                cv2.imwrite(f"my_dataset/free/free_{count}.jpg", c)
                count += 1
                
    print("Dataset created successfully!")

if __name__ == "__main__":
    create_dataset()
    import os
    os.system('python "parking_ml (1).py" --data my_dataset --epochs 25')
