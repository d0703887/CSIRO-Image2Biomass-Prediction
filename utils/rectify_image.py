import pandas as pd
import cv2
import os

def rectify_rotated_image(image_path, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 3. Find the largest contour (the valid grass area)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)

    # 4. Get the minimum area rotated rectangle
    rect = cv2.minAreaRect(c)
    ((cx, cy), (w, h), angle) = rect

    # 5. Handle rotation angle quirks
    # minAreaRect returns angles in range [-90, 0).
    # We standardize to keep the image "upright".
    if w < h:
        w, h = h, w
        angle += 90

    # 6. Rotate the image back to align with axes
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # 7. Crop the valid center area
    # getRectSubPix extracts the rectangle defined by center, size, and rotation 0 (since we already rotated the img)
    crop = cv2.getRectSubPix(rotated_img, (int(w), int(h)), (cx, cy))

    if save_path:
        cv2.imwrite(save_path, crop)

    return crop

if __name__ == '__main__':
    data_dir = "../data/GrassClover/test/images"
    output_dir = "../data/GrassClover/rectified_test"
    for img in os.listdir(data_dir):
        rectify_rotated_image(os.path.join(data_dir, img), os.path.join(output_dir, img))

