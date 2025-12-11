import cv2


if __name__ == "__main__":
    img = cv2.imread("data/train/ID4464212.jpg")
    left_img = img[:, :1000, :]
    right_img = img[:, 1000: :]
    resized_left_img = cv2.resize(left_img, (768, 768), interpolation=cv2.INTER_AREA)
    resized_right_img = cv2.resize(right_img, (768, 768), interpolation=cv2.INTER_AREA)
    cv2.imwrite("test_left.png", resized_left_img)
    cv2.imwrite("test_right.png", resized_right_img)