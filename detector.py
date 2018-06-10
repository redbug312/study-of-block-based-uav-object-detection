from tqdm import tqdm
import numpy as np
import cv2

import homographier


def detect(frames, last_thresh=None):
    last_thresh = np.ones(frames[2].img.shape[:2], dtype=np.uint8) \
                  if last_thresh is None else last_thresh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    H2 = [np.dot(frames[1].H, frames[2].H),
          frames[2].H,
          np.identity(3),
          np.linalg.inv(frames[3].H),
          np.linalg.inv(np.dot(frames[3].H, frames[4].H))]

    image = np.array(frames[2].img)
    dims = image.shape[:2][::-1]

    img = [cv2.warpPerspective(f.img, H, dims) for f, H in zip(frames, H2)]
    img = [cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in img]
    img = [cv2.medianBlur(i, 5) for i in img]

    D2 = [cv2.absdiff(i, img[2]) for i in img]
    D2 = [cv2.threshold(D, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for D in D2]

    thresh = (D2[0] * D2[4]) + (D2[1] * D2[3])
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    thresh = cv2.erode(thresh, kernel, iterations=2)

    cnts = cv2.findContours(thresh * last_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in cnts:
        if cv2.contourArea(cnt) <= np.sum(kernel):
            continue
        (x, y, w, h) = cv2.boundingRect(cnt)
        if np.all(np.array([x, y, x+w, y+h]) - (0, 0, image.shape[1], image.shape[0])):
            cv2.rectangle(image, (x, y), (x+w, y+h), (165, 66, 66), 2)

    return homographier.PanoFrame(image=image), thresh
