from tqdm import tqdm
import numpy as np
import cv2

import homographier


def detect(pano_frames, progress_bar=True):
    detected_frames = list()
    last_thresh = np.ones(pano_frames[0].img.shape[:2], dtype=np.uint8)

    adjacent_5frames = zip(pano_frames, pano_frames[1:], pano_frames[2:], pano_frames[3:], pano_frames[4:])
    adjacent_5frames = tqdm(adjacent_5frames, total=len(pano_frames)-4) if progress_bar else adjacent_5frames

    for index, frames in enumerate(adjacent_5frames):
        H2 = [np.dot(frames[1].H, frames[2].H),
              frames[2].H,
              np.identity(3),
              np.linalg.inv(frames[3].H),
              np.linalg.inv(np.dot(frames[3].H, frames[4].H))]

        image = np.array(frames[2].img)
        dims = image.shape[:2][::-1]

        img = [cv2.warpPerspective(f.img, H, dims) for f, H in zip(frames, H2)]
        img = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]
        img = [cv2.medianBlur(i, 5) for i in img]

        D2 = [cv2.absdiff(i, img[2]) for i in img]
        D2 = [cv2.threshold(D, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for D in D2]

        thresh = (D2[0] * D2[4]) + (D2[1] * D2[3])
        thresh = cv2.dilate(thresh, None, iterations=20)
        thresh = cv2.erode(thresh, None, iterations=10)

        cnts = cv2.findContours(last_thresh * thresh * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        for cnt in cnts:
            if cv2.contourArea(cnt) < 50:
                continue
            (x, y, w, h) = cv2.boundingRect(cnt)
            if not np.all(np.array([x, y, x+w, y+h]) - (0, 0, image.shape[1], image.shape[0])):
                continue
            cv2.rectangle(image, (x, y), (x+w, y+h), (66, 66, 165), 2)

        # cv2.imwrite('detect/detect{:05d}.jpg'.format(index + 2), image)
        detected_frames.append(homographier.PanoFrame(image=image))
        last_thresh = thresh

    return detected_frames
