import cv2
import numpy as np

def detect_abnormality(img_path):

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize
    gray = gray / 255.0

    # detect strong intensity regions
    abnormal = cv2.GaussianBlur(gray, (21,21),0)

    heatmap = cv2.applyColorMap(
        np.uint8(255*abnormal),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    path = "static/uploads/abnormality.png"
    cv2.imwrite(path,overlay)

    return path