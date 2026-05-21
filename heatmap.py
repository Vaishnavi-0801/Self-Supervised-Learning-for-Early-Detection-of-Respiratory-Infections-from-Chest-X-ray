import cv2
import numpy as np

def generate_heatmap(img_path):

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    output = "static/uploads/heatmap.png"

    cv2.imwrite(output, heatmap)

    return output