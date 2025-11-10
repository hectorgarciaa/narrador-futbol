import cv2
import numpy as np
from sklearn.cluster import KMeans

class ShirtDetector:
    def __init__(self,  n_clusters=2, init='k-means++', n_init=10, random_state=0,
                 lower_grass=np.array([35, 40, 40]), upper_grass=np.array([85, 255, 255])):
        self.lower_grass = lower_grass
        self.upper_grass = upper_grass

        self.km = KMeans(n_clusters=n_clusters, init=init, n_init=n_init,
                         random_state=random_state)

    def is_grass(self, c):
        h, s, v = c
        return  self.lower_grass[0] <= h <= self.upper_grass[0] and \
                self.lower_grass[1] <= s <= self.upper_grass[1] and \
                self.lower_grass[2] <= v <= self.upper_grass[2]

    def getColorKMeans(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3)
        if len(pixels) < 2:
            return np.array([0, 0, 0])
        
        self.km.fit(pixels)
        centers = self.km.cluster_centers_

        if self.is_grass(centers[0]) and not self.is_grass(centers[1]):
            shirt = centers[1]
        elif self.is_grass(centers[1]) and not self.is_grass(centers[0]):
            shirt = centers[0]
        else:
            shirt = centers[np.argmax(centers[:, 1])]
        return shirt


