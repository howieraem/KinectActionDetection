import cv2
import numpy as np
from dataset.skeleton import *
from utils.converter import world_coords_to_image


if __name__ == '__main__':  # Must run with main for multiprocessing
    skeleton_dataset = SkeletonDatasetFlorence('D:/Illumine/Y4/METR4901/dataset/Florence', 0.8)
    labels = skeleton_dataset.get_labels()
    for (data, label) in skeleton_dataset:
        for frame in data:
            img = np.zeros([640, 480, 3], dtype=np.uint8)
            for idx in range(15):
                joint_coords = (int(frame[idx * 3].numpy()),
                                int(frame[idx * 3 + 1].numpy()),
                                int(frame[idx * 3 + 2].numpy()))
                joint_img_coords = world_coords_to_image(joint_coords, 531.15, 640, 480)
                cv2.circle(img, joint_img_coords, 5, (255, 255, 0), thickness=2)
            cv2.namedWindow('Florence')
            cv2.imshow('Florence', img)
            if cv2.waitKey(34) == 27:
                break
