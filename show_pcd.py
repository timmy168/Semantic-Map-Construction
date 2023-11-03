import numpy as np
import open3d as o3d
import argparse
import os
import copy
import time
import math
from sklearn.neighbors import NearestNeighbors
import cv2

if __name__ == '__main__':
    point_cloud_file = "second_floor_model_3.pcd"
    voxel_size = 0.0025
    pcd = point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    o3d.visualization.draw_geometries([pcd])