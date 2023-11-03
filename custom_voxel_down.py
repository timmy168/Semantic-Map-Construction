import numpy as np
import open3d as o3d
import argparse
import os
import copy
import time
import math
from sklearn.neighbors import NearestNeighbors
import cv2

def custom_voxel_down_1(pcd ,voxel_size):

    xyz_point = np.asarray(pcd.points)
    xyz_color = np.asarray(pcd.colors)

    # bounding box
    box = pcd.get_axis_aligned_bounding_box()
    box.color = (1,0,0)
    center = box.get_center() 
    margin = box.get_extent() 
    x_min = round(center[0] - margin[0]/2, 3)
    x_max = round(center[0] + margin[0]/2, 3)
    y_min = round(center[1] - margin[1]/2, 3)
    y_max = round(center[1] + margin[1]/2, 3)
    z_min = round(center[2] - margin[2]/2, 3)
    z_max = round(center[2] + margin[2]/2, 3)

    # downsample point cloud
    down_sampled_pcd = o3d.geometry.PointCloud()
    point = []
    color = []
    count = 1
    step_size = int(1/voxel_size)

    for i in range(int(x_min * step_size), int(x_max * step_size), int(voxel_size*step_size)):
        for j in range(int(y_min*step_size), int(y_max*step_size), int(voxel_size*step_size)):
            for k in range(int(z_min*step_size), int(z_max*step_size), int(voxel_size*step_size)):
                count = count + 1
                box_color_voxel = xyz_color[np.where((xyz_point[:,0]>=(i/step_size))*(xyz_point[:,0]<=(i/step_size+voxel_size))*(xyz_point[:,1]>=(j/step_size))*(xyz_point[:,1]<=(j/step_size+voxel_size))*(xyz_point[:,2]>=(k/step_size))*(xyz_point[:,2]<=(k/step_size+voxel_size)))]
                
                # find the color that stands for the voxel
                box_color_voxel = np.around(box_color_voxel, 2)
                unique, counts = np.unique(box_color_voxel, axis=0, return_counts=True)
                if len(counts) > 0:
                    major_color = unique[counts.tolist().index(max(counts))]
                    print("major color:", major_color)
                    point.append([(i/step_size)+voxel_size/2, (j/step_size)+(voxel_size)/2, (k/step_size)+(voxel_size)/2])
                    color.append(major_color)

    down_sampled_pcd.points = o3d.utility.Vector3dVector(point)
    down_sampled_pcd.colors = o3d.utility.Vector3dVector(color)
    
    o3d.visualization.draw_geometries([down_sampled_pcd,box])

def custom_voxel_down(pcd, voxel_size):
    xyz_points = np.asarray(pcd.points)
    xyz_colors = np.asarray(pcd.colors)

    # determine each point is belongs to what voxel
    voxel_indices = ((xyz_points / voxel_size) - (xyz_points.min(0) / voxel_size)).astype(int)

    # create voxel point cloud
    voxel_point_cloud = {}
    voxel_colors = {}
    for i in range(len(xyz_points)):
        voxel_index = tuple(voxel_indices[i])
        if voxel_index not in voxel_point_cloud:
            voxel_point_cloud[voxel_index] = []
            voxel_colors[voxel_index] = []
        voxel_point_cloud[voxel_index].append(xyz_points[i])
        voxel_colors[voxel_index].append(tuple(xyz_colors[i]))

    downsampled_points = []
    downsampled_colors = []
    for voxel_index, points in voxel_point_cloud.items():
        colors = voxel_colors[voxel_index]
        major_color = max(set(colors), key=colors.count)
        downsampled_points.append(np.mean(points, axis=0))
        downsampled_colors.append(np.array(major_color))

    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    downsampled_pcd.colors = o3d.utility.Vector3dVector(downsampled_colors)

    o3d.visualization.draw_geometries([downsampled_pcd])



if __name__ == '__main__':
    point_cloud_file = "reconstruct.pcd"
    voxel_size = 0.0025
    pcd = point_cloud = o3d.io.read_point_cloud(point_cloud_file)
    # o3d.visualization.draw_geometries([pcd])
    custom_voxel_down(pcd,voxel_size)

