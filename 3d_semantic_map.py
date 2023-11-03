import numpy as np
import open3d as o3d
import argparse
import os
import copy
import time
import math
from sklearn.neighbors import NearestNeighbors
import cv2

def depth_image_to_point_cloud(rgb, depth):

    # Camera instrinsics
    principal_point = [256, 256]
    focal_length = np.tan(np.deg2rad(90/2)) * 256

    # Read the rgb images and the depth images
    image = np.asarray(o3d.io.read_image(rgb))
    depth = np.asarray(o3d.io.read_image(depth))
    rgb_colors = (image / 255).reshape(-1, 3)
    pixel_coords = np.indices((512, 512)).reshape(2, -1)
    d = depth.reshape(-1) / 1000 # real depth unit is m
    f = focal_length

    x = (principal_point[0] - pixel_coords[1]) * d / f
    y = (principal_point[1] - pixel_coords[0]) * d / f
    z = d

    xyz_points = np.vstack([x, y, z]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    #calculate camera point
    camera_pcd = o3d.geometry.PointCloud()
    camera_pcd.points.append([0,0,0])
    camera_pcd.colors.append([1,0,0]) # set the color to red

    return pcd, camera_pcd

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

    # o3d.visualization.draw_geometries([downsampled_pcd])
    return downsampled_pcd

def preprocess_point_cloud(pcd, voxel_size):

    # pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down = custom_voxel_down(pcd,voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh          

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])  

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

'''
Start Implementing tht local_icp_algorithm
'''

def best_fit_transform(source_points, target_points):

    # Calculates the least-squares best-fit transform between corresponding 3D points source -> target
    assert np.shape(source_points) == np.shape(target_points)

    # number of dimensions
    m = np.shape(source_points)[1]

    # translate points to their centroids
    centroid_A = np.mean(source_points, axis=0)
    centroid_B = np.mean(target_points, axis=0)
    AA = source_points - centroid_A
    BB = target_points - centroid_B

    # rotation matrix
    W = np.dot(BB.T, AA)
    U, S, VT = np.linalg.svd(W)
    rotation = np.dot(U, VT)

    # special reflection case
    if np.linalg.det(rotation) < 0:
        VT[m-1,:] *= -1
        rotation = np.dot(U, VT)

    # translation
    translation = centroid_B.T - np.dot(rotation, centroid_A.T)

    # homogeneous transformation
    transformation = np.identity(m+1)
    transformation[:m, :m],transformation[:m, m] = rotation,translation

    return transformation, rotation, translation

def nearest_neighbor(source_points, target_points):

    # Find the nearest (Euclidean) neighbor in target points for each point in source points
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(target_points)
    
    # Euclidean distance (errors) of the nearest neighbor and the nearest neighbor
    distances, indices = neigh.kneighbors(source_points, return_distance=True)
    valid = distances < np.median(distances)*0.8
    return distances[valid].ravel(), indices[valid].ravel(),valid.ravel()

def implemented_local_icp_algorithm(source_down, target_down, trans_init=None,  max_iterations=100000 ,tolerance=0.000005): 
    # the user may tuns the parameter

    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    m = np.shape(source_points)[1]

    # make points homogeneous, copy them so as to maintain the originals
    src = np.ones((m+1,source_points.shape[0]))
    dst = np.ones((m+1,target_points.shape[0]))
    src[:m,:] = np.copy(source_points.T)
    dst[:m,:] = np.copy(target_points.T)

    # apply the initial pose estimation
    if trans_init is not None:
        src = np.dot(trans_init, src)
    prev_error = 0

    # main part
    for i in range(max_iterations):
        # find the nearest neighbours between the current source and destination points
        distances, indices, valid = nearest_neighbor(src[0:m,:].T, dst[0:m,:].T)

        # compute the transformation between the current source and nearest destination points
        transformation,_,_ = best_fit_transform(src[0:m,valid].T, dst[0:m,indices].T)
        
        # update the current source
        src = np.dot(transformation, src)

        # check error
        mean_error = np.sum(distances) / distances.size
        if abs(prev_error-mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculcate final tranformation
    transformation,_,_ = best_fit_transform(source_points, src[0:m,:].T)

    return transformation 

def read_pose(args):

    # setting the file path
    file_path = args.data_root + "/GT_pose.npy"
    poses = np.load(file_path)
    print(len(poses)) 

    # transforming to real points with unit:m
    if args.floor == 1 :
        # mm to m / rw => 10 / 0.25
        x = -poses[:, 0]/ 40 
        y = poses[:, 1]/ 40
        z = -poses[:, 2]/ 40
    elif args.floor == 2:
        # 10 / rw => 10 / 0.25
        x = -poses[:, 0] / 40 -0.00582
        y = ( poses[:, 1] / 40 ) - 0.07313
        z = -poses[:, 2]/ 40 -0.03

    xyz_points = np.vstack([x,y,z]).T

    gt_pose_pcd = o3d.geometry.PointCloud()
    gt_pose_pcd.points = o3d.utility.Vector3dVector(xyz_points)
    gt_pose_pcd.paint_uniform_color([0,0,0]) #set color to black

    gt_lines = []
    for i in range(len(xyz_points) - 1):
        gt_lines.append([i, i + 1])

    gt_line_set = o3d.geometry.LineSet()
    gt_line_set.points = o3d.utility.Vector3dVector(xyz_points)
    gt_line_set.lines = o3d.utility.Vector2iVector(gt_lines)

    return gt_pose_pcd, gt_line_set

def reconstruct(args):

    # config
    voxel_size = 0.00225
    point_cloud = o3d.geometry.PointCloud()
    estimate_camera_cloud = o3d.geometry.PointCloud()
    data_folder_path = args.data_root
    rgb_images = os.listdir(os.path.join(data_folder_path, "result_model_3/"))
    # rgb_images = os.listdir(os.path.join(data_folder_path, "gt_result/"))
    depth_images = os.listdir(os.path.join(data_folder_path, "depth/"))
    if args.floor == 1:
        print("Start reconstructing the first floor...")
    if args.floor == 2:
        print("Start reconstructing the second floor...")
    reconstruct_start = time.time()
    print("Numbers of images is %d" % len(rgb_images))

    # temps
    pcd = []
    camera_pcd = []
    pcd_down = []
    pcd_transformed = [] # contain the pcd transformed to the main axis
    fpfh = []

    for i in range(1,len(rgb_images)):
    # for i in range(1,10):
        if i == 1:
            # target point cloud
            rgb_principal = data_folder_path + "/result_model_3/%d.png" % i
            # rgb_principal = data_folder_path + "/gt_result/%d.png" % i
            depth_principal = data_folder_path + "/depth/%d.png" % i
            print("Principal picture is set as picture %d." % i)
            pcd.append(depth_image_to_point_cloud(rgb_principal, depth_principal)[0]) # pcd[i-1]
            camera_pcd.append(depth_image_to_point_cloud(rgb_principal, depth_principal)[1])
            pcd_down.append(preprocess_point_cloud(pcd[i-1],voxel_size)[0]) # pcd_down[i-1]
            fpfh.append(preprocess_point_cloud(pcd[i-1],voxel_size)[1]) # fpfh[i-1]
            pcd_transformed.append(pcd_down[i-1])
        else:
            # get source point cloud    
            rgb= data_folder_path + "/result_model_3/%d.png" % i
            # rgb= data_folder_path + "/gt_result/%d.png" % i
            depth = data_folder_path + "/depth/%d.png" % i
            print("------------------------------------")
            print("dealing with picture %d..." % i)
            pcd.append(depth_image_to_point_cloud(rgb, depth)[0]) # pcd[i-1]
            camera_pcd.append(depth_image_to_point_cloud(rgb, depth)[1])
            pcd_down.append(preprocess_point_cloud(pcd[i-1],voxel_size)[0]) # pcd_down[i-1]
            fpfh.append(preprocess_point_cloud(pcd[i-1],voxel_size)[1]) # target_fpfh[i-1]

            # Global registeration
            global_start = time.time()
            result_ransac = execute_fast_global_registration(pcd_down[i-1], pcd_transformed[i-2], 
                                                             fpfh[i-1], fpfh[i-2], voxel_size)
            print("Global registeration took %.3f sec." % (time.time() - global_start))
        
            # ICP
            icp_start = time.time()
            if args.version == 'open3d':
                print("Using open3d's icp...")
                result_icp = local_icp_algorithm(pcd_down[i-1], pcd_transformed[i-2], 
                                                 result_ransac.transformation, voxel_size)
                transformation = result_icp.transformation # transformation of i to i-1
            elif args.version == 'my_icp':
                print("Using the implemented icp...")
                result_icp = implemented_local_icp_algorithm(pcd_down[i-1], pcd_transformed[i-2], 
                                                 result_ransac.transformation)
                transformation = result_icp # transformation of i to i-1
                # draw_registration_result(pcd_down[i-1],pcd_transformed[i-2],transformation)
            print("ICP took %.3f sec.\n" % (time.time() - icp_start))

            # transformation to 1st camera axis
            # transformation = result_icp.transformation # transformation of i to i-1
            pcd_transformed.append(pcd_down[i-1].transform(transformation))
            camera_pcd[i-1] = camera_pcd[i-1].transform(transformation)

    for pcd in pcd_transformed:
        point_cloud += pcd

    for pcd in camera_pcd:
        estimate_camera_cloud += pcd

    estimate_lines = []
    for i in range(len(estimate_camera_cloud.points) - 1):
        estimate_lines.append([i, i + 1])
    estimate_line_set = o3d.geometry.LineSet()
    estimate_line_set.points = o3d.utility.Vector3dVector(estimate_camera_cloud.points)
    estimate_line_set.lines = o3d.utility.Vector2iVector(estimate_lines)
    estimate_line_set.paint_uniform_color([1, 0, 0])

    # filter the ceiling
    xyz_points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    if args.floor == 1:
        threshold_y = 0.0135
    elif args.floor == 2:
        if args.version == 'open3d':
            threshold_y = 0.0115
        elif args.version == 'my_icp':
            threshold_y = 0.009
    filtered_xyz_points = xyz_points[xyz_points[:, 1] <= threshold_y]
    filtered_colors = colors[xyz_points[:, 1] <= threshold_y]
    point_cloud.points = o3d.utility.Vector3dVector(filtered_xyz_points)
    point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    gt_pose_cloud, gt_line_set = read_pose(args)
    print("------------------------------------")
    print("3D reconstruction took %.3f sec." % (time.time() - reconstruct_start))
    return point_cloud, gt_pose_cloud, gt_line_set, estimate_camera_cloud, estimate_line_set
    # return point_cloud, estimate_camera_cloud, estimate_line_set

def calculate_mean_l2_distance(gt_pos_pcd, estimate_camera_cloud):
    sum = 0
    for i in range(len(estimate_camera_cloud.points)):
        x = gt_pos_pcd.points[i][0]-estimate_camera_cloud.points[i][0]
        y = gt_pos_pcd.points[i][1]-estimate_camera_cloud.points[i][1]
        z = gt_pos_pcd.points[i][2]-estimate_camera_cloud.points[i][2]
        sum += math.sqrt(x**2 + y**2 + z**2)
    return sum/len(estimate_camera_cloud.points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='my_icp', help='open3d')
    parser.add_argument('--data_root', type=str, default='data_collection/first_floor/')
    args = parser.parse_args()

    if args.floor == 1:
        args.data_root = "data_collection/first_floor/"
    elif args.floor == 2:
        args.data_root = "data_collection/second_floor/"
    
    # Output result point cloud and estimated camera pose
    result_pcd, gt_pos_pcd, line_set, estimate_camera_cloud, estimate_line_set= reconstruct(args)

    # Calculate and print L2 distance
    print("Mean L2 distance:", calculate_mean_l2_distance(gt_pos_pcd, estimate_camera_cloud))

    # Visualize result
    # o3d.visualization.draw_geometries([result_pcd,gt_pos_pcd,line_set,estimate_camera_cloud, estimate_line_set])
    o3d.visualization.draw_geometries([result_pcd])
    o3d.io.write_point_cloud('first_floor_model_3_2.pcd',result_pcd)
    # result = custom_voxel_down(result_pcd, 0.0023)
    # o3d.io.write_point_cloud('first_floor_model_3_2.pcd',result)
    print("3D reconstruction finished.")