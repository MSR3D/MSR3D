import os
import torch
import json

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

import numpy as np

def quaternion_to_euler(quaternion):
    euler_angles = R.from_quat(quaternion).as_euler('xyz', degrees=False)
    x_angle, y_angle, z_angle = euler_angles
    return [x_angle, y_angle, z_angle]

def get_view_vector(quaternion):
    # translate quaternion to view vector in MSNN
    angle = quaternion_to_euler(quaternion)[-1]   # angle rotate around z axis
    view_vector = [np.cos(angle), np.sin(angle), 0.0]
    return view_vector

def load_json(path):
    with open(path, 'r') as f:
        json_file = json.load(f)
    return json_file

def visualize_point_cloud_with_instances(points, colors, instance_labels, location=None, orientation=None, situation=None):
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector((colors + 1) / 2)  # Rescale to [0, 1]

    # Retain all unique labels, including negative ones
    unique_labels = np.unique(instance_labels)
    max_label = unique_labels.max()
    min_label = unique_labels.min()

    # Generate colors for all labels (positive and negative)
    label_colors = np.random.uniform(0, 1, size=(max_label - min_label + 1, 3))
    instance_colors = label_colors[(instance_labels - min_label).astype(int)]

    # Create a point cloud for instances
    instance_pcd = o3d.geometry.PointCloud()
    instance_pcd.points = o3d.utility.Vector3dVector(points)
    instance_pcd.colors = o3d.utility.Vector3dVector(instance_colors)

    # Create a list of geometries for visualization
    geometries = [pcd]

    # Add an arrow to visualize the QA pair's location and orientation
    if location is not None and orientation is not None:
        arrow = create_arrow(location, orientation, scale=0.5)
        geometries.append(arrow)

    # Visualize
    o3d.visualization.draw_geometries(geometries, window_name=f"{situation}")

def align_situation(pos, ori, scene_center, align_matrix):
    """
    We need to transform the location and orientation to align with pcd
    pos: [x, y, z]; ori: [_x, _y, _z, _w]
    """
    if isinstance(pos, dict):
        pos = [pos['x'], pos['y'], pos['z']]
    pos = np.array(pos)

    if isinstance(ori, dict):
        ori = [ori['_x'], ori['_y'], ori['_z'], ori['_w']]
    ori = np.array(ori)

    pos_new = pos.reshape(1, 3) @ align_matrix.T
    pos_new += scene_center
    pos_new = pos_new.reshape(-1)

    ori = R.from_quat(ori).as_matrix()
    ori_new = align_matrix @ ori
    flip_matrix = R.from_euler('z', 180, degrees=True).as_matrix()
    ori_new = flip_matrix @ ori_new
    ori_new = R.from_matrix(ori_new).as_quat()
    ori_new = ori_new.reshape(-1)
    return pos_new, ori_new

def create_arrow(origin, direction, scale=0.5):
    """
    Creates an Open3D arrow aligned with a given direction.
    
    :param origin: (x, y, z) starting point
    :param direction: (x, y, z) direction vector
    :param scale: Scaling factor for better visibility
    :return: Open3D arrow geometry
    """
    # Normalize the direction vector and scale
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction) * scale

    # Create the arrow geometry
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,  # Thicker shaft
        cone_radius=0.06,  # Bigger arrowhead
        cylinder_height=scale * 0.8,  # Length of the shaft
        cone_height=scale * 0.2  # Height of the arrowhead
    )

    # Compute rotation: Align default Z-axis (0,0,1) to the given direction
    default_dir = np.array([0, 0, 1])  # Default arrow direction in Open3D
    if not np.allclose(direction, default_dir):  # Only rotate if different
        rotation_matrix = R.align_vectors([direction], [default_dir])[0].as_matrix()
        arrow.rotate(rotation_matrix)

    # Translate to the origin point
    arrow.translate(origin)

    # Set the arrow color (red)
    arrow.paint_uniform_color([1, 0, 0])

    return arrow

if __name__ == "__main__":

    # # # load MSQA data
    # root_dir = ""
    # data_dict = load_json(f"{root_dir}/msqa_scannet_test.json")
    # pcd_root = ""  # Path to the directory containing the point cloud data; e.g., "data/pcd_with_global_alignment/" 
    # # Load the data
    # for data_id in range(10):
    #     qa_pair = data_dict[data_id]
    #     pcd_path = os.path.join(pcd_root, scan_id + ".pth")
    #     pcd_data = torch.load(pcd_path)
    #     points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    #     colors = colors / 127.5 - 1

    #     visualize_point_cloud_with_instances(points, colors, instance_labels, qa_pair['location'], qa_pair['orientation'], qa_pair['situation'])
    
    # # load MSNN data
    # root_dir = ""
    # data_dict = load_json(f"{root_dir}/msnn_scannet.json")
    # for scan_id, data in data_dict.items():
    #     for item_id, item in data.items():
    #         if 'scene' in item:
    #             continue
    #         pcd_path = os.path.join(pcd_root, scan_id + ".pth")
    #         pcd_data = torch.load(pcd_path)
    #         points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
    #         colors = colors / 127.5 - 1
    #         quaternion = np.array(item['orientation'])
    #         view_vector = get_view_vector(quaternion)

    #         visualize_point_cloud_with_instances(points, colors, instance_labels, item['location'], view_vector, item['situation'])
    
    # load SQA3D data
    for data_id in range(10):
        pcd_root = ""  # pcd_with_global_alignment
        anno_path = ""  # v1_balanced_sqa_annotations_val_scannetv2.json
        question_path = ""   # v1_balanced_questions_val_scannetv2.json
        align_matrices_path = ""
        align_matrices = torch.load(align_matrices_path)  # axisAlignment.pth
        
        data_dict = load_json(anno_path)
        data_instance = data_dict['annotations'][data_id]
        data_instance_question = load_json(question_path)['questions'][data_id]
        situation = data_instance_question['situation']
        
        scan_id = data_instance['scene_id']
        pcd_path = os.path.join(pcd_root, scan_id + ".pth")
        pcd_data = torch.load(pcd_path)
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        scene_center = (points.max(0) + points.min(0)) / 2
        colors = colors / 127.5 - 1
        align_matrix = align_matrices[scan_id]
        location, quaternion = align_situation(data_instance['position'], data_instance['rotation'], scene_center, align_matrix)
        quaternion = np.array(quaternion)
        view_vector = get_view_vector(quaternion)
        visualize_point_cloud_with_instances(points, colors, instance_labels, location, view_vector, situation)