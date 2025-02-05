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

    # # load MSQA data
    root_dir = ""
    data_dict = load_json(f"{root_dir}/MSQA_scannet_test_v1.json")
    pcd_root = "/mnt/fillipo/scratch/masaccio/existing_datasets/scannet/scan_data/pcd_with_global_alignment"  # Path to the directory containing the point cloud data; e.g., "data/pcd_with_global_alignment/" 
    # Load the data
    for scan_id, data in data_dict.items():
        for qa_pair in data['response']:
            pcd_path = os.path.join(pcd_root, scan_id + ".pth")
            pcd_data = torch.load(pcd_path)
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1

            visualize_point_cloud_with_instances(points, colors, instance_labels, qa_pair['location'], qa_pair['orientation'], qa_pair['situation'])
    
    # load MSNN data
    root_dir = ""
    data_dict = load_json(f"{root_dir}/next_step_navi_v1.json")
    for scan_id, data in data_dict.items():
        for item_id, item in data.items():
            if 'scene' in item:
                continue
            pcd_path = os.path.join(pcd_root, scan_id + ".pth")
            pcd_data = torch.load(pcd_path)
            points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
            colors = colors / 127.5 - 1
            quaternion = np.array(item['orientation'])
            view_vector = get_view_vector(quaternion)

            visualize_point_cloud_with_instances(points, colors, instance_labels, item['location'], view_vector, item['situation'])