
import argparse
import os
import torch
from torch.serialization import save
import laspy
import numpy as np

from geomapi.utils import geometryutils as gmu
import open3d as o3d

def handle_process(file_name, output_folder):
    
    coords = []
    scene_id = os.path.basename(file_name)

    name, ext = os.path.splitext(scene_id)
    
    if ext not in  [".las", ".laz"]:
        return

    # Read LAS/LAZ
    # populate dict
    las = laspy.read(file_name)
    print(list(las.point_format.dimension_names))

    pcd = gmu.las_to_pcd(las)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    
    coords = np.stack([las.x, las.y, las.z], axis=1)
    normals = np.asarray(pcd.normals)    
    # verticality = np.stack([las.verticality * 0 + 1, las.omnivariance * 0 + 1, las.normal_change_rate * 0 + 1], axis=1)
    verticality = np.nan_to_num(las.verticality)
    max = np.max(verticality)
    verticality = verticality / (max / 2.) - 1.
    save_dict = dict(coord=coords, normal=normals, verticality=verticality, scene_id=scene_id, semantic_gt=las.labels.astype(int))

    print(save_dict)

    torch.save(save_dict, os.path.join(output_folder, f"{name}_feat.pth"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the 3DOM dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where processed files will be saved",
    )
    config = parser.parse_args()

    os.makedirs(config.output_root, exist_ok=True)

    for file_name in os.listdir(config.dataset_root):

        handle_process(os.path.join(config.dataset_root, file_name), config.output_root)