import os
import sys
import mujoco
import argparse
from pathlib import Path

def convert_urdf_to_mjcf(input_folder):
    # 1. Setup paths
    input_path = Path(input_folder).resolve()
    output_path = input_path / "xml"
    
    if not input_path.is_dir():
        print(f"Error: {input_folder} is not a valid directory.")
        return

    # Create the /xml directory if it doesn't exist
    output_path.mkdir(exist_ok=True)

    # 2. Iterate through URDF files
    urdf_files = list(input_path.glob("*.urdf"))
    
    if not urdf_files:
        print(f"No .urdf files found in {input_path}")
        return

    print(f"Found {len(urdf_files)} URDF files. Starting conversion...")

    for urdf_file in urdf_files:
        # Define output filename
        xml_filename = urdf_file.stem + ".xml"
        save_to = output_path / xml_filename

        try:
            # Load the URDF. 
            # Note: MuJoCo needs to be able to find meshes referenced in the URDF.
            # It usually looks relative to the URDF file location.
            model = mujoco.MjModel.from_xml_path(str(urdf_file))
            
            # Export the compiled model to MJCF XML
            mujoco.mj_saveLastXML(str(save_to), model)
            
            print(f" [SUCCESS] Converted: {urdf_file.name} -> xml/{xml_filename}")
            
        except Exception as e:
            print(f" [FAILED]  Could not convert {urdf_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all URDFs in a folder to MuJoCo XMLs.")
    parser.add_argument("folder", type=str, help="Path to the folder containing URDF files")
    
    args = parser.parse_args()
    convert_urdf_to_mjcf(args.folder)