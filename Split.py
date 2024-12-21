import os
import shutil
import re

def organize_fingerprints(source_folder, destination_folder):
    """
    Organizes fingerprint images into folders by person and finger.

    Args:
        source_folder (str): Path to the folder containing fingerprint images.
        destination_folder (str): Path to the folder where organized files will be saved.
    """
    # Regular expression to extract metadata from filenames
    pattern = r"(\d+)__([MF])_([A-Za-z_]+)_([A-Za-z]+)\.BMP"

    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".BMP"):
            match = re.match(pattern, filename)
            if match:
                person_id = match.group(1)
                gender = match.group(2)
                finger = match.group(3)

                # Create a subdirectory path for each person and finger
                subfolder = os.path.join(destination_folder, f"Person_{person_id}", finger)
                os.makedirs(subfolder, exist_ok=True)

                # Move the file to the appropriate subdirectory
                src_path = os.path.join(source_folder, filename)
                dest_path = os.path.join(subfolder, filename)
                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")

if __name__ == "__main__":
    # Replace with your source and destination folders
    source_folder = "Dataset/Real"
    destination_folder = "Dataset/Real"

    organize_fingerprints(source_folder, destination_folder)

