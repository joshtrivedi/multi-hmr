import os
import shutil
import subprocess

def move_glb_files(source_dir, destination_dir):
    # Get a list of all files in the source directory
    files = os.listdir(source_dir)

    # Iterate over the files and move the .glb files to the destination directory
    for file in files:
        if file.endswith(".glb" or ".png"):
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(destination_dir, file)
            shutil.move(source_path, destination_path)

def compress_image(input_path, output_path):
    # Compress Image
    subprocess.run(["ffmpeg", "-i", input_path, "-vf", "scale=640:480", "-q:v", "2", output_path])

# Specify the source and destination directories
source_directory = "video_out"
destination_directory = "tmp_data"

input_path = "video_out/000001.png" 
output_path = "compressed/000001_compressed.png"

# Call the function to move the .glb files
# move_glb_files(source_directory, destination_directory)

compress_image(input_path, output_path)