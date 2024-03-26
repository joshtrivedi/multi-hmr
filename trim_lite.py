import os

folder_path = 'lite'
video_count = 0

# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.mp4'):
        video_count += 1
        if video_count > 3000:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)