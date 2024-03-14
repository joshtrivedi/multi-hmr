from moviepy.editor import VideoFileClip
import os
import os.path as path

def read_file(string):
    return 'a' 

def extract_frames(movie, times, imgdir):
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    
    clip = VideoFileClip(movie)
    for t in times:
        imgpath = os.path.join(imgdir, '{}.png'.format(int(t * clip.fps)))
        clip.save_frame(imgpath, t)


movie = './input_5.mp4'
imgdir = './demo_out_5'
clip = VideoFileClip(movie)
times = [i/clip.fps for i in range(int(clip.fps * clip.duration))]
extract_frames(movie, times, imgdir)
print(path.isfile(movie))