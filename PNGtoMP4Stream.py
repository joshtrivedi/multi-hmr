from moviepy.editor import ImageSequenceClip
import os

def compile_video(imgdir, output_video):
    images = sorted([img for img in os.listdir(imgdir) if img.endswith(".png")])
    clip = ImageSequenceClip([os.path.join(imgdir, img) for img in images], fps=1)
    clip.write_videofile(output_video, codec='libx264')

imgdir = './demo_out_6'
output_video = './output_video_5.mp4'
compile_video(imgdir, output_video)
