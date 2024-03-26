# write code to take input video, apply ffmpeg compression and output a video
import ffmpeg
(
    ffmpeg
    .input('lite/gBR_sBM_c01_d04_mBR0_ch02.mp4')
    .hflip()
    .output('output.mp4')
    .run()
)