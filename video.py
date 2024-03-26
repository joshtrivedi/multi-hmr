# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

# python3 video.py --video D0-talawa_technique_intro-Scene-002.mp4 --out_folder output --save_mesh 0 --save_image 0 --fov 80 --start_frame 0 --end_frame 10000000 --save_pkl 1 


import os 
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import subprocess
import sys
from argparse import ArgumentParser
import random
import pickle as pkl
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import time
import cv2
import ffmpeg

from utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from model import Model
from pathlib import Path
import warnings
import pickle 

torch.cuda.empty_cache()

np.random.seed(seed=0)
random.seed(0)

def open_image(img_pil, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    # img_pil = Image.open(img_path).convert('RGB')
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))

    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K

def load_model(model_name, device=torch.device('cuda')):
    """ Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths. """
    # Model
    ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, model_name+ '.pt')
    if not os.path.isfile(ckpt_path):
        os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
        print(f"{ckpt_path} not found...")
        print("It should be the first time you run the demo code")
        print("Downloading checkpoint from NAVER LABS Europe website...")
        
        try:
            os.system(f"wget -O {ckpt_path} https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt")
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            assert "Please contact fabien.baradel@naverlabs.com or open an issue on the github repo"

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # Build the model.
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Weights have been loaded")

    return model

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)
            #print (humans)

    return humans

def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):

    # Color of humans seen in the image.
    _color = [color[0] for _ in range(len(humans))] if unique_color else color
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    verts_list = [humans[j]['verts_smplx'].cpu().numpy() for j in range(len(humans))]
    faces_list = [model.smpl_layer['neutral'].bm_x.faces for j in range(len(humans))]

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(np.asarray(img_pil), 
            verts_list,
            faces_list,
            {'focal': focal, 'princpt': princpt},
            alpha=1.0,
            color=_color)

    return pred_rend_array, _color

def apply_ffmpeg_compression(input_video_path, output_video_path):
    """Apply FFMPEG Resizing over input video and store it temporarily"""
    filename = os.path.basename(input_video_path)
    temp_folder = 'temp_compressed/'
    (
        ffmpeg
        .input("lite/gBR_sBM_c01_d04_mBR0_ch02.mp4")
        .hflip()
        .output(output_video_path)
        .run()
    )

if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--model_name", type=str, default='multiHMR_896_L')
        parser.add_argument("--video", type=str, default='')
        parser.add_argument("--ffmpeg", type=int, default=0, choices=[0,1])
        parser.add_argument("--out_folder", type=str, default='demo_out')
        parser.add_argument("--start_frame", type=int, default=0)
        parser.add_argument("--end_frame", type=int, default=10000000)
        parser.add_argument("--save_pkl", type=int, default=0, choices=[0,1])
        parser.add_argument("--save_mesh", type=int, default=0, choices=[0,1])
        parser.add_argument("--save_image", type=int, default=1, choices=[0,1])
        parser.add_argument("--extra_views", type=int, default=0, choices=[0,1])
        parser.add_argument("--det_thresh", type=float, default=0.3)
        parser.add_argument("--nms_kernel_size", type=float, default=3)
        parser.add_argument("--fov", type=float, default=60)
        parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
        parser.add_argument("--unique_color", type=int, default=0, choices=[0,1], help='only one color for all humans')
        
        args = parser.parse_args()

        dict_args = vars(args)

        assert torch.cuda.is_available()

        # SMPL-X models
        smplx_fn = os.path.join(SMPLX_DIR, 'smplx', 'SMPLX_NEUTRAL.npz')
        if not os.path.isfile(smplx_fn):
            print(f"{smplx_fn} not found, please download SMPLX_NEUTRAL.npz file")
            print("To do so you need to create an account in https://smpl-x.is.tue.mpg.de")
            print("Then download 'SMPL-X-v1.1 (NPZ+PKL, 830MB) - Use thsi for SMPL-X Python codebase'")
            print(f"Extract the zip file and move SMPLX_NEUTRAL.npz to {smplx_fn}")
            print("Sorry for this incovenience but we do not have license for redustributing SMPLX model")
            assert NotImplementedError
        else:
             print('SMPLX found')
             
        # SMPL mean params download
        if not os.path.isfile(MEAN_PARAMS):
            print('Start to download the SMPL mean params')
            os.system(f"wget -O {MEAN_PARAMS}  https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4")
            print('SMPL mean params have been succesfully downloaded')
        else:
            print('SMPL mean params is already here')

        # Loading
        model = load_model(args.model_name)

        # Model name for saving results.
        model_name = os.path.basename(args.model_name)

        # All images
        os.makedirs(args.out_folder, exist_ok=True)
        l_duration = []

        # Creating a VideoCapture object to read the video
        if(args.ffmpeg):
            temp_filename = os.path.basename(args.video)
            temp_filepath = os.path.join("temp_compressed", temp_filename)
            apply_ffmpeg_compression(args.video, os.path.join(temp_filepath))
            cap = cv2.VideoCapture(temp_filepath)
        else:   
            cap = cv2.VideoCapture(args.video)


        allFrames = []
        startTime = time.time()

        i = 0 
        # Loop until the end of the video
        while (cap.isOpened()):
            allHumans = []
            ret, frame = cap.read()
            if (ret==True):
                print (i)
                if (i >= args.start_frame) and (i <= args.end_frame):
                    # Path where the image + overlays of human meshes + optional views will be saved.
                    save_fn = os.path.join(args.out_folder, str(i).zfill(6)+".png")

                    # Get input in the right format for the model
                    img_size = model.img_size


                    converted = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    pil_im = Image.fromarray(converted)

                    x, img_pil_nopad = open_image(pil_im, img_size)

                    # Get camera parameters
                    p_x, p_y = None, None
                    K = get_camera_parameters(model.img_size, fov=args.fov, p_x=p_x, p_y=p_y)
                    # print (K)
                    # Make model predictions
                    start = time.time()
                    humans = forward_model(model, x, K,
                                           det_thresh=args.det_thresh,
                                           nms_kernel_size=args.nms_kernel_size)

                   
                    duration = time.time() - start
                    l_duration.append(duration)
                    human = {}
                    # Convert tensors to regular variables
                    for h in humans:
                        # scores
                        # loc
                        # transl
                        # transl_pelvis
                        # rotvec
                        # expression
                        # shape
                        # verts_smplx
                        # j3d_smplx
                        # j2d_smplx
                        human['scores']=h['scores'].cpu().numpy()
                        human['loc']=h['loc'].cpu().numpy()
                        human['transl']=h['transl'].cpu().numpy()
                        human['transl_pelvis']=h['transl_pelvis'].cpu().numpy()
                        human['rotvec']=h['rotvec'].cpu().numpy()
                        human['expression']=h['expression'].cpu().numpy()
                        human['shape']=h['shape'].cpu().numpy()
                        human['verts_smplx']=h['verts_smplx'].cpu().numpy()
                        human['j3d_smplx']=h['j3d_smplx'].cpu().numpy()
                        human['j2d_smplx']=h['j2d_smplx'].cpu().numpy()
                        allHumans.append(human)

                    img_array = []
                    img_pil_visu = []
                    if (args.save_mesh or args.save_image):
                        img_array = np.asarray(img_pil_nopad)
                        img_pil_visu = Image.fromarray(img_array)
                        
                    # Superimpose predicted human meshes to the input image.
                    if args.save_image:
                        pred_rend_array, _color = overlay_human_meshes(humans, K, model, img_pil_visu, unique_color=args.unique_color)

                        # Optionally add distance as an annotation to each mesh
                        if args.distance:
                            pred_rend_array = print_distance_on_image(pred_rend_array, humans, _color)

                        # List of images too view side by side.
                        l_img = [img_array, pred_rend_array]

                        # More views
                        if args.extra_views:
                            # Render more side views of the meshes.
                            pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev = render_side_views(img_array, _color, humans, model, K)

                            # Concat
                            _img1 = np.concatenate([img_array, pred_rend_array],1).astype(np.uint8)
                            _img2 = np.concatenate([pred_rend_array_bis, pred_rend_array_sideview, pred_rend_array_bev],1).astype(np.uint8)
                            _h = int(_img2.shape[0] * (_img1.shape[1]/_img2.shape[1]))
                            _img2 = np.asarray(Image.fromarray(_img2).resize((_img1.shape[1], _h)))
                            _img = np.concatenate([_img1, _img2],0).astype(np.uint8)
                        else:
                            # Concatenate side by side
                            _img = np.concatenate([img_array, pred_rend_array],1).astype(np.uint8)

                        # Save to path.
                        Image.fromarray(_img).save(save_fn)
                    print(f"Avg Multi-HMR inference time={int(1000*np.median(np.asarray(l_duration[-1:])))}ms on a {torch.cuda.get_device_name()}")

                    # Saving mesh
                    if args.save_mesh:
                        # npy file
                        l_mesh = [hum['verts_smplx'].cpu().numpy() for hum in humans]
                        mesh_fn = save_fn+'.npy'
                        np.save(mesh_fn, np.asarray(l_mesh), allow_pickle=True)
                        x = np.load(mesh_fn, allow_pickle=True)

                        # glb file
                        l_mesh = [humans[j]['verts_smplx'].detach().cpu().numpy() for j in range(len(humans))]
                        l_face = [model.smpl_layer['neutral'].bm_x.faces for j in range(len(humans))]
                        scene = create_scene(img_pil_visu, l_mesh, l_face, color=None, metallicFactor=0., roughnessFactor=0.5)
                        scene_fn = save_fn+'.glb'
                        scene.export(scene_fn)

                allFrames.append(allHumans)
                i += 1
            else: 
                    break
                
        endTime = time.time()
        print("Time: ",endTime - startTime)
        print("Time par frame: ",(endTime - startTime)/i)
        print("IPS: ",i/(endTime - startTime))

        if args.save_pkl:
            save_pkl = os.path.join(args.out_folder,'allFrames.pkl')
            with open(save_pkl, 'wb') as handle:
                pickle.dump(allFrames, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        print('end')
        # delete the contents of temp_compressed


def extract_frames(input_video, output_dir, frame_rate=30):
     os.makedirs(output_dir, exist_ok=True)
     cmd = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"fps={frame_rate}",
        os.path.join(output_dir, "frame_%04d.jpg")
     ]
     subprocess.run(cmd)

def reconstruct_video(input_dir, output_video, frame_rate = 30):
     cmd = [
          "ffmpeg",
          "-framerate", str(frame_rate),
          "-i", os.path.join(input_dir, "frame_%04d.jpg"),
          "-c:v", "libx264",
          "-pix_fmt", "yuv420p",
          output_video
     ]
     subprocess.run(cmd)

