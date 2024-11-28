import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import yaml
import signal

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F

timestamps = []
def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(sequence_path, rgb_txt, calibration_yaml, stride):
    """ image generator """

    
    with open(calibration_yaml, 'r') as file:
        lines = file.readlines()
    if lines and lines[0].strip() == '%YAML:1.0':
        lines = lines[1:]

    calibration = yaml.safe_load(''.join(lines))  

    fx, fy, cx, cy = calibration["Camera.fx"],calibration["Camera.fy"],calibration["Camera.cx"],calibration["Camera.cy"]
    camera_distortion = True
    dist_coeffs = np.array([calibration["Camera.k1"], calibration["Camera.k2"], calibration["Camera.p1"], calibration["Camera.p2"], calibration["Camera.k3"]], dtype=np.float32)
    if (calibration["Camera.k1"] == 0 and calibration["Camera.k2"] == 0 and calibration["Camera.k3"] == 0
            and calibration["Camera.p1"] == 0 and calibration["Camera.p2"] == 0):
        camera_distortion = False

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy
       
    image_list = []
    timestamps.clear()
    with open(rgb_txt, 'r') as file:
        for line in file:
            timestamp, path, *extra = line.strip().split(' ')
            image_list.append(path)
            timestamps.append(timestamp)
            
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(sequence_path, imfile))
        if camera_distortion:
            image = cv2.undistort(image, K, dist_coeffs)

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence_path", type=str, help="path to image directory")
    parser.add_argument("--calibration_yaml", type=str, help="path to calibration file")
    parser.add_argument("--rgb_txt", type=str, help="path to image list")
    parser.add_argument("--exp_folder", type=str, help="path to save results")
    parser.add_argument("--exp_it", type=str, help="experiment iteration")
    parser.add_argument("--settings_yaml", type=str, help="settings_yaml")
    parser.add_argument("--verbose", type=str, help="verbose")
    parser.add_argument("--upsample", type=str, help="upsample")

    args = parser.parse_args()
    verbose = bool(int(args.verbose))
    args.disable_vis = not bool(int(args.verbose))
    args.upsample = bool(int(args.upsample))

    with open(args.settings_yaml, 'r') as file:
        settings = yaml.safe_load(file)

    args.t0 = settings['settings']['t0']
    args.stride = settings['settings']['stride']

    args.weights = settings['settings']['weights']
    args.buffer = settings['settings']['buffer']

    args.beta = settings['settings']['beta']
    args.filter_thresh = settings['settings']['filter_thresh']
    args.warmup = settings['settings']['warmup']

    args.keyframe_thresh = settings['settings']['keyframe_thresh']
    args.frontend_thresh = settings['settings']['frontend_thresh']
    args.frontend_window = settings['settings']['frontend_window']
    args.frontend_radius = settings['settings']['frontend_radius']
    args.frontend_nms = settings['settings']['frontend_nms']

    args.backend_thresh = settings['settings']['backend_thresh']
    args.backend_radius = settings['settings']['backend_radius']
    args.backend_nms = settings['settings']['backend_nms']

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    for (t, image, intrinsics) in tqdm(image_stream(args.sequence_path, args.rgb_txt, args.calibration_yaml, args.stride)):
        if t < args.t0:
            continue

        if verbose:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]         
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.sequence_path, args.rgb_txt, args.calibration_yaml, args.stride))
    
    poses = droid.video.poses[:t].cpu().numpy()
    keyFrameTrajectory_txt = os.path.join(args.exp_folder, args.exp_it.zfill(5) + '_KeyFrameTrajectory' + '.txt')
    
    with open(keyFrameTrajectory_txt, 'w') as file:
    	for i_pose, pose in enumerate(traj_est):
    	    ts = timestamps[i_pose]
    	    tx, ty, tz = pose[0], pose[1], pose[2]
    	    qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
    	    line = str(ts) + " " + str(tx) + " " + str(ty) + " " + str(tz) + " " + str(qx) + " " + str(qy) + " " + str(qz) + " " + str(qw) + "\n"
    	    file.write(line)
