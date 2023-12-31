# Revising and adapting the code from the notebook for the requested functionalities

# Assumed imports based on the notebook
import os
import sys
import numpy as np
import cv2
import torch



sys.path.append('MotionEstimation/RAFT/core')
from raft import RAFT
from utiles import flow_viz
from utiles.utils import InputPadder

class Args():
  def __init__(self, model='', path='', small=False, mixed_precision=True, alternate_corr=False):
    self.model = model
    self.path = path
    self.small = small
    self.mixed_precision = mixed_precision
    self.alternate_corr = alternate_corr

  """ Sketchy hack to pretend to iterate through the class objects """
  def __iter__(self):
    return self

  def __next__(self):
    raise StopIteration

# Function to load the RAFT model
def load_model(weights_path, args=Args()):
    model = RAFT(args)
    pretrained_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to("cpu")
    return model

# Function to calculate optical flow using RAFT
def calculate_optical_flow(model, frame1, frame2):
    image1 = torch.from_numpy(frame1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(frame2).permute(2, 0, 1).float()

    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1[None], image2[None])

    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

    return flow_up[0].permute(1, 2, 0).cpu().numpy()

# Function to visualize optical flow
def visualize_optical_flow(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Function to filter optical flow
def filter_optical_flow(flow, threshold=1):
    mag = np.linalg.norm(flow, axis=2)
    filtered_flow = flow.copy()
    filtered_flow[mag < threshold] = 0
    return filtered_flow

# Function to segment movement
def segment_movement(flow, threshold=1):
    mag = np.linalg.norm(flow, axis=2)
    mask = mag > threshold
    segment = np.zeros_like(mag, dtype=np.uint8)
    segment[mask] = 255
    return segment

#Función para usar la mascara de segmentación
def segmentacion(frame, segmentacion):
    frame = cv2.bitwise_and(frame, frame, mask=segmentacion)
    return frame


def process_img(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def inference(model, frame1, frame2, device, pad_mode='sintel',
              iters=20, flow_init=None, upsample=True, test_mode=True):

    model.eval()
    with torch.no_grad():
        # preprocess
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)

        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)

        # predict flow
        if test_mode:
          flow_low, flow_up = model(frame1,
                                    frame2,
                                    iters=iters,
                                    flow_init=flow_init,
                                    upsample=upsample,
                                    test_mode=test_mode)



          return flow_low, flow_up

        else:
            flow_iters = model(frame1,
                               frame2,
                               iters=iters,
                               flow_init=flow_init,
                               upsample=upsample,
                               test_mode=test_mode)

            return flow_iters

import cv2
import torch
import numpy as np

# Asumiendo que todas tus funciones definidas anteriormente están aquí...



if __name__ == "__main__":
    # Uso de la función
    video_path = "Videos/trimmed_0.mp4"
    model_path = "MotionEstimation/RAFT/models/raft-things.pth"

