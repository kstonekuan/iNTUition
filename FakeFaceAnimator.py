import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
from PIL import Image
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale

def run_batch(source_images, pose_images):
    return model(pose_images, *source_images)

def load_src_img(file_path):
    img = Image.open(file_path)  
    transform = Compose([Scale((256,256)), ToTensor()])
    return Variable(transform(img)).cuda()

def load_drv_img(file_path):
    img = Image.open(file_path)
    width, height = img.size   # Get dimensions
    
    buffer = (width - height) / 2 + 400
    
    left = buffer
    top = 400
    right = width - buffer
    bottom = height - 200

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    
    transform = Compose([Scale((256,256)), ToTensor()])
    return Variable(transform(img)).cuda()

BASE_MODEL = './release_models/' # Change to your path
state_dict = torch.load(BASE_MODEL + 'x2face_model_forpython3.pth', map_location='cpu')
model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
model.load_state_dict(state_dict['state_dict'])
model = model.cuda()
model = model.eval()

driver_path = './examples/chester/'
source_path = './examples/test/'

for i in range (13):
    driver_imgs = [driver_path + d for d in sorted(os.listdir(driver_path))][i:i+1] # 1 driving frame
    source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:1] # 1 source frame
    # Driving the source image with the driving sequence
    source_images = []
    for img in source_imgs:
        source_images.append(load_src_img(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))

    driver_images = None
    for img in driver_imgs:
        if driver_images is None:
            driver_images = load_drv_img(img).unsqueeze(0)
        else:
            driver_images = torch.cat((driver_images, load_drv_img(img).unsqueeze(0)), 0)

    # Run the model for each
    with torch.no_grad():
        result = run_batch(source_images, driver_images)
    result = result.clamp(min=0, max=1)
    img = torchvision.utils.make_grid(result.cpu().data)

    # Visualise the results
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 24.
    fig_size[1] = 24.
    plt.rcParams["figure.figsize"] = fig_size
    plt.axis('off')

    result_images = img.permute(1,2,0).numpy()
    driving_images = torchvision.utils.make_grid(driver_images.cpu().data).permute(1,2,0).numpy()

    plt.imsave('../out/out{}.png'.format(i), result_images)