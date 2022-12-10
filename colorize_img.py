from __future__ import print_function

import os

import torch
import torchvision.transforms as transform_lib
from PIL import Image, ImageOps

import lib.TestTransforms as transforms
from models.FrameColor import frame_colorization
from utils.util import (batch_lab2rgb_transpose_mc, mkdir_if_not, save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import Normalize, RGB2Lab, ToTensor


def colorize_image(input_path, reference_file, output_path, nonlocal_net, colornet, vggnet):
    mkdir_if_not(output_path)
    transform = transforms.Compose(
        [transform_lib.Resize((216 * 2, 384 * 2)), RGB2Lab(), ToTensor(), Normalize()]
    )

    reference_img = Image.open(reference_file)
    reference_img = reference_img.convert("RGB")

    IB_lab_large = transform(reference_img).unsqueeze(0).cpu()

    IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")

    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

    input_img = Image.open(os.path.join(input_path, "input.jpg"))
    input_img = ImageOps.exif_transpose(input_img)

    width, height = input_img.size
    input_img = input_img.convert("RGB")

    IA_lab_large = transform(input_img).unsqueeze(0).cpu()
    IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
    IA_l = IA_lab[:, 0:1, :, :]

    I_last_lab_predict = torch.zeros_like(IA_lab).cpu()

    # start the colorization
    with torch.no_grad():
        I_current_lab = IA_lab
        I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
            I_current_lab,
            I_reference_lab,
            I_last_lab_predict,
            features_B,
            vggnet,
            nonlocal_net,
            colornet,
            feature_noise=0,
            temperature=1e-10,
        )
        I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

    # upsampling
    curr_bs_l = IA_lab_large[:, 0:1, :, :]
    curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
    )

    # filtering
    IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

    # save the frames
    output_img = Image.fromarray(IA_predict_rgb)
    output_img = output_img.resize((width, height))
    save_frames(output_img, output_path, image_name="output.jpg")
