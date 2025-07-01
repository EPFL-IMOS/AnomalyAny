import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch

from utils import ptp_utils
from utils.ptp_utils import AttentionStore, aggregate_attention
import matplotlib.pyplot as plt


def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    images = []

    # show spatial attention for all tokens
    for i in range(len(tokens)):
        if i == 0 or i == len(tokens) - 1: # Skip start and end tokens
            continue
        image = attention_maps[:, :, i]
        image = show_image_relevance(image, orig_image)
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
        token_text = decoder(int(tokens[i]))
        # Add a visual indicator for altered tokens
        if i in indices_to_alter:
            token_text = f"*{token_text}*"
        image = ptp_utils.text_under_image(image, token_text)
        images.append(image)

    ptp_utils.view_images(np.stack(images, axis=0))
    # TODO
    return images


def show_image_relevance(image_relevance, image: Image.Image, relevnace_res=16):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))
    image = np.array(image)

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1], image_relevance.shape[-1])
    image_relevance = image_relevance.cuda() # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2, mode='bilinear')
    image_relevance = image_relevance.cpu() # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image


def vis_ann(attention_images, threshold=40):
    att_image = attention_images[:256, :, :]
    att = ((0.2989*att_image[:, :, 0] + 0.5870*att_image[:, :, 1] + 0.1140*att_image[:, :, 2])).astype(int)
    # 0.2989 R + 0.5870 G + 0.1140 B 
    # att = Image.fromarray(att, 'RGB')
    # att.show()
    # att = att.convert('L')
    # att.show()
    # att = np.array(att)

    # print(att.shape)
    # print(att)
    # print(np.min(att), np.max(att))
    plt.imshow(att)
    # for i in range(np.min(att), np.max(att)):
        # print(i, np.sum(att==i))

    att_median = np.max(att) - threshold
    # print("median:", att_median)
    att_show = att.copy()
    att_show[att < att_median] = 255
    att_show[att >= att_median] = 0
    # print(att_show)
    # print("min:", np.min(att_show),"max:", np.max(att_show))
    # img = Image.fromarray(att_show, 'L')
    # img.show()
    plt.imshow(att_show)
    return att_show