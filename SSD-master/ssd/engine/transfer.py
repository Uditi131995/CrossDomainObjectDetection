from pathlib import Path
import random

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import ssd.adain.net as net
from ssd.adain.function import adaptive_instance_normalization


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)

    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def apply_style_transfer(vgg_path, decoder_path, content_batch, style_batch, p):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed()
    alpha = 1.0

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_len = list(content_batch.shape)[0]
    style_len = list(style_batch.shape)[0]

    # process one content and one style
    for i in range(content_len):
        j = random.randrange(style_len)

        # batch tensors have shape [32, 3, 300, 300]
        content = content_batch[i, :, :, :]
        style = style_batch[j, :, :, :]

        content = content.to(device).unsqueeze(0)
        style = style.to(device).unsqueeze(0)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style, alpha)

        # decoder output tensor has shape [1, 3, 304, 304]
        content_batch[i, :, :, :] = output[:, :, :300, :300]
