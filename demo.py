import os
import os.path as osp

import json
import sys
sys.path.append("/mnt/efs/packages")
sys.path.append("/mnt/efs/stoke_portrait")
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

import argparse
import torch
import torch.optim as optim

from model import BiSeNet
from painter import *
import time

# settings
parser = argparse.ArgumentParser(description='Stroke Portrait')
parser.add_argument('--iters_per_stroke', type=int, default=50)
parser.add_argument('--img_path', type=str, default='./output/abc.jpeg', metavar='str',
                    help='path to test image (default: ./test_images/apple.jpg)')
parser.add_argument('--renderer', type=str, default='markerpen', metavar='str',
                    help='renderer: [watercolor, markerpen, oilpaintbrush, rectangle (default oilpaintbrush)')
parser.add_argument('--canvas_color', type=str, default='white', metavar='str',
                    help='canvas_color: [black, white] (default black)')
parser.add_argument('--canvas_size', type=int, default=512, metavar='str',
                    help='size of the canvas for stroke rendering')
parser.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                    help='keep input aspect ratio when saving outputs')
parser.add_argument('--max_m_strokes', type=int, default=15, metavar='str',
                    help='max number of strokes (default 500)')
parser.add_argument('--max_divide', type=int, default=1, metavar='N',
                    help='divide an image up-to max_divide x max_divide patches (default 5)')
parser.add_argument('--beta_L1', type=float, default=1.0,
                    help='weight for L1 loss (default: 1.0)')
parser.add_argument('--with_ot_loss', action='store_true', default=True,
                    help='imporve the convergence by using optimal transportation loss')
parser.add_argument('--beta_ot', type=float, default=0.1,
                    help='weight for optimal transportation loss (default: 0.1)')
parser.add_argument('--net_G', type=str, default='zou-fusion-net', metavar='str',
                    help='net_G: plain-dcgan, plain-unet, huang-net, zou-fusion-net, '
                         'or zou-fusion-net-light (default: zou-fusion-net-light)')
parser.add_argument('--renderer_checkpoint_dir', type=str, default=r'./checkpoints_G_markerpen', metavar='str',
                    help='dir to load neu-renderer (default: ./checkpoints_G_oilpaintbrush_light)')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate for stroke searching (default: 0.005)')
parser.add_argument('--output_dir', type=str, default='./output', metavar='str',
                    help='dir to save painting results (default: ./output)')
parser.add_argument('--output_filename', type=str, default='portrait-final.png', metavar='str',
                    help='dir to save painting results (default: ./output)')
parser.add_argument('--disable_preview', action='store_true', default=True,
                    help='disable cv2.imshow, for running remotely without x-display')
args = parser.parse_args()

#args.output_dir = f"{args.output_dir}/{args.img_path.split('/')[-1].split('.')[0]}"
#print(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.get_device_name(device))

def vis_parsing_maps(im, parsing_anno, stride, save_im=False):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    # Save result or not    
    if save_im:
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255        
        # Colors for all 20 parts
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]

        
        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        cv2.imwrite(f"{args.output_dir}/face_map.png", vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_parsing_anno

def evaluate(img_path, checkpoint_path):

    #net = BiSeNet(n_classes=19).cuda()
    net = BiSeNet(n_classes=19)

    #print(torch.cuda.is_available(device))

    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    with torch.no_grad():
        img = Image.open(osp.join(img_path)).convert('RGB')
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0) #img = torch.unsqueeze(img, 0).cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        vis_parsing_anno = vis_parsing_maps(image, parsing, stride=1, save_im=False)
    return vis_parsing_anno


def clamp_x():
    pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1) 
    pt.x_ctt.data[:,:,-2:]= torch.clamp(pt.x_ctt.data[:,:,-2:], 0.1, 0.3) #radius
    pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
    pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0.95, 1)

    
def step(pt, CANVAS_tmp, clamp=True):
    pt.G_pred_canvas = CANVAS_tmp

    # update x
    pt.optimizer_x.zero_grad()
    clamp_x()

    pt._forward_pass()
#     pt._drawing_step_states()
    pt._backward_x()

    clamp_x()
    pt.optimizer_x.step()
    pt.step_id += 1    
    
def optimize_x(pt):

    pt._load_checkpoint()
    pt.net_G.eval()

#     print('begin drawing...')

    PARAMS = np.zeros([1, 0, pt.rderr.d], np.float32)

    if pt.rderr.canvas_color == 'white':
        CANVAS_tmp = torch.ones([1, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
    else:
        CANVAS_tmp = torch.zeros([1, 3, pt.net_G.out_size, pt.net_G.out_size]).to(device)
        
    pt.m_grid = 1
    pt.img_batch = torch.tensor(pt.img_.transpose([2, 0, 1])).unsqueeze(0).to(device)
    pt.G_final_pred_canvas = CANVAS_tmp

    pt.initialize_params()
    pt.x_ctt.requires_grad = True 
    pt.x_color.requires_grad = True 
    pt.x_alpha.requires_grad = True 
    utils.set_requires_grad(pt.net_G, False)

    pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr, centered=True)

    # step1: draw face outline
    pt.step_id = 0
    pt.preprocess_face()
    for pt.anchor_id in range(0, pt.m_strokes_per_block):
        pt.stroke_sampler(pt.anchor_id)
        for i in range(args.iters_per_stroke):
            step(pt, CANVAS_tmp, clamp=True)

    # step2: draw eyes, nose, mouth
    pt.postprocess_face()
    for pt.anchor_id in range(pt.m_strokes_per_block, pt.m_strokes_per_block+pt.len_post):
        pt.stroke_sampler(pt.anchor_id)
        for i in range(2):
            step(pt, CANVAS_tmp, clamp=True)

    v = pt._normalize_strokes(pt.x)
    v = pt._shuffle_strokes_and_reshape(v)
    PARAMS = np.concatenate([PARAMS, v], axis=1)
    CANVAS_tmp = pt._render(PARAMS, save_jpgs=False, save_video=False)
    CANVAS_tmp = utils.img2patches(CANVAS_tmp, pt.m_grid + 1, pt.net_G.out_size).to(device)

#     pt._save_stroke_params(PARAMS)
    final_rendered_image = pt._render(PARAMS, save_jpgs=True, save_video=False)



if __name__ == '__main__':
    st = time.time()
    args.output_filename = 'portrait-output.png'
    face_attributes = evaluate(img_path=args.img_path, checkpoint_path='./79999_iter.pth')    
    pt = ProgressivePainter(args, face_attributes)
    optimize_x(pt)
    end = time.time()
    print(end-st)

