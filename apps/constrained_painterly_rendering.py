# Created from painterly_rendering.py

"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import os

pydiffvg.set_print_timing(True)

gamma = 1.0

def main(args):
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())
    
    target_path = '/Users/simon/code/plots/resources/avrecum2 Plot/original_resized.jpeg'
    initial_svg_path = '/Users/simon/code/plots/resources/avrecum2 Plot/hatched_preview.svg'
    
    #target = torch.from_numpy(skimage.io.imread('imgs/lena.png')).to(torch.float32) / 255.0
    target = torch.from_numpy(skimage.io.imread(target_path)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW
    #target = torch.nn.functional.interpolate(target, size = [256, 256], mode = 'area')
    canvas_width, canvas_height = target.shape[3], target.shape[2]
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    # Load SVG
    svg_width, svg_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(initial_svg_path)
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    print(svg_width)
    print(canvas_width)
    assert svg_width == canvas_width and svg_height == canvas_height
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/constrained_painterly_rendering/init.png', gamma=gamma)

    # Collect variables to optimize
    points_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    
    # Optimize
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), 'results/constrained_rendering/iter_{}.png'.format(t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).abs().mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        # color_optim.step()

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('results/constrained_painterly_rendering/iter_{}.svg'.format(t),
                              canvas_width, canvas_height, shapes, shape_groups)
    
    # Render the final result.
    img = render(target.shape[1], # width
                 target.shape[0], # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), 'results/constrained_painterly_rendering/final.png'.format(t), gamma=gamma)
    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "results/constrained_painterly_rendering/iter_%d.png", "-vb", "20M",
        "results/constrained_painterly_rendering/out.mp4"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("target", help="target image path")
    # parser.add_argument("--num_paths", type=int, default=512)
    # parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    # parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    args = parser.parse_args()
    main(args)
