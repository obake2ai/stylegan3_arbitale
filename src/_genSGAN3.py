import os
import os.path as osp
import argparse
import random
import numpy as np
from imageio import imsave

import torch

import dnnlib
import legacy
from torch_utils import misc

from util.utilgan import latent_anima, basename, img_read
from util.progress_bar import progbar

import moviepy.editor

desc = "Customized StyleGAN3 on PyTorch"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--out_dir', default='_out', help='output directory')
parser.add_argument('-m', '--model', default='models/ffhq-1024.pkl', help='path to pkl checkpoint file')
parser.add_argument('-l', '--labels', type=int, default=None, help='labels/categories for conditioning')
# custom
parser.add_argument('-s', '--size', default=None, help='Output resolution')
parser.add_argument('-sc', '--scale_type', default='pad', help="may include pad, side, symm (also centr, fit)")
parser.add_argument('-lm', '--latmask', default=None, help='external mask file (or directory) for multi latent blending')
parser.add_argument('-n', '--nXY', default='1-1', help='multi latent frame split count by X (width) and Y (height)')
parser.add_argument('--splitfine', type=float, default=0, help='multi latent frame split edge sharpness (0 = smooth, higher => finer)')
parser.add_argument('--splitmax', type=int, default=None, help='max count of latents for frame splits (to avoid OOM)')
parser.add_argument('--trunc', type=float, default=0.8, help='truncation psi 0..1 (lower = stable, higher = various)')
parser.add_argument('--save_lat', action='store_true', help='save latent vectors to file')
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument("--noise_seed", type=int, default=3025)
# animation
parser.add_argument('-f', '--frames', default='200-25', help='total frames to generate, length of interpolation step')
parser.add_argument("--cubic", action='store_true', help="use cubic splines for smoothing")
parser.add_argument("--gauss", action='store_true', help="use Gaussian smoothing")
# transform SG3
parser.add_argument('-at', "--anim_trans", action='store_true', help="add translation animation")
parser.add_argument('-ar', "--anim_rot", action='store_true', help="add rotation animation")
parser.add_argument('-sb', '--shiftbase', type=float, default=0., help='Shift to the tile center?')
parser.add_argument('-sm', '--shiftmax',  type=float, default=0., help='Random walk around tile center')
parser.add_argument('--digress', type=float, default=0, help='distortion technique by Aydao (strength of the effect)')
#Affine Convertion
parser.add_argument('--affine_angle', type=float, default=0.0)
parser.add_argument('--affine_transform', default='0.0-0.0')
parser.add_argument('--affine_scale', default='1.0-1.0')
#Video Setting
parser.add_argument('--framerate', default=30)
parser.add_argument('--prores', action='store_true', help='output video in ProRes format')

a = parser.parse_args()

if a.size is not None:
    a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size) == 1: a.size = a.size * 2
[a.frames, a.fstep] = [int(s) for s in a.frames.split('-')]

if a.affine_transform is not None: a.affine_transform = [float(s) for s in a.affine_transform.split('-')][::-1]
if a.affine_scale is not None: a.affine_scale = [float(s) for s in a.affine_scale.split('-')][::-1]

def transform(G, angle, tx, ty, sx, sy):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)

    m[0][0] = sx*c
    m[0][1] = sx*s
    m[0][2] = tx
    m[1][0] = -sy*s
    m[1][1] = sy*c
    m[1][2] = ty

    m = np.linalg.inv(m)
    G.synthesis.input.transform.copy_(torch.from_numpy(m))

def make_out_name(a):
    def fmt_f(v):
        return str(v).replace('.', '_')

    model_name = basename(a.model)

    out_name = f"{model_name}_seed{a.noise_seed}"

    if a.size is not None:
        out_name += f"_size{a.size[1]}x{a.size[0]}"

    out_name += f"_nXY{a.nXY}"
    out_name += f"_frames{a.frames}-{a.fstep}"
    out_name += f"_trunc{fmt_f(a.trunc)}"
    if a.cubic:
        out_name += "_cubic"
    if a.gauss:
        out_name += "_gauss"
    if a.anim_trans:
        out_name += "_at"
    if a.anim_rot:
        out_name += "_ar"
    out_name += f"_sb{fmt_f(a.shiftbase)}"
    out_name += f"_sm{fmt_f(a.shiftmax)}"
    out_name += f"_digress{fmt_f(a.digress)}"
    if (a.affine_angle != 0.0 or
        a.affine_transform != [0.0, 0.0] or
        a.affine_scale    != [1.0, 1.0]):
        out_name += "_affine"
        out_name += f"_a{fmt_f(a.affine_angle)}"
        out_name += f"_t{fmt_f(a.affine_transform[0])}-{fmt_f(a.affine_transform[1])}"
        out_name += f"_s{fmt_f(a.affine_scale[0])}-{fmt_f(a.affine_scale[1])}"
    out_name += f"_fps{a.framerate}"

    return out_name

def checkout(output, i):
    ext = 'png' if output.shape[3]==4 else 'jpg'
    filename = osp.join(a.out_dir, "%06d.%s" % (i,ext))
    imsave(filename, output[0], quality=95)

def generate(noise_seed):
    torch.manual_seed(noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda')

    # setup generator
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type

    # mask/blend latents with external latmask or by splitting the frame
    if a.latmask is None:
        nHW = [int(s) for s in a.nXY.split('-')][::-1]
        assert len(nHW)==2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        if a.splitmax is not None: n_mult = min(n_mult, a.splitmax)
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = a.splitfine
        if a.splitmax is not None: Gs_kwargs.splitmax = a.splitmax
        if a.verbose is True and n_mult > 1: print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = [None]

    else:
        n_mult = 2
        nHW = [1,1]
        if osp.isfile(a.latmask): # single file
            lmask = np.asarray([[img_read(a.latmask)[:,:,0] / 255.]]) # [1,1,h,w]
        elif osp.isdir(a.latmask): # directory with frame sequence
            lmask = np.expand_dims(np.asarray([img_read(f)[:,:,0] / 255. for f in img_list(a.latmask)]), 1) # [n,1,h,w]
        else:
            print(' !! Blending mask not found:', a.latmask); exit(1)
        if a.verbose is True: print(' Latent blending with mask', a.latmask, lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1) # [n,2,h,w]
        lmask = torch.from_numpy(lmask).to(device)

    # load base or custom network
    pkl_name = osp.splitext(a.model)[0]
    if '.pkl' in a.model.lower():
        custom = False
        print(' .. Gs from pkl ..', basename(a.model))
    else:
        custom = True
        print(' .. Gs custom ..', basename(a.model))
    rot = True if ('-r-' in a.model.lower() or 'sg3r-' in a.model.lower()) else False
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device) # type: ignore

    if a.size is None: a.size = [Gs.img_resolution] * 2

    if a.verbose is True: print(' making timeline..')
    latents = latent_anima((n_mult, Gs.z_dim), a.frames, a.fstep, cubic=a.cubic, gauss=a.gauss, seed=a.noise_seed, verbose=False) # [frm,X,512]
    print(' latents', latents.shape)
    latents = torch.from_numpy(latents).to(device)
    frame_count = latents.shape[0]

    # labels / conditions
    label_size = Gs.c_dim
    if label_size > 0:
        labels = torch.zeros((frame_count, n_mult, label_size), device=device) # [frm,X,lbl]
        if a.labels is None:
            label_ids = []
            for i in range(n_mult):
                label_ids.append(random.randint(0, label_size-1))
        else:
            label_ids = [int(x) for x in a.labels.split('-')]
            label_ids = label_ids[:n_mult] # ensure we have enough labels
        for i, l in enumerate(label_ids):
            labels[:,i,l] = 1
    else:
        labels = [None]

    # NEW SG3
    if hasattr(Gs.synthesis, 'input'): # SG3
        if a.anim_trans is True:
            hw_centers = [np.linspace(-1+1/n, 1-1/n, n) for n in nHW]
            yy,xx = np.meshgrid(*hw_centers)
            xscale = [s / Gs.img_resolution for s in a.size]
            hw_centers = np.dstack((yy.flatten()[:n_mult], xx.flatten()[:n_mult])) * xscale * 0.5 * a.shiftbase
            hw_scales = np.array([2. / n for n in nHW]) * a.shiftmax
            shifts = latent_anima((n_mult, 2), a.frames, a.fstep, uniform=True, cubic=a.cubic, gauss=a.gauss, seed=a.noise_seed, verbose=False) # [frm,X,2]
            shifts = hw_centers + (shifts - 0.5) * hw_scales
        else:
            shifts = np.zeros((1, n_mult, 2))
        if a.anim_rot is True:
            angles = latent_anima((n_mult, 1), a.frames, a.frames//4, uniform=True, cubic=a.cubic, gauss=a.gauss, seed=a.noise_seed, verbose=False) # [frm,X,1]
            angles = (angles - 0.5) * 180.
        else:
            angles = np.zeros((1, n_mult, 1))
        # 拡大率 (scale_x, scale_y) を各フレームに反映
        # ここでは a.affine_scale = [scale_y, scale_x] を想定し、全フレーム同一値にしています
        # (毎フレームアニメさせたい場合は latent_anima 等で生成してもOK)
        scale_array = np.array([a.affine_scale[0], a.affine_scale[1]], dtype=np.float32)  # (scale_y, scale_x)
        # shifts.shape = [frame_count, n_mult, 2]
        # angles.shape = [frame_count, n_mult, 1]
        #  → 同じフレーム数・枚数に合わせて scale_array をタイル展開
        scales = np.tile(scale_array, (shifts.shape[0], shifts.shape[1], 1))  # [frame_count, n_mult, 2]

        shifts = torch.from_numpy(shifts).to(device)
        angles = torch.from_numpy(angles).to(device)
        scales = torch.from_numpy(scales).to(device)   # [frame_count, X, 2]

        trans_params = list(zip(shifts, angles, scales))

    # Affine Convertion ***not working ***
    if (a.affine_transform != [0.0, 0.0] or a.affine_scale != [1.0, 1.0] or a.affine_angle != 0.0):
        print("Applying Affine Convertion...")
        transform(Gs, a.affine_angle, a.affine_transform[0], a.affine_transform[1], a.affine_scale[0], a.affine_scale[1])

    # distort image by tweaking initial const layer
    first_layer_channels = Gs.synthesis.input.channels
    first_layer_size     = Gs.synthesis.input.size
    if isinstance(first_layer_size, (list, tuple, np.ndarray)):
        h, w = first_layer_size[0], first_layer_size[1]
    else:
        h, w = first_layer_size, first_layer_size

    shape_for_dconst = [1, first_layer_channels, h, w]
    #("debug shape_for_dconst =", shape_for_dconst)

    if a.digress != 0:
        dconst_list = []
        for i in range(n_mult):
            dc_tmp = a.digress * latent_anima(
                shape_for_dconst,  # [1, 1024, 36, 36] 等
                a.frames, a.fstep, cubic=True, seed=noise_seed, verbose=False
            )
            dconst_list.append(dc_tmp)
        dconst = np.concatenate(dconst_list, axis=1)
    else:
        dconst = np.zeros([latents.shape[0], 1, first_layer_channels, h, w])

    dconst = torch.from_numpy(dconst).to(device).to(torch.float32)

    # warm up
    if custom:
        if hasattr(Gs.synthesis, 'input'): # SG3
            _ = Gs(latents[0], labels[0], lmask[0], trans_params[0], dconst[0], noise_mode='const')
        else: # SG2
            _ = Gs(latents[0], labels[0], lmask[0], noise_mode='const')
    else:
        _ = Gs(latents[0], labels[0], noise_mode='const')

    # Video Generation
    frame_count = latents.shape[0]
    duration_sec = frame_count / a.framerate
    out_name = make_out_name(a)

    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * 30), 0, frame_count - 1))
        latent  = latents[frame_idx] # [X,512]
        label   = labels[frame_idx % len(labels)]
        latmask = lmask[frame_idx % len(lmask)] # [X,h,w] or None
        dc      = dconst[frame_idx % len(dconst)] # [X,512,4,4]

        if hasattr(Gs.synthesis, 'input'): # SG3
            trans_param = trans_params[frame_idx % len(trans_params)]

        # generate multi-latent result
        if custom:
            if hasattr(Gs.synthesis, 'input'): # SG3
                output = Gs(latent, label, latmask, trans_param, dc, truncation_psi=a.trunc, noise_mode='const')
            else: # SG2
                output = Gs(latent, label, latmask, truncation_psi=a.trunc, noise_mode='const')
        else:
            output = Gs(latent, label, truncation_psi=a.trunc, noise_mode='const')
        output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        return output[0]

    if a.prores:
        moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(
            osp.join(a.out_dir, "%s.mov" % out_name),
            fps=a.framerate,
            codec='prores',
            bitrate='16M'
        )
    else:
        moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(
            osp.join(a.out_dir, "%s.mp4" % out_name),
            fps=a.framerate,
            codec='libx264',
            bitrate='16M'
        )
    # generate images from latent timeline
    # pbar = progbar(frame_count)
    # for i in range(frame_count):
    #
    #     latent  = latents[i] # [X,512]
    #     label   = labels[i % len(labels)]
    #     latmask = lmask[i % len(lmask)] # [X,h,w] or None
    #     if hasattr(Gs.synthesis, 'input'): # SG3
    #         trans_param = trans_params[i % len(trans_params)]
    #
    #     # generate multi-latent result
    #     if custom:
    #         if hasattr(Gs.synthesis, 'input'): # SG3
    #             output = Gs(latent, label, latmask, trans_param, truncation_psi=a.trunc, noise_mode='const')
    #         else: # SG2
    #             output = Gs(latent, label, latmask, truncation_psi=a.trunc, noise_mode='const')
    #     else:
    #         output = Gs(latent, label, truncation_psi=a.trunc, noise_mode='const')
    #     output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    #
    #     # save image
    #     checkout(output, i)
    #     pbar.upd()


    if a.save_lat is True:
        latents = latents.squeeze(1) # [frm,512]
        if a.size is None: a.size = ['']*2
        filename = '{}-{}-{}.npy'.format(basename(a.model), a.size[1], a.size[0])
        filename = osp.join(osp.dirname(a.out_dir), filename)
        latents = latents.cpu().numpy()
        np.save(filename, latents)
        print('saved latents', latents.shape, 'to', filename)


if __name__ == '__main__':
    generate(a.noise_seed)
