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

from util.utilgan import latent_anima, basename, img_read, img_list
from util.progress_bar import progbar

import moviepy.editor

desc = "Customized StyleGAN3 on PyTorch"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-o', '--out_dir', default='_out', help='output directory')
parser.add_argument('-m', '--model', default='models/ffhq-1024.pkl', help='path to pkl checkpoint file')
parser.add_argument('-l', '--labels', type=str, default=None, help='labels/categories for conditioning')
# custom
parser.add_argument('-s', '--size', default=None, help='Output resolution, ex) 1024-1024')
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
parser.add_argument('-f', '--frames', default='200-25', help='total frames to generate, length of interpolation step, ex) 200-25')
parser.add_argument("--cubic", action='store_true', help="use cubic splines for smoothing")
parser.add_argument("--gauss", action='store_true', help="use Gaussian smoothing")
# transform SG3
parser.add_argument('-at', "--anim_trans", action='store_true', help="add translation animation")
parser.add_argument('-ar', "--anim_rot", action='store_true', help="add rotation animation")
parser.add_argument('-sb', '--shiftbase', type=float, default=0., help='Shift to the tile center?')
parser.add_argument('-sm', '--shiftmax',  type=float, default=0., help='Random walk around tile center')
parser.add_argument('--digress', type=float, default=0, help='distortion technique by Aydao (strength of the effect)')
# Affine Convertion
parser.add_argument('-as','--affine_scale', default='1.0-1.0', help='Scale factor for (height-width) ex) 1.0-1.0')
# Video Setting
parser.add_argument('--framerate', type=int, default=30)
parser.add_argument('--prores', action='store_true', help='output video in ProRes format')
parser.add_argument('--variations', type=int, default=1)
# 追加: 画像出力モード
parser.add_argument('--image', action='store_true', help='Save frames as individual images instead of making a video')

a = parser.parse_args()

# 前処理
if a.size is not None:
    a.size = [int(s) for s in a.size.split('-')][::-1]
    if len(a.size) == 1:
        a.size = a.size * 2
[a.frames, a.fstep] = [int(s) for s in a.frames.split('-')]

if a.affine_scale is not None:
    a.affine_scale = [float(s) for s in a.affine_scale.split('-')][::-1]


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
    if a.affine_scale != [1.0, 1.0]:
        out_name += "_affine"
        out_name += f"_s{fmt_f(a.affine_scale[0])}-{fmt_f(a.affine_scale[1])}"
    out_name += f"_fps{a.framerate}"

    return out_name


def generate(noise_seed):
    torch.manual_seed(noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup generator
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type

    # mask/blend latents with external latmask or by splitting the frame
    if a.latmask is None:
        nHW = [int(s) for s in a.nXY.split('-')][::-1]
        assert len(nHW) == 2, ' Wrong count nXY: %d (must be 2)' % len(nHW)
        n_mult = nHW[0] * nHW[1]
        if a.splitmax is not None:
            n_mult = min(n_mult, a.splitmax)
        Gs_kwargs.countHW = nHW
        Gs_kwargs.splitfine = a.splitfine
        if a.splitmax is not None:
            Gs_kwargs.splitmax = a.splitmax
        if a.verbose and n_mult > 1:
            print(' Latent blending w/split frame %d x %d' % (nHW[1], nHW[0]))
        lmask = [None]
    else:
        # external mask
        n_mult = 2
        nHW = [1, 1]
        if osp.isfile(a.latmask):
            # single file
            mask = img_read(a.latmask)
            lmask = np.asarray([[mask[:, :, 0] / 255.]])  # [1,1,h,w]
        elif osp.isdir(a.latmask):
            # directory -> multiple
            files = img_list(a.latmask)
            lmask = np.expand_dims(
                np.asarray([img_read(f)[:, :, 0] / 255. for f in files]),
                1
            )  # [n,1,h,w]
        else:
            print(' !! Blending mask not found:', a.latmask)
            exit(1)
        if a.verbose:
            print(' Latent blending with mask', a.latmask, lmask.shape)
        lmask = np.concatenate((lmask, 1 - lmask), 1)  # [n,2,h,w]
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
        Gs = legacy.load_network_pkl(
            f, custom=custom, rot=rot, **Gs_kwargs
        )['G_ema'].to(device)  # type: ignore

    # もしサイズ指定が無ければ、Gs から解像度を取得
    if a.size is None:
        a.size = [Gs.img_resolution] * 2

    if a.verbose:
        print(' making timeline..')

    # latents: [frame_count, n_mult, z_dim]
    latents = latent_anima(
        (n_mult, Gs.z_dim), a.frames, a.fstep,
        cubic=a.cubic, gauss=a.gauss,
        seed=a.noise_seed, verbose=False
    )
    latents = torch.from_numpy(latents).to(device)
    frame_count = latents.shape[0]

    # labels / conditions
    label_size = Gs.c_dim
    if label_size > 0:
        # [frame_count, n_mult, label_size]
        labels = torch.zeros((frame_count, n_mult, label_size), device=device)
        if a.labels is None:
            label_ids = [random.randint(0, label_size - 1) for _ in range(n_mult)]
        else:
            # 例: --labels 1-7-4
            label_ids = [int(x) for x in a.labels.split('-')]
            label_ids = label_ids[:n_mult]
        for i, l in enumerate(label_ids):
            labels[:, i, l] = 1
    else:
        labels = [None]  # dummy

    # set up parameters for SG3 if needed
    if hasattr(Gs.synthesis, 'input'):  # SG3
        # shift (anim_trans)
        if a.anim_trans:
            hw_centers = [np.linspace(-1 + 1/n, 1 - 1/n, n) for n in [nHW[0], nHW[1]]]
            yy, xx = np.meshgrid(*hw_centers)
            xscale = [s / Gs.img_resolution for s in a.size]
            hw_centers = np.dstack((yy.flatten()[:n_mult], xx.flatten()[:n_mult])) * xscale * 0.5 * a.shiftbase
            hw_scales = np.array([2. / n for n in nHW]) * a.shiftmax
            shifts = latent_anima(
                (n_mult, 2), a.frames, a.fstep,
                uniform=True, cubic=a.cubic, gauss=a.gauss,
                seed=a.noise_seed, verbose=False
            )  # [frame_count, n_mult, 2]
            shifts = hw_centers + (shifts - 0.5) * hw_scales
        else:
            shifts = np.zeros((1, n_mult, 2))

        # rotation (anim_rot)
        if a.anim_rot:
            angles = latent_anima(
                (n_mult, 1), a.frames, a.frames // 4,
                uniform=True, cubic=a.cubic, gauss=a.gauss,
                seed=a.noise_seed, verbose=False
            )  # [frame_count, n_mult, 1]
            angles = (angles - 0.5) * 180.
        else:
            angles = np.zeros((1, n_mult, 1))

        # scale (affine_scale)
        scale_array = np.array([a.affine_scale[0], a.affine_scale[1]], dtype=np.float32)
        # shifts.shape = [frame_count, n_mult, 2]
        # angles.shape = [frame_count, n_mult, 1]
        # scale_arrayをフレーム数でタイル
        frame_trans_count = max(shifts.shape[0], angles.shape[0])
        # もしアニメの長さがフレーム数と異なっていた場合、最長に合わせてタイリング
        # ここでは簡単のため frame_count で統一
        if shifts.shape[0] != frame_count:
            shifts = np.tile(shifts, (frame_count, 1, 1))
        if angles.shape[0] != frame_count:
            angles = np.tile(angles, (frame_count, 1, 1))

        scales = np.tile(scale_array, (frame_count, n_mult, 1))  # [frame_count, n_mult, 2]

        shifts = torch.from_numpy(shifts).to(device)
        angles = torch.from_numpy(angles).to(device)
        scales = torch.from_numpy(scales).to(device)

        trans_params = list(zip(shifts, angles, scales))
    else:
        trans_params = [None] * frame_count

    # distort image with digress
    first_layer_channels = Gs.synthesis.input.channels if hasattr(Gs.synthesis, 'input') else 0
    first_layer_size = Gs.synthesis.input.size if hasattr(Gs.synthesis, 'input') else 0
    if isinstance(first_layer_size, (list, tuple, np.ndarray)):
        h, w = first_layer_size[0], first_layer_size[1]
    else:
        h, w = first_layer_size, first_layer_size

    shape_for_dconst = [1, first_layer_channels, h, w] if first_layer_channels > 0 else [0]

    if a.digress != 0 and first_layer_channels > 0:
        dconst_list = []
        for _ in range(n_mult):
            dc_tmp = a.digress * latent_anima(
                shape_for_dconst, a.frames, a.fstep,
                cubic=True, seed=noise_seed, verbose=False
            )
            dconst_list.append(dc_tmp)
        dconst = np.concatenate(dconst_list, axis=1)  # [frame_count, n_mult*channels, h, w]
    else:
        # 全フレームゼロ
        if first_layer_channels > 0:
            dconst = np.zeros([a.frames, 1, first_layer_channels, h, w], dtype=np.float32)
        else:
            dconst = np.zeros([a.frames, 0])

    dconst = torch.from_numpy(dconst).to(device).to(torch.float32)

    # warm up
    # (generate 1 frame to initialize CUDA, etc.)
    if custom:
        if hasattr(Gs.synthesis, 'input'):  # SG3
            _ = Gs(
                latents[0], labels[0] if label_size else None,
                lmask[0] if len(lmask) > 1 else lmask[0],
                trans_params[0],
                dconst[0] if len(dconst.shape) > 1 else None,
                noise_mode='const',
                truncation_psi=a.trunc
            )
        else:  # SG2
            _ = Gs(
                latents[0], labels[0] if label_size else None,
                lmask[0] if len(lmask) > 1 else None,
                noise_mode='const',
                truncation_psi=a.trunc
            )
    else:
        _ = Gs(
            latents[0], labels[0] if label_size else None,
            noise_mode='const',
            truncation_psi=a.trunc
        )

    out_name = make_out_name(a)

    # -------------------------------------------------------------------------
    # 画像出力モード (--image 指定時)
    # -------------------------------------------------------------------------
    if a.image:
        # out_dir 下に out_name のフォルダを作成して、その中にフレームを保存
        image_out_dir = osp.join(a.out_dir, out_name)
        os.makedirs(image_out_dir, exist_ok=True)
        print(f"Saving frames to: {image_out_dir}")

        bar = progbar(frame_count)
        for i in range(frame_count):
            latent = latents[i]  # [n_mult, z_dim]
            lbl = labels[i] if label_size else None
            lat_m = lmask[i % len(lmask)] if len(lmask) > 1 else lmask[0]
            trp = trans_params[i] if trans_params[i] else None
            dct = dconst[i] if len(dconst.shape) > 1 else None

            if custom:
                if hasattr(Gs.synthesis, 'input'):  # SG3
                    output = Gs(
                        latent, lbl, lat_m, trp, dct,
                        truncation_psi=a.trunc, noise_mode='const'
                    )
                else:  # SG2
                    output = Gs(
                        latent, lbl, lat_m,
                        truncation_psi=a.trunc, noise_mode='const'
                    )
            else:
                output = Gs(
                    latent, lbl,
                    truncation_psi=a.trunc, noise_mode='const'
                )

            # output: [n_mult, C, H, W], 今回は n_mult=1 で出力すると仮定
            output = (output.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            output = output.to(torch.uint8).cpu().numpy()

            # とりあえず先頭のみを保存
            # (n_mult>1のときは何枚か並んだ画像が合成されるので出力枚数に注意)
            img = output[0]

            filename = osp.join(image_out_dir, f"{i:06d}.png")
            imsave(filename, img, quality=95)
            bar.upd()
        print("All frames saved.")
    else:
        # ---------------------------------------------------------------------
        # 動画出力モード (デフォルト)
        # ---------------------------------------------------------------------
        print("Generating video...")

        frame_count = latents.shape[0]
        duration_sec = frame_count / a.framerate

        def make_frame(t):
            frame_idx = int(np.clip(np.round(t * a.framerate), 0, frame_count - 1))
            latent = latents[frame_idx]
            lbl = labels[frame_idx] if label_size else None
            lat_m = lmask[frame_idx % len(lmask)] if len(lmask) > 1 else lmask[0]
            trp = trans_params[frame_idx] if trans_params[frame_idx] else None
            dct = dconst[frame_idx] if len(dconst.shape) > 1 else None

            if custom:
                if hasattr(Gs.synthesis, 'input'):  # SG3
                    output = Gs(
                        latent, lbl, lat_m, trp, dct,
                        truncation_psi=a.trunc, noise_mode='const'
                    )
                else:  # SG2
                    output = Gs(
                        latent, lbl, lat_m,
                        truncation_psi=a.trunc, noise_mode='const'
                    )
            else:
                output = Gs(
                    latent, lbl,
                    truncation_psi=a.trunc, noise_mode='const'
                )

            output = (output.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)
            output = output.to(torch.uint8).cpu().numpy()
            return output[0]

        # 動画保存
        if a.prores:
            moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(
                osp.join(a.out_dir, f"{out_name}.mov"),
                fps=a.framerate,
                codec='prores',
                bitrate='16M'
            )
        else:
            moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(
                osp.join(a.out_dir, f"{out_name}.mp4"),
                fps=a.framerate,
                codec='libx264',
                bitrate='16M'
            )

    # latent の保存 (--save_lat 指定時)
    if a.save_lat:
        # latents: [frame_count, n_mult, z_dim]
        latents_for_save = latents.squeeze(1) if latents.shape[1] == 1 else latents.reshape(frame_count, -1)
        filename = '{}-{}-{}.npy'.format(basename(a.model), a.size[1], a.size[0])
        filename = osp.join(a.out_dir, filename)
        latents_for_save = latents_for_save.cpu().numpy()
        np.save(filename, latents_for_save)
        print('saved latents', latents_for_save.shape, 'to', filename)


if __name__ == '__main__':
    for i in range(a.variations):
        generate(a.noise_seed + i)
