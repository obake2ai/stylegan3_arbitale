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

# ----------- 追加: Colab の出力更新用 -----------
import cv2  # ローカル用 (cv2.imshow を使用するリアルタイムプレビュー向け)
import time
from IPython.display import display, clear_output
from PIL import Image

desc = "Customized StyleGAN3 on PyTorch (リアルタイムプレビュー & Colab デモ版)"
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
parser.add_argument('--affine_scale', default='1.0-1.0')
#Video Setting (通常の動画保存フラグはそのまま残す)
parser.add_argument('--framerate', default=30)
parser.add_argument('--prores', action='store_true', help='output video in ProRes format')
parser.add_argument('--variations', type=int, default=1)

# ------------- Colab 用デモフラグ -------------
parser.add_argument('--colab_demo', action='store_true', help='Colab上でサンプル動作をするモード')

def img_resize_for_cv2(img):
    """
    OpenCVウィンドウに表示するときに大きすぎる場合があるので、
    ウィンドウに収まるように必要なら縮小するための簡易関数。
    """
    max_w = 1920
    max_h = 1080
    h, w, c = img.shape
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def make_out_name(a):
    def fmt_f(v):
        return str(v).replace('.', '_')

    model_name = basename(a.model)

    out_name = f"{model_name}_seed{a.noise_seed}"

    if a.size is not None:
        out_name += f"_size{a.size[1]}x{a.size[0]}"

    out_name += f"_nXY{a.nXY}"
    out_name += f"_frames{a.frames}"
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

def checkout(output, i, out_dir):
    """
    1枚ずつファイルに保存したい場合に使う関数。
    """
    ext = 'png' if output.shape[3] == 4 else 'jpg'
    filename = osp.join(out_dir, "%06d.%s" % (i, ext))
    imsave(filename, output[0], quality=95)

def setup_generator(a, noise_seed=0):
    """
    生成に必要な前処理(ネットワーク読み込み・潜在ベクトル/パラメータ生成など)を行い、
    後段の描画ループで使うオブジェクトをまとめて返す。
    """

    torch.manual_seed(noise_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # フレーム数と補間ステップ数
    frames_str = a.frames.split('-')
    if len(frames_str) == 2:
        a.frames, a.fstep = [int(s) for s in frames_str]
    else:
        # "-f 200" のように1つだけ渡された場合の簡易処理
        a.frames = int(frames_str[0])
        a.fstep = 25  # 適当に固定

    # サイズ設定
    if a.size is not None:
        a.size = [int(s) for s in a.size.split('-')][::-1]
        if len(a.size) == 1:
            a.size = a.size * 2

    # Affineスケール
    if a.affine_scale is not None:
        a.affine_scale = [float(s) for s in a.affine_scale.split('-')][::-1]
        if len(a.affine_scale) == 1:
            a.affine_scale = a.affine_scale * 2

    # ネットワーク呼び出し用の設定
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = a.size
    Gs_kwargs.scale_type = a.scale_type

    # latentマスク設定 (今回は簡略)
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
        print(' !! Blending mask is not fully implemented in this example !!')
        lmask = [None]
        nHW = [1,1]
        n_mult = 1

    # ネットワークを読み込み
    pkl_name = osp.splitext(a.model)[0]
    if '.pkl' in a.model.lower():
        custom = False
        print(' .. Gs from pkl ..', basename(a.model))
    else:
        custom = True
        print(' .. Gs custom ..', basename(a.model))

    rot = True if ('-r-' in a.model.lower() or 'sg3r-' in a.model.lower()) else False
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)  # type: ignore

    if a.size is None:
        a.size = [Gs.img_resolution] * 2

    # 潜在ベクトル
    latents_np = latent_anima(
        (n_mult, Gs.z_dim), a.frames, a.fstep,
        cubic=a.cubic, gauss=a.gauss,
        seed=noise_seed, verbose=False
    )  # shape = [frame_count, n_mult, z_dim]
    latents = torch.from_numpy(latents_np).to(device)
    frame_count = latents.shape[0]

    # ラベル（条件付け）
    label_size = Gs.c_dim
    if label_size > 0:
        labels = torch.zeros((frame_count, n_mult, label_size), device=device)
        if a.labels is None:
            label_ids = []
            for i in range(n_mult):
                label_ids.append(random.randint(0, label_size - 1))
        else:
            label_ids = [int(x) for x in str(a.labels).split('-')]
            label_ids = label_ids[:n_mult]
        for i, l_ in enumerate(label_ids):
            labels[:, i, l_] = 1
    else:
        labels = [None]

    # SG3のアニメ用パラメータ(平行移動・回転・スケールなど)
    if hasattr(Gs.synthesis, 'input'):
        if a.anim_trans is True:
            hw_centers = [np.linspace(-1+1/n, 1-1/n, n) for n in nHW]
            yy, xx = np.meshgrid(*hw_centers)
            xscale = [s / Gs.img_resolution for s in a.size]
            hw_centers = np.dstack((yy.flatten()[:n_mult], xx.flatten()[:n_mult])) * xscale * 0.5 * a.shiftbase
            hw_scales = np.array([2. / n for n in nHW]) * a.shiftmax
            shifts_np = latent_anima(
                (n_mult, 2), a.frames, a.fstep, uniform=True,
                cubic=a.cubic, gauss=a.gauss, seed=noise_seed, verbose=False
            )  # [frame_count, n_mult, 2]
            shifts_np = hw_centers + (shifts_np - 0.5) * hw_scales
        else:
            shifts_np = np.zeros((frame_count, n_mult, 2))

        if a.anim_rot is True:
            angles_np = latent_anima(
                (n_mult, 1), a.frames, a.frames // 4, uniform=True,
                cubic=a.cubic, gauss=a.gauss, seed=noise_seed, verbose=False
            )
            angles_np = (angles_np - 0.5) * 180.
        else:
            angles_np = np.zeros((frame_count, n_mult, 1))

        scale_array = np.array([a.affine_scale[0], a.affine_scale[1]], dtype=np.float32)
        scales_np = np.tile(scale_array, (frame_count, n_mult, 1))

        shifts = torch.from_numpy(shifts_np).to(device)
        angles = torch.from_numpy(angles_np).to(device)
        scales = torch.from_numpy(scales_np).to(device)
        trans_params = list(zip(shifts, angles, scales))
    else:
        trans_params = [None] * frame_count

    # distort (digress)
    if hasattr(Gs.synthesis, 'input'):
        first_layer_channels = Gs.synthesis.input.channels
        first_layer_size = Gs.synthesis.input.size
        if isinstance(first_layer_size, (list, tuple, np.ndarray)):
            h_, w_ = first_layer_size[0], first_layer_size[1]
        else:
            h_, w_ = first_layer_size, first_layer_size
        shape_for_dconst = [1, first_layer_channels, h_, w_]
    else:
        first_layer_channels = 0
        shape_for_dconst = [1, 0, 0, 0]

    if a.digress != 0 and first_layer_channels > 0:
        dconst_list = []
        for i in range(n_mult):
            dc_tmp = a.digress * latent_anima(
                shape_for_dconst,
                a.frames, a.fstep, cubic=True, seed=noise_seed, verbose=False
            )
            dconst_list.append(dc_tmp)
        dconst_np = np.concatenate(dconst_list, axis=1)  # [frame_count, n_mult*channels, h_, w_]
    else:
        # 何もしない
        dconst_np = np.zeros([frame_count, 1, first_layer_channels, 1])  # 省略形

    dconst = torch.from_numpy(dconst_np).to(device).to(torch.float32)

    # ワームアップ
    with torch.no_grad():
        if custom:
            if hasattr(Gs.synthesis, 'input'):
                _ = Gs(latents[0], labels[0], lmask[0], trans_params[0], dconst[0], noise_mode='const')
            else:
                _ = Gs(latents[0], labels[0], lmask[0], noise_mode='const')
        else:
            _ = Gs(latents[0], labels[0], truncation_psi=a.trunc, noise_mode='const')

    # 返却
    return {
        'Gs': Gs,
        'latents': latents,
        'labels': labels,
        'lmask': lmask,
        'trans_params': trans_params,
        'dconst': dconst,
        'frame_count': frame_count,
        'n_mult': n_mult,
        'device': device,
        'custom': custom,
    }

def generate_colab_demo(a, noise_seed):
    """
    Colab上で短いループを回して画像をノートブックセルに表示し続けるサンプルモード。
    """
    context = setup_generator(a, noise_seed)

    Gs          = context['Gs']
    latents     = context['latents']
    labels      = context['labels']
    lmask       = context['lmask']
    trans_params= context['trans_params']
    dconst      = context['dconst']
    frame_count = context['frame_count']
    n_mult      = context['n_mult']
    device      = context['device']
    custom      = context['custom']

    print("=== Colab デモ開始 ===")
    print("ノートブックセルの出力が画像で上書きされ続けます。ループ終了後、最後の画像だけが残ります。")

    # ループ回数を小さくして試す（ここでは30フレーム）
    demo_frames = 30
    for i in range(demo_frames):
        idx = i % frame_count

        latent      = latents[idx]
        label       = labels[idx % len(labels)]
        latmask     = lmask[idx % len(lmask)]
        dc          = dconst[idx % len(dconst)]
        trans_param = trans_params[idx % len(trans_params)]

        with torch.no_grad():
            if custom:
                if hasattr(Gs.synthesis, 'input'):  # SG3
                    output = Gs(latent, label, latmask, trans_param, dc,
                                truncation_psi=a.trunc, noise_mode='const')
                else:
                    output = Gs(latent, label, latmask,
                                truncation_psi=a.trunc, noise_mode='const')
            else:
                output = Gs(latent, label,
                            truncation_psi=a.trunc, noise_mode='const')

        # 後処理
        output = (output.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        output_np = output[0].cpu().numpy()  # shape = [H, W, 3]

        # ノートブックセルへ表示 (前フレームを消して新しい画像を出力)
        clear_output(wait=True)
        display(Image.fromarray(output_np, 'RGB'))
        time.sleep(0.2)

    print("=== Colab デモ終了 ===")

def generate_realtime_local(a, noise_seed):
    """
    ローカルマシンで cv2.imshow によるリアルタイムプレビューを行う場合の関数。
    """
    context = setup_generator(a, noise_seed)

    Gs          = context['Gs']
    latents     = context['latents']
    labels      = context['labels']
    lmask       = context['lmask']
    trans_params= context['trans_params']
    dconst      = context['dconst']
    frame_count = context['frame_count']
    n_mult      = context['n_mult']
    device      = context['device']
    custom      = context['custom']

    print("=== Start Real-time Preview (ローカル環境専用) ===")
    print("ウィンドウが表示されます。終了する場合はウィンドウをアクティブにして 'q' キーを押してください。")

    frame_idx = 0
    while True:
        current_idx = frame_idx % frame_count

        latent      = latents[current_idx]
        label       = labels[current_idx % len(labels)]
        latmask     = lmask[current_idx % len(lmask)]
        dc          = dconst[current_idx % len(dconst)]
        trans_param = trans_params[current_idx % len(trans_params)]

        with torch.no_grad():
            if custom:
                if hasattr(Gs.synthesis, 'input'):  # SG3
                    output = Gs(latent, label, latmask, trans_param, dc,
                                truncation_psi=a.trunc, noise_mode='const')
                else:
                    output = Gs(latent, label, latmask,
                                truncation_psi=a.trunc, noise_mode='const')
            else:
                output = Gs(latent, label,
                            truncation_psi=a.trunc, noise_mode='const')

        output = (output.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        output_np = output[0].cpu().numpy()  # shape = [H, W, 3]

        # OpenCVで表示 (BGRに変換)
        preview_img = output_np[..., ::-1]
        preview_img = img_resize_for_cv2(preview_img)
        cv2.imshow("StyleGAN3 Real-time Preview", preview_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("プレビューを終了します。")
            break

        frame_idx += 1

    cv2.destroyAllWindows()


def main():
    a = parser.parse_args()

    # Colab デモモードフラグが立っているかどうかで挙動を分ける
    if a.colab_demo:
        print("Colabデモモードで起動します (cv2によるリアルタイムウィンドウは使いません)")
        for i in range(a.variations):
            generate_colab_demo(a, a.noise_seed + i)
    else:
        # 通常はローカルPCでのリアルタイムプレビューを実行
        print("ローカル環境でのリアルタイムプレビューを行います (cv2使用)")
        for i in range(a.variations):
            generate_realtime_local(a, a.noise_seed + i)

if __name__ == '__main__':
    main()
