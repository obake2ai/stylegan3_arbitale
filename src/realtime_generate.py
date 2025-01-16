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

# Colab or local preview
import cv2
import time
import sys
from IPython.display import display, clear_output
from PIL import Image

torch.backends.cudnn.benchmark = True

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
parser.add_argument('-f', '--frames', default='200-25', help='(未使用) total frames to generate, length of interpolation step')
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

# Colab 用デモフラグ
parser.add_argument('--colab_demo', action='store_true', help='Colab上でサンプル動作をするモード')

# 新規追加: 無限リアルタイム生成の方式を指定
parser.add_argument('--method', default='smooth', choices=['smooth', 'random_walk'],
                    help='smooth: latent_animaを使ったなめらかな無限補間, random_walk: 毎フレーム少し乱数を足す。')

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
    if a.affine_scale is not None and a.affine_scale != [1.0, 1.0]:
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

# -------------------------------------------------------
# ▼ もともとの latent_anima 関数 (util.utilgan 側) を再掲
#   今回は外部から参照するだけなのでコピーは不要ですが、
#   動作イメージのため、ユーザ要望に沿って貼り付け。
#
# def latent_anima(shape, frames, transit, key_latents=None, somehot=None, smooth=0.5,
#                  uniform=False, cubic=False, gauss=False, seed=None, start_lat=None,
#                  loop=True, verbose=True):
#     ...
#     return latents
# -------------------------------------------------------


# =============================================
# 1) スムース (latent_anima) を無限に返すジェネレータ
# =============================================
def infinite_latent_smooth(z_dim, device, cubic=False, gauss=False, seed=None,
                           chunk_size=30, uniform=False):
    """
    latent_anima を使って「2つの潜在ベクトル間を補間するフレームをchunk_size生成」→
    次の区間へ移るときに新しいランダム潜在ベクトルを用意→… と繰り返し、
    無限に潜在ベクトルをyieldするジェネレータ。
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    lat1 = rng.randn(z_dim)
    while True:
        lat2 = rng.randn(z_dim)
        key_latents = np.stack([lat1, lat2], axis=0)  # (2, z_dim)

        # latent_anima で chunk_size 個分の補間を生成
        # transit=chunk_size, frames=chunk_size, loop=False → 1区間分を単純slerp or cubic
        latents_np = latent_anima(
            shape=(z_dim,),
            frames=chunk_size,
            transit=chunk_size,
            key_latents=key_latents,
            somehot=None,
            smooth=0.5,
            uniform=uniform,  # Trueにすると lerp, Falseにすると slerp
            cubic=cubic,
            gauss=gauss,
            seed=None,
            start_lat=None,
            loop=False,
            verbose=False
        )  # shape=(chunk_size, z_dim)

        # 今区間で生成されたフレームを1つずつyield
        for i in range(len(latents_np)):
            yield torch.from_numpy(latents_np[i]).unsqueeze(0).to(device)  # (1, z_dim)
        # 次のループでは lat2 から 新しいlat3 へ補間
        lat1 = lat2


# =============================================
# 2) ランダムウォークで無限に返すジェネレータ
# =============================================
def infinite_latent_random_walk(z_dim, device, seed=None, step_size=0.02):
    """
    毎フレーム、前回の潜在ベクトルに少しだけ乱数を加えて更新する。
    ピクピク動きやすいが簡単。
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    z_prev = rng.randn(z_dim)  # 初期
    while True:
        # 乱数を加えて更新 (ランダムウォーク)
        z_prev = z_prev + rng.randn(z_dim) * step_size
        yield torch.from_numpy(z_prev).unsqueeze(0).to(device)


# =============================================
# 3) 実際にリアルタイムプレビューする関数
# =============================================
def generate_realtime_local(a, noise_seed):
    """
    こちらが無限リアルタイム生成 + OpenCV表示を行う本体。
    --method smooth か --method random_walk でモードを切り替える。
    """

    torch.manual_seed(noise_seed)
    np.random.seed(noise_seed)
    random.seed(noise_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out_dir, exist_ok=True)

    # ネットワーク読み込み --------------------------------
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.verbose = a.verbose
    Gs_kwargs.size = None
    Gs_kwargs.scale_type = a.scale_type

    pkl_name = osp.splitext(a.model)[0]
    if '.pkl' in a.model.lower():
        custom = False
        print(' .. Gs from pkl ..', basename(a.model))
    else:
        custom = True
        print(' .. Gs custom ..', basename(a.model))

    rot = True if ('-r-' in a.model.lower() or 'sg3r-' in a.model.lower()) else False
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f, custom=custom, rot=rot, **Gs_kwargs)['G_ema'].to(device)

    z_dim = Gs.z_dim
    c_dim = Gs.c_dim

    # ラベル（条件付け）
    if c_dim > 0 and a.labels is not None:
        label = torch.zeros([1, c_dim], device=device)
        label_idx = min(int(a.labels), c_dim - 1)
        label[0, label_idx] = 1
    else:
        label = None

    # SG3 の distortion 用 const
    if custom and hasattr(Gs.synthesis, 'input'):
        first_layer_channels = Gs.synthesis.input.channels
        if isinstance(Gs.synthesis.input.size, int):
            h_, w_ = Gs.synthesis.input.size, Gs.synthesis.input.size
        else:
            h_, w_ = Gs.synthesis.input.size[0], Gs.synthesis.input.size[1]
        dconst_current = torch.zeros([1, first_layer_channels, h_, w_], device=device)
    else:
        dconst_current = None

    # 初回ウォームアップ推論
    with torch.no_grad():
        if custom and hasattr(Gs.synthesis, 'input'):
            _ = Gs(torch.randn([1, z_dim], device=device), label, None,
                   (torch.zeros([1,2], device=device),
                    torch.zeros([1,1], device=device),
                    torch.ones ([1,2], device=device)),
                   dconst_current, noise_mode='const')
        else:
            _ = Gs(torch.randn([1, z_dim], device=device), label,
                   truncation_psi=a.trunc, noise_mode='const')

    # どちらのモードで潜在ベクトルを無限生成するか切り替え
    if a.method == 'random_walk':
        print("=== Real-time Preview (random_walk mode) ===")
        latent_gen = infinite_latent_random_walk(
            z_dim=z_dim, device=device, seed=noise_seed, step_size=0.02
        )
    else:
        print("=== Real-time Preview (smooth latent_anima mode) ===")
        latent_gen = infinite_latent_smooth(
            z_dim=z_dim, device=device,
            cubic=a.cubic, gauss=a.gauss,
            seed=noise_seed,
            chunk_size=60,   # 1区間に60フレームで補間 (お好みで)
            uniform=False    # False→slerp, True→lerp
        )

    print("ウィンドウが表示されます。終了する場合は 'q' キーを押してください。")

    # FPS計測用
    fps_count = 0
    t0 = time.time()

    # メインループ (無限)
    frame_idx = 0
    while True:
        # 1フレームぶんの latent をジェネレータから取得
        z_current = next(latent_gen)

        # SG3パラメータ (今回は簡易: 平行移動/回転/スケールは使わず固定する)
        # anim_trans, anim_rot を本格的に使うなら、ここでも同様に補間 or random_walk すればOK
        if custom and hasattr(Gs.synthesis, 'input'):
            trans_param = (
                torch.zeros([1,2], device=device),   # shift
                torch.zeros([1,1], device=device),   # angle
                torch.ones([1,2],  device=device)    # scale
            )
        else:
            trans_param = None

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            if custom and hasattr(Gs.synthesis, 'input'):
                output = Gs(z_current, label, None,
                            trans_param, dconst_current,
                            truncation_psi=a.trunc, noise_mode='const')
            else:
                output = Gs(z_current, label,
                            truncation_psi=a.trunc, noise_mode='const')

        # 後処理
        output = (output.permute(0,2,3,1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out_np = output[0].cpu().numpy()
        out_cv = out_np[..., ::-1]  # BGR

        # OpenCVで表示
        out_cv = img_resize_for_cv2(out_cv)
        cv2.imshow("StyleGAN3 Real-time Preview", out_cv)

        # FPS計測
        fps_count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:  # 1秒ごとに更新
            fps = fps_count / elapsed
            # 同じ行に上書き表示 ("\r" で行頭に戻り、print(..., end='')
            print(f"\r{fps:.2f} fps", end='')
            sys.stdout.flush()
            t0 = time.time()
            fps_count = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n終了します。")
            break

        frame_idx += 1

    cv2.destroyAllWindows()

def generate_colab_demo(a, noise_seed):
    """
    Colab上で短いループを回して画像をノートブックセルに表示し続けるサンプルモード。
    こちらはオフライン(バッチ)想定でフレーム数決め打ち、ループ再生する。
    """
    print("=== Colab デモ開始 ===")
    print("(こちらは従来のフレーム固定デモです)")

    # 必要最低限だけネットワーク読み込み (簡略)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pkl_name = osp.splitext(a.model)[0]
    with dnnlib.util.open_url(pkl_name + '.pkl') as f:
        Gs = legacy.load_network_pkl(f)['G_ema'].to(device)

    frames = 30
    for i in range(frames):
        z = torch.randn([1, Gs.z_dim], device=device)
        with torch.no_grad():
            output = Gs(z, None, truncation_psi=a.trunc, noise_mode='const')
        output = (output.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)
        out_np = output[0].cpu().numpy()
        clear_output(wait=True)
        display(Image.fromarray(out_np, 'RGB'))
        time.sleep(0.2)

    print("=== Colab デモ終了 ===")

def main():
    a = parser.parse_args()

    if a.colab_demo:
        print("Colabデモモードで起動します (cv2によるリアルタイムウィンドウは使いません)")
        for i in range(a.variations):
            generate_colab_demo(a, a.noise_seed + i)
    else:
        print("ローカル環境でのリアルタイムプレビューを行います (cv2使用)")
        # 無限生成プレビュー (smooth or random_walk)
        for i in range(a.variations):
            generate_realtime_local(a, a.noise_seed + i)

if __name__ == '__main__':
    main()
