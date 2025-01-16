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

import queue
import threading

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

    # 初回ウォームアップ推論
    with torch.no_grad():
        if custom and hasattr(Gs.synthesis, 'input'):
            _ = Gs(torch.randn([1, z_dim], device=device), label, lmask[0],
                   (torch.zeros([1,2], device=device),
                    torch.zeros([1,1], device=device),
                    torch.ones ([1,2], device=device)),
                   dconst_current, noise_mode='const')
        else:
            _ = Gs(torch.randn([1, z_dim], device=device), label, lmask[0],
                   truncation_psi=a.trunc, noise_mode='const')

    # 1) 生成したフレームを格納するキュー。maxsizeは適当。
    frame_queue = queue.Queue(maxsize=30)

    # 2) スレッド停止用のイベント
    stop_event = threading.Event()

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

    # 3) フレーム生成(推論)を行うサブスレッドを定義
    def producer_thread():
        frame_idx_local = 0
        while not stop_event.is_set():
            # ここで latent を1個取り出して推論
            z_current = next(latent_gen)
            latmask   = lmask[frame_idx_local % len(lmask)]
            dconst_current = dconst[frame_idx % len(dconst)]

            if custom and hasattr(Gs.synthesis, 'input'):
                trans_param = (
                    torch.zeros([1,2], device=device),
                    torch.zeros([1,1], device=device),
                    torch.ones ([1,2], device=device)
                )
            else:
                trans_param = None

            with torch.no_grad():
                if custom and hasattr(Gs.synthesis, 'input'):
                    out = Gs(z_current, label, latmask,
                             trans_param, dconst_current,
                             truncation_psi=a.trunc, noise_mode='const')
                else:
                    out = Gs(z_current, label, latmask,
                             truncation_psi=a.trunc, noise_mode='const')

            out = (out.permute(0,2,3,1) * 127.5 + 128).clamp(0,255).to(torch.uint8)
            out_np = out[0].cpu().numpy()[..., ::-1]  # BGR

            # キューが満杯ならブロックし、空きが出るまで待機
            frame_queue.put(out_np)
            frame_idx_local += 1

    # 4) スレッドを起動
    thread_prod = threading.Thread(target=producer_thread, daemon=True)
    thread_prod.start()

    print("ウィンドウが表示されます。終了する場合は 'q' キーを押してください。")

    # FPS計測用
    fps_count = 0
    t0 = time.time()

    # メインループ (無限)
    frame_idx = 0
    while True:
        # 5) バッファから1枚取り出して描画
        #    producer側でまだフレームが用意できていないときはブロック (待機)
        out_cv = frame_queue.get()

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
            stop_event.set()
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
    if a.size is not None:
        a.size = [int(s) for s in a.size.split('-')][::-1]
        if len(a.size) == 1: a.size = a.size * 2
    if a.affine_scale is not None: a.affine_scale = [float(s) for s in a.affine_scale.split('-')][::-1]
    [a.frames, a.fstep] = [int(s) for s in a.frames.split('-')]

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
