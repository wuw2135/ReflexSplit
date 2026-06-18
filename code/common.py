"""
Shared utilities for the ReflexSplit Fig-1 reproduction.

The whole point of this file is to make the THREE methods (ReflexSplit, DSIT, RDNet) use
*exactly the same*, *correct* image conversion and PSNR, so the comparison is valid.

CRITICAL BUG THIS FILE FIXES  ----------------------------------------------------------
All three SIRS models in this study output images in the range [0, 1].
DSIT ships a helper `util.tensor2im` that does  (x + 1) / 2 * 255 , i.e. it assumes the
tensor is in [-1, 1] (a tanh output).  If you feed a [0, 1] tensor to it you get:

    [0, 1]  ->  [0.5, 1.0] * 255  =  [127, 255]      # washed out / almost white

and, worse, if you compute PSNR between a washed prediction and a washed ground truth, the
error magnitude is halved, the MSE drops 4x, and the PSNR is inflated by +6.02 dB.

That single mismatch made DSIT look like 29.5 dB (and "beat" ReflexSplit) when its TRUE
value is 23.4 dB.  Always use `clip255` below for every model.  Never mix in tensor2im.
---------------------------------------------------------------------------------------
"""
import os
import random
from os.path import join

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr


def seed_all(seed: int = 42) -> None:
    """Seed every RNG we touch so a run is byte-for-byte reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip255(t: torch.Tensor) -> np.ndarray:
    """Correct [0,1]-tensor -> HxWx3 float[0,255] image.  Use this for ALL three models.

    `t` is a 4-D tensor (1, 3, H, W).  We clamp to [0,1] and scale by 255 -- NO (x+1)/2 shift.
    """
    x = np.clip(t[0].detach().cpu().float().numpy(), 0.0, 1.0)
    return np.transpose(x, (1, 2, 0)) * 255.0


def psnr_rgb(pred: np.ndarray, gt: np.ndarray) -> float:
    """SIRS-standard PSNR: per-channel skimage PSNR, data_range=255, averaged over RGB,
    on the full image.  `pred`/`gt` are HxWx3 arrays in [0,255]."""
    return float(np.mean([sk_psnr(gt[..., c], pred[..., c], data_range=255) for c in range(3)]))


class SimpleAug(torch.utils.data.Dataset):
    """Light, identical augmentation for fair in-domain fine-tuning of every method:
    random `size`x`size` crop + horizontal flip only.  No rotation / scale jitter.

    Reads OpenRR-1k pairs from  <root>/blended/*.jpg  and  <root>/transmission_layer/*.jpg .
    Returns tensors in [0,1] (what all three models expect).  With num_workers=0 + seed_all
    the whole stream is deterministic.
    """

    def __init__(self, root: str, size: int = 384):
        self.b = join(root, "blended")
        self.t = join(root, "transmission_layer")
        self.fns = sorted(os.listdir(self.b))
        self.size = size

    def __len__(self) -> int:
        return len(self.fns)

    def __getitem__(self, i: int):
        fn = self.fns[i]
        s = self.size
        I = Image.open(join(self.b, fn)).convert("RGB")
        T = Image.open(join(self.t, fn)).convert("RGB")
        w, h = I.size
        if w < s or h < s:                       # guarantee crop fits
            sc = s / min(w, h) + 1e-3
            I = I.resize((int(w * sc) + 1, int(h * sc) + 1))
            T = T.resize(I.size)
            w, h = I.size
        x = random.randint(0, w - s)
        y = random.randint(0, h - s)
        I = I.crop((x, y, x + s, y + s))
        T = T.crop((x, y, x + s, y + s))
        if random.random() < 0.5:
            I = TF.hflip(I)
            T = TF.hflip(T)
        I = TF.to_tensor(I)
        T = TF.to_tensor(T)
        return {"input": I, "target_t": T, "target_r": torch.clip(I - T, 0, 1), "fn": fn, "real": True}
