# Metrics/Indexes
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from functools import partial
import numpy as np
import torch
from util import niqe
import piq

metric_lpips = lpips.LPIPS().cuda()


class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-1]
        bwindex = []
        for ch in range(C):
            x = X[..., ch]
            y = Y[..., ch]
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex


cal_bwpsnr = Bandwise(partial(compare_psnr, data_range=255))
cal_bwssim = Bandwise(partial(compare_ssim, data_range=255))


def compare_ncc(x, y):
    return np.mean((x - np.mean(x)) * (y - np.mean(y))) / (np.std(x) * np.std(y))


def ssq_error(correct, estimate):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate ** 2) > 1e-5:
        alpha = np.sum(correct * estimate) / np.sum(estimate ** 2)
    else:
        alpha = 0.
    return np.sum((correct - alpha * estimate) ** 2)


def local_error(correct, estimate, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N, C = correct.shape
    ssq = total = 0.
    for c in range(C):
        for i in range(0, M - window_size + 1, window_shift):
            for j in range(0, N - window_size + 1, window_shift):
                correct_curr = correct[i:i + window_size, j:j + window_size, c]
                estimate_curr = estimate[i:i + window_size, j:j + window_size, c]
                ssq += ssq_error(correct_curr, estimate_curr)
                total += np.sum(correct_curr ** 2)
    # assert np.isnan(ssq/total)
    return ssq / total


_LPIPS_MODEL = None
_DISTS_MODEL = None

def _as_torch(img_np: np.ndarray) -> torch.Tensor:
    x = np.asarray(img_np)
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0

    if x.ndim == 2:              
        x = x[None, None, ...]   
    elif x.ndim == 3:            
        if x.shape[-1] in (1, 3):
            x = np.transpose(x, (2, 0, 1))[None, ...] 
        else:                     
            x = x[None, ...]      
    elif x.ndim == 4:            
        if x.shape[-1] in (1, 3): 
            x = np.transpose(x, (0, 3, 1, 2))         
    else:
        raise ValueError(f"Unsupported image dimensions:{x.shape}")
    return torch.from_numpy(x)

def _ensure_3ch(t: torch.Tensor) -> torch.Tensor:
    if t.shape[1] == 1:
        return t.repeat(1, 3, 1, 1)
    if t.shape[1] == 3:
        return t
    raise ValueError(f"It must have 1 or 3 channels, but received C={t.shape[1]}")


def _get_lpips_model(device):
    global _LPIPS_MODEL
    if (_LPIPS_MODEL is None) or (next(_LPIPS_MODEL.parameters()).device != device):
        _LPIPS_MODEL = piq.LPIPS(reduction='mean').to(device).eval()
    return _LPIPS_MODEL

def _get_dists_model(device):
    global _DISTS_MODEL
    if (_DISTS_MODEL is None) or (next(_DISTS_MODEL.parameters()).device != device):
        _DISTS_MODEL = piq.DISTS(reduction='mean').to(device).eval()
    return _DISTS_MODEL

def quality_assess(X, Y):
    # Y: correct; X: estimate
    psnr = np.mean(cal_bwpsnr(Y, X))
    ssim = np.mean(cal_bwssim(Y, X))
    lmse = local_error(Y, X, 20, 10)
    ncc = compare_ncc(Y, X)

    X_t = _as_torch(X)
    Y_t = _as_torch(Y)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = X_t.to(device)
    Y_t = Y_t.to(device)

    X_3 = _ensure_3ch(X_t)
    Y_3 = _ensure_3ch(Y_t)

    with torch.no_grad():
        lpips_model = _get_lpips_model(device)
        dists_model = _get_dists_model(device)

        lpips_val = lpips_model(X_3, Y_3).item()   
        dists_val = dists_model(X_3, Y_3).item()   

        niqe_val = niqe.calculate_niqe(X)

    return {
        'PSNR': psnr,
        'SSIM': ssim,
        'LMSE': lmse,
        'NCC': ncc,
        'LPIPS': lpips_val,
        'DISTS': dists_val,
        'NIQE': niqe_val,
    }


def quality_assess_per(X, Y, X_numpy, Y_numpy):
    # Y: correct; X: estimate
    psnr = np.mean(cal_bwpsnr(Y_numpy, X_numpy))
    ssim = np.mean(cal_bwssim(Y_numpy, X_numpy))
    lp = metric_lpips(Y, X)
    return {'PSNR': psnr, 'SSIM': ssim, 'LPIPS': lp}
