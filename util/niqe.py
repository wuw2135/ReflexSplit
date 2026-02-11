import numpy as np
from os.path import join
from PIL import Image
from scipy import special
from scipy.linalg import pinv

# Precompute gamma ranges
gamma_range = np.arange(0.2, 10, 0.001)
a = special.gamma(2.0 / gamma_range)
a *= a
b = special.gamma(1.0 / gamma_range)
c = special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

# AGGD features
def aggd_features(imdata):
    imdata = imdata.flatten()
    left_data = imdata[imdata < 0]
    right_data = imdata[imdata >= 0]
    left_mean_sqrt = np.sqrt(np.mean(left_data**2)) if len(left_data) > 0 else 0
    right_mean_sqrt = np.sqrt(np.mean(right_data**2)) if len(right_data) > 0 else 0
    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf
    imdata_squared = imdata**2
    r_hat = (np.mean(np.abs(imdata))**2 / np.mean(imdata_squared)) if np.mean(imdata_squared) != 0 else np.inf
    rhat_norm = r_hat * (((gamma_hat**3 + 1) * (gamma_hat + 1)) / ((gamma_hat**2 + 1)**2))
    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]
    gam1 = special.gamma(1.0 / alpha)
    gam2 = special.gamma(2.0 / alpha)
    gam3 = special.gamma(3.0 / alpha)
    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt
    N = (br - bl) * (gam2 / gam1)
    return alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt

# Paired products for MSCN
def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)
    return shift1*new_im, shift2*new_im, shift3*new_im, shift4*new_im

# Gaussian window generator
def gen_gauss_window(lw, sigma):
    lw = int(lw)
    weights = np.zeros(2*lw+1, dtype=np.float32)
    weights[lw] = 1.0
    sum_w = 1.0
    sigma2 = sigma**2
    for i in range(1, lw+1):
        w = np.exp(-0.5 * (i*i) / sigma2)
        weights[lw+i] = w
        weights[lw-i] = w
        sum_w += 2*w
    return weights / sum_w

# Compute MSCN transform
def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(image.shape) == 2
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = image.astype(np.float32)
    # Correlations along rows and columns
    from scipy.ndimage import correlate1d
    correlate1d(image, avg_window, axis=0, output=mu_image, mode=extend_mode)
    correlate1d(mu_image, avg_window, axis=1, output=mu_image, mode=extend_mode)
    correlate1d(image**2, avg_window, axis=0, output=var_image, mode=extend_mode)
    correlate1d(var_image, avg_window, axis=1, output=var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    mscn_coeffs = (image - mu_image) / (var_image + C)
    return mscn_coeffs, var_image, mu_image

# Extract NIQE features from a patch
def _niqe_extract_subband_feats(mscncoefs):
    alpha_m, N, bl, br, _, _ = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    feats = [alpha_m, (bl+br)/2.0]
    for pps in [pps1, pps2, pps3, pps4]:
        alpha, N, bl, br, _, _ = aggd_features(pps)
        feats.extend([alpha, N, bl, br])
    return np.array(feats)

# Extract features on patches
def extract_on_patches(img, patch_size):
    h, w = img.shape
    patches = [img[j:j+patch_size, i:i+patch_size]
               for j in range(0, h-patch_size+1, patch_size)
               for i in range(0, w-patch_size+1, patch_size)]
    patch_features = [ _niqe_extract_subband_feats(p) for p in patches ]
    return np.array(patch_features)

# Generic patch extraction
def _get_patches_generic(img, patch_size, stride):
    h, w = img.shape
    if h < patch_size or w < patch_size:
        raise ValueError("Input image is too small")
    hoffset = h % patch_size
    woffset = w % patch_size
    if hoffset > 0: img = img[:-hoffset, :]
    if woffset > 0: img = img[:, :-woffset]
    img = img.astype(np.float32)
    img2 = np.array(Image.fromarray(img).resize((img.shape[1]//2, img.shape[0]//2), Image.Resampling.BICUBIC), dtype=np.float32)
    mscn1, _, _ = compute_image_mscn_transform(img)
    mscn2, _, _ = compute_image_mscn_transform(img2)
    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size//2)
    return np.hstack((feats_lvl1, feats_lvl2))

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, stride)

# Main NIQE function
def niqe(inputImgData, data_folder='./weights'):
    patch_size = 96

    # Load .npz files instead of .mat
    clean_params = np.load(join(data_folder, 'clean_image_parameters.npz'))
    pop_mu = clean_params['clean_mean'].ravel()
    pop_cov = clean_params['clean_cov']

    M, N = inputImgData.shape
    if M <= patch_size*2 or N <= patch_size*2:
        raise ValueError("Input image too small, requires >192x192")

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)
    X = sample_mu - pop_mu
    covmat = (pop_cov + sample_cov)/2.0
    pinvmat = pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
    return niqe_score

# Convenience wrapper for images
def calculate_niqe(image_input):
    if isinstance(image_input, np.ndarray):
        # 若不是灰階，就轉灰階
        if image_input.ndim == 3 and image_input.shape[-1] in (3, 4):
            gray_image = np.dot(image_input[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = image_input
        gray_image = gray_image.astype(np.float64)
    else:
        # 預設當作路徑字串
        gray_image = np.array(Image.open(image_input).convert('L'), dtype=np.float64)
    return niqe(gray_image)


