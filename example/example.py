import torch
import cv2
import numpy as np

import horcorr

def blockify(img, blockdim=7):
    '''
    Slow! Do not use this!
    '''
    s = blockdim // 2
    h, w, chan = img.shape
    padded = np.zeros((h + blockdim - 1, w + blockdim - 1, chan), dtype=img.dtype)
    padded[s:-s, s:-s] = img

    m = np.empty((h, w, blockdim * blockdim * chan))
    for r in range(h):
        for c in range(w):
            m[r, c] = padded[r: r + blockdim, c: c + blockdim].flatten()
    return m

def main():
    img = cv2.imread('Cambridge_Kings_Hedges_Golden_Hind_RL.jpg')
    # m = np.mean(img, axis=(0, 1))
    # std = np.std(img, axis=(0, 1))
    # nimg = (img - m) * np.reciprocal(std[None, None])
    nimg = img.astype(np.float32)

    width = nimg.shape[1] // 2
    left = nimg[:, :width]
    right = nimg[:, width:]

    def prepare(x):
        x = cv2.blur(x, (32, 32))
        x = cv2.resize(x, (96 * 2, 64 * 2))
        print(x.shape)
        return blockify(x).transpose(2, 0, 1)

    left = prepare(left)
    right = prepare(right)

    print(left.shape)

    device = torch.device('cuda:0')
    left = torch.from_numpy(left).to(device).float()[None]
    right = torch.from_numpy(right).to(device).float()[None]

    left = torch.cat((
        torch.zeros(left.shape[:3] + (horcorr.n_windows() - 1,), dtype=left.dtype, device=left.device),
        left), dim=3)

    corr = horcorr.HorCorrFunction.apply(left, right)

    print(corr.shape)
    disp = torch.argmax(corr.squeeze(), dim=0).reshape(corr.shape[2:])
    disp = disp.cpu().numpy()

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[1].imshow(disp)
    plt.savefig('example.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    with torch.no_grad():
        main()
