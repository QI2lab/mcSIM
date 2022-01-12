"""Periodic & smooth image decomposition

Code included here for convenience with permission from the authors
https://github.com/jacobkimmel/ps_decomp

References
----------
Periodic Plus Smooth Image Decomposition
Moisan, L. J Math Imaging Vis (2011) 39: 161.
doi.org/10.1007/s10851-010-0227-1
"""

import numpy as np

def periodic_smooth_decomp(I: np.ndarray) -> (np.ndarray, np.ndarray):
    '''Performs periodic-smooth image decomposition

    Parameters
    ----------
    I : np.ndarray
        [M, N] image. will be coerced to a float.

    Returns
    -------
    P : np.ndarray
        [M, N] image, float. periodic portion.
    S : np.ndarray
        [M, N] image, float. smooth portion.
    '''
    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = np.fft.fftn(v)
    s = v2s(v_fft)
    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f # u = p + s
    return p, s_f

def u2v(u: np.ndarray) -> np.ndarray:
    '''Converts the image `u` into the image `v`

    Parameters
    ----------
    u : np.ndarray
        [M, N] image

    Returns
    -------
    v : np.ndarray
        [M, N] image, zeroed expect for the outermost rows and cols
    '''
    v = np.zeros(u.shape, dtype=np.float64)

    v[0, :] = np.subtract(u[-1, :], u[0,  :], dtype=np.float64)
    v[-1,:] = np.subtract(u[0,  :], u[-1, :], dtype=np.float64)

    v[:,  0] += np.subtract(u[:, -1], u[:,  0], dtype=np.float64)
    v[:, -1] += np.subtract(u[:,  0], u[:, -1], dtype=np.float64)
    return v

def v2s(v_hat: np.ndarray) -> np.ndarray:
    '''Computes the maximally smooth component of `u`, `s` from `v`


    s[q, r] = v[q, r] / (2*np.cos( (2*np.pi*q)/M )
        + 2*np.cos( (2*np.pi*r)/N ) - 4)

    Parameters
    ----------
    v_hat : np.ndarray
        [M, N] DFT of v
    '''
    M, N = v_hat.shape

    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2*np.cos( np.divide((2*np.pi*q), M) ) \
         + 2*np.cos( np.divide((2*np.pi*r), N) ) - 4)
    s = np.divide(v_hat, den, out=np.zeros_like(v_hat), where=den!=0)
    s[0, 0] = 0
    return s

if __name__ == '__main__':
    '''Plot the astronaut with and without P&S decomp'''
    import matplotlib
    matplotlib.rcParams.update({'font.size':30})
    import matplotlib.pyplot as plt
    from skimage.data import astronaut
    import skimage

    Irgb = astronaut()
    Ig = skimage.color.rgb2gray(Irgb)
    Ig = skimage.img_as_float(Ig)

    p, s = periodic_smooth_decomp(Ig)

    fig, ax = plt.subplots(3, 3, figsize=(20,20))

    labs = ['u', 'p', 's']
    for i, j in enumerate([Ig, p, s]):
        jf = np.fft.fftn(j)
        if i == 2:
            ax[0, i].imshow(j, cmap='gray')
        else:
            ax[0, i].imshow(j, cmap='gray', vmin=0., vmax=1.)
        ax[0, i].set_title(labs[i])
        ax[1, i].imshow(np.log(np.abs(np.fft.fftshift(jf)) + 1), cmap='gray')
        ax[2, i].imshow(np.angle(np.fft.fftshift(jf)), cmap='gray')
        for k in range(3):
            ax[i, k].set_xticks([])
            ax[i, k].set_yticks([])

    ax[0,0].text(1.04, 1.0, '=',
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax[0,0].transAxes)

    ax[0,1].text(1.04, 1.0, '+',
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax[0,1].transAxes)

    ax[0,0].set_ylabel('Image')
    ax[1,0].set_ylabel('log(Amplitude+1)')
    ax[2,0].set_ylabel('Phase')
    plt.tight_layout()
    plt.savefig('astronaut_psd.png')
