import numpy as np
import matplotlib.pyplot as plt
import topnet
import imageio

if __name__ == '__main__':
    # Grab and normalize blobs image.
    f = -1 * np.asarray(imageio.imread('Data/blobs.png')[:256, :256, 0], dtype=np.float32)
    f -= f.min()
    f /= f.max()
    f *= 255
    # Parameters
    hom_dim = 0
    card = 10000
    kernel_size = 4
    pool_mode = 'simplex'
    eps = 50.0
    num_perturb = 100
    pers_cutoff = 50
    # Generate noisy, downsampled gradients
    f_down = topnet.spool(f, kernel_size, pool_mode)[0]
    grads = np.zeros((num_perturb, np.prod(f_down.shape)))
    for i in range(num_perturb):
        noisy_f = f + np.random.uniform(-eps, eps, f.shape)
        f_down = topnet.spool(noisy_f, kernel_size, pool_mode)[0]
        dgm, cof = topnet.compute_dgm(f_down, card, hom_dim)
        big_pers = (dgm[:, 1] - dgm[:, 0]) > pers_cutoff
        dgm, cof = dgm[big_pers, :], cof[big_pers, :]
        grad_dgm = np.zeros(dgm.shape)
        grad_dgm[:, 0], grad_dgm[:, 1] = 2 * (dgm[:, 0] - dgm[:, 1]), 2 * (dgm[:, 1] - dgm[:, 0])
        bsm, dsm = topnet.compute_dgm_grad(grad_dgm, cof, f_down)
        grads[i, :] = dsm.ravel()
    # Visualize inner products of gradients
    G = np.dot(grads, grads.T)
    plt.figure(figsize=(16, 8))
    plt.subplot(121)
    plt.imshow(G)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    s = np.diag(G)
    cs = (1 / s) / np.sum(1 / s)
    plt.subplot(122)
    plt.plot(cs)
    plt.ylim(0, .04)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
