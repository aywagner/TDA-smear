import numpy as np
import matplotlib.pyplot as plt
import imageio
import topnet

if __name__ == '__main__':
    # Grab and normalize image
    f = np.asarray(imageio.imread('Data/circle.png')[0:210, 0:210, 1], dtype=np.float32)
    f -= f.min()
    f /= f.max()
    f *= 255
    # Compute diagram
    dgm = topnet.compute_thresh_dgm(f, 10000, 1, [float('-inf'), float('inf'), 30, float('inf')])[0]
    dgm = dgm[dgm.any(axis=1)]
    grad_dgm = np.zeros(dgm.shape)
    grad_dgm[:, 0] = 1
    grad_dgm[:, 1] = -1
    # Compute critical smear
    critical_smear = topnet.compute_spawn_sw(grad_dgm, dgm, f, 10000,
                                             1, 5, 'simplex', 50, 1000, 20,
                                             pers_region=[float('-inf'), float('inf'), 30, float('inf')])
    # Superimpose image and critical smear
    plt.imshow(f / 255, cmap='gray')
    masked0 = np.ma.masked_where(critical_smear[0] * 1000 == 0, critical_smear[0] * 1000)
    masked1 = np.ma.masked_where(critical_smear[1] * 1000 == 0, critical_smear[1] * 1000)
    plt.imshow(masked0, alpha=0.8, cmap='autumn', interpolation='none')
    plt.imshow(masked1, alpha=0.8, cmap='winter', interpolation='none')
    plt.title("Visualizing Smear", fontsize=16)
    plt.show()
