import numpy as np
import matplotlib.pyplot as plt
import topnet
import imageio

if __name__ == '__main__':
    # Grab and normalize circle image.
    f = np.asarray(imageio.imread('Data/circle.png')[0:210, 0:210, 1], dtype=np.float32)
    f -= f.min()
    f /= f.max()
    f *= 255
    plt.figure()
    plt.title('Original Image')
    plt.imshow(f)
    plt.show()
    # Parameters
    hom_dim = 1
    card = 10000
    good_pers = [float('-inf'), float('inf'), 50, float('inf')]
    lr = 5e-2
    a = (1 - 1 / np.prod(f.shape))
    steps = 10000
    kernel_size = 5
    eps = 100
    pool_mode = 'simplex'


    def update_func(grad_dgm, dgm, cof, x):
        bsm, dsm = topnet.compute_dgm_grad(grad_dgm, cof, x)
        return bsm


    def SpawnTopLoss(x):
        x_noisy = topnet.UniformNoise(x, eps)
        x_down = topnet.Spool(x_noisy, kernel_size, pool_mode)[0]
        dgm = topnet.Cubical(x_down, card, hom_dim, update_func)[0]
        good_top = topnet.AbsPersInRegion(dgm, good_pers)
        return -good_top


    spawn_optima = topnet.TopBackprop(f, SpawnTopLoss, a, lr, steps)
    plt.figure()
    plt.title('STUMP Optima')
    plt.imshow(spawn_optima)
    plt.show()


    def VanillaTopLoss(x):
        dgm = topnet.Cubical(x, card, hom_dim, update_func)[0]
        good_top = topnet.SqPersInRegion(dgm, good_pers)
        return -good_top


    vanilla_optima = topnet.TopBackprop(f, VanillaTopLoss, a, lr, steps)
    plt.figure()
    plt.title('Vanilla Optima')
    plt.imshow(vanilla_optima)
    plt.show()

    noise_eps = [0, 5, 10, 15, 20, 25]
    num_perturbations = 50
    for p in range(1, 3):
        van_test_avg, van_test_min, _ = topnet.robustness_test(vanilla_optima,
                                                               noise_eps,
                                                               num_perturbations,
                                                               [float('-inf'), float('inf'), 50, float('inf')],
                                                               p=p,
                                                               hom_dim=1)
        spa_test_avg, spa_test_min, _ = topnet.robustness_test(spawn_optima,
                                                               noise_eps,
                                                               num_perturbations,
                                                               [float('-inf'), float('inf'), 50, float('inf')],
                                                               p=p,
                                                               hom_dim=1)
        plt.figure()
        plt.plot(noise_eps, van_test_avg, 'r-')
        plt.plot(noise_eps, van_test_min, 'r--')
        plt.plot(noise_eps, spa_test_avg, 'b-')
        plt.plot(noise_eps, spa_test_min, 'b--')
        plt.title("W" + str(p) + " Total H1 Persistence in Region")
        plt.xlabel("Noise Level")
        plt.show()
