import numpy as np
import matplotlib.pyplot as plt
import topnetv2
import imageio
import timeit
import tensorflow as tf


def TopBackpropSpeedtest(f, TopLoss, a, lr, seconds):
    # Initialize TensorFlow
    tf.reset_default_graph()
    # Create input variable initialized with image values
    x = tf.get_variable("X", initializer=np.array(f), trainable=True)
    # Compute persistence
    bad_top = TopLoss(x)
    # Compute loss
    loss = a * bad_top + (1 - a) * tf.losses.mean_squared_error(f, x)
    # Optimization
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    # Train it
    train = opt.minimize(loss)
    # Training!
    init = tf.global_variables_initializer()
    start_time = timeit.default_timer()
    times, losses = [], []
    with tf.Session() as sess:
        sess.run(init)
        while True:
            losses.append(sess.run([train, loss])[1])
            times.append(timeit.default_timer() - start_time)
            if times[-1] > seconds:
                break
        pred = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "X")[0])
        return {"pred": pred, "times": times, "losses": losses}


if __name__ == '__main__':
    # Grab and normalize blobs image.
    f = -1 * np.asarray(imageio.imread('Data/blobs.png')[:256, :256, 0], dtype=np.float32)
    f -= f.min()
    f /= f.max()
    f *= 255
    # Create random image
    f_rand = np.random.randint(0, 255, (256, 256)).astype('float32')
    # Parameters
    hom_dim = 0
    card = 10000
    bad_pers = [float('-inf'), float('inf'), 50, float('inf')]
    lr = 5e-2
    a = (1 - 1 / np.prod(f.shape))
    seconds = 600
    kernel_size = 4
    pool_mode = 'simplex'


    def update_func(grad_dgm, dgm, cof, x):
        bsm, dsm = topnetv2.compute_dgm_grad(grad_dgm, cof, x)
        return dsm


    def VanillaTopLoss(x):
        dgm = topnetv2.Cubical(x, card, hom_dim, update_func)[0]
        bad_top = topnetv2.SqPersInRegion(dgm, bad_pers)
        return bad_top


    vanilla_L2 = TopBackpropSpeedtest(f, VanillaTopLoss, a, lr, seconds)
    rand_vanilla_L2 = TopBackpropSpeedtest(f_rand, VanillaTopLoss, a, lr, seconds)


    def VanillaTopLoss(x):
        dgm = topnetv2.Cubical(x, card, hom_dim, update_func)[0]
        bad_top = topnetv2.AbsPersInRegion(dgm, bad_pers)
        return bad_top


    vanilla_L1 = TopBackpropSpeedtest(f, VanillaTopLoss, a, lr, seconds)
    rand_vanilla_L1 = TopBackpropSpeedtest(f_rand, VanillaTopLoss, a, lr, seconds)


    def SpawnTopLoss(x):
        x_noisy = topnetv2.UniformNoise(x, 50)
        x_down = topnetv2.Spool(x_noisy, kernel_size, pool_mode)[0]
        dgm = topnetv2.Cubical(x_down, card, hom_dim, update_func)[0]
        bad_top = topnetv2.AbsPersInRegion(dgm, bad_pers)
        return bad_top


    spawn_noise = TopBackpropSpeedtest(f, SpawnTopLoss, a, lr, seconds)
    rand_spawn_noise = TopBackpropSpeedtest(f_rand, SpawnTopLoss, a, lr, seconds)


    def SpawnTopLoss(x):
        x_down = topnetv2.Spool(x, kernel_size, pool_mode)[0]
        dgm = topnetv2.Cubical(x_down, card, hom_dim, update_func)[0]
        bad_top = topnetv2.AbsPersInRegion(dgm, bad_pers)
        return bad_top


    spawn_no_noise = TopBackpropSpeedtest(f, SpawnTopLoss, a, lr, seconds)
    rand_spawn_no_noise = TopBackpropSpeedtest(f_rand, SpawnTopLoss, a, lr, seconds)

    plt.figure(figsize=(10, 10))
    plt.plot(spawn_noise["times"],
             (spawn_noise["losses"][0] - spawn_noise["losses"]) / spawn_noise["losses"][0])
    plt.plot(spawn_no_noise["times"],
             (spawn_no_noise["losses"][0] - spawn_no_noise["losses"]) / spawn_no_noise["losses"][0])
    plt.plot(vanilla_L2["times"],
             (vanilla_L2["losses"][0] - vanilla_L2["losses"]) / vanilla_L2["losses"][0])
    plt.plot(vanilla_L1["times"],
             (vanilla_L1["losses"][0] - vanilla_L1["losses"]) / vanilla_L1["losses"][0])
    plt.legend(["STUMP w/ noise", "STUMP w/o noise", "Vanilla W2", "Vanilla W1"], fontsize=18)
    plt.xlabel('Seconds', fontsize=20)
    plt.ylabel('Percentage Reduction of Starting Loss', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.plot(rand_spawn_noise["times"],
             (rand_spawn_noise["losses"][0] - rand_spawn_noise["losses"]) / rand_spawn_noise["losses"][0])
    plt.plot(rand_spawn_no_noise["times"],
             (rand_spawn_no_noise["losses"][0] - rand_spawn_no_noise["losses"]) / rand_spawn_no_noise["losses"][0])
    plt.plot(rand_vanilla_L2["times"],
             (rand_vanilla_L2["losses"][0] - rand_vanilla_L2["losses"]) / rand_vanilla_L2["losses"][0])
    plt.plot(rand_vanilla_L1["times"],
             (rand_vanilla_L1["losses"][0] - rand_vanilla_L1["losses"]) / rand_vanilla_L1["losses"][0])
    plt.legend(["STUMP w/ noise", "STUMP w/o noise", "Vanilla W2", "Vanilla W1"], fontsize=18)
    plt.xlabel('Seconds', fontsize=20)
    plt.ylabel('Percentage Reduction of Starting Loss', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
