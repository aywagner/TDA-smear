import numpy as np
import gudhi as gd
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf
from tensorflow.python.framework import ops
import timeit


def compute_dgm(f, card, hom_dim):
    """
    Computes the persistence diagram of an image.
    :param f: image
    :param card: maximum number of bars kept
    :param hom_dim: dimension of homology
    :return: persistence diagram, critical pixels
    """
    dgm = np.zeros([card, 2], dtype=np.float32)
    cof = np.zeros([card, 2], dtype=np.int32)
    cc = gd.CubicalComplex(dimensions=f.shape, top_dimensional_cells=f.ravel())
    cc.compute_persistence()
    # Return zero arrays if no finite bars
    num_bars = len(cc.persistence_intervals_in_dimension(hom_dim))
    if ((hom_dim == 0) and (num_bars == 1)) or ((hom_dim > 0) and (num_bars == 0)):
        return dgm, cof
    # These are all the critical pixels
    all_cof = cc.cofaces_of_persistence_pairs()[0][hom_dim]
    # Generate the persistence diagram
    birth_times, death_times = f.flat[all_cof[:, 0]], f.flat[all_cof[:, 1]]
    # Return at most param:card bars
    min_card = min(len(birth_times), card)
    dgm[:min_card, 0], dgm[:min_card, 1] = birth_times[:min_card], death_times[:min_card]
    cof[:min_card, :] = all_cof[:min_card, :]
    return dgm, cof


def compute_dgm_grad(grad_dgm, cof, f):
    """
    Uses grad_dgm to compute birth/death critical pixels
    :param grad_dgm: gradient wrt dgm
    :param cof: critical pixels
    :param f: input image
    :return: gradient of births/deaths wrt f
    """
    grad_f_births = np.zeros(f.shape, dtype=np.float32)
    grad_f_deaths = np.zeros(f.shape, dtype=np.float32)
    # Identify which rows correspond to a persistence dot.
    is_nonzero = cof.any(axis=1)
    if not np.any(is_nonzero):
        return grad_f_births, grad_f_deaths
    # Filter by relevant rows
    cof_nonzero = cof[is_nonzero, :]
    grad_dgm_nonzero = grad_dgm[is_nonzero, :]
    # Add gradient at appropriate places.
    np.add.at(grad_f_births.ravel(), cof_nonzero[:, 0].ravel(), grad_dgm_nonzero[:, 0].ravel())
    np.add.at(grad_f_deaths.ravel(), cof_nonzero[:, 1].ravel(), grad_dgm_nonzero[:, 1].ravel())
    return grad_f_births, grad_f_deaths


def compute_thresh_dgm(f, card, hom_dim, pers_region=None):
    """
    Computes thresholded persistent homology of an image.
    :param f: input image
    :param card: max cardinality of persistence diagram
    :param hom_dim: degree of homology
    :param pers_region: np.array([birth_low, birth_high, lifetime_low, lifetime_high])
    :return: persistence diagram and associated critical pixels
    """
    dgm = np.zeros([card, 2], dtype=np.float32)
    cof = np.zeros([card, 2], dtype=np.int32)
    cc = gd.CubicalComplex(dimensions=f.shape, top_dimensional_cells=f.ravel())
    cc.compute_persistence()
    # Return zero arrays if no finite bars
    num_bars = len(cc.persistence_intervals_in_dimension(hom_dim))
    if ((hom_dim == 0) and (num_bars == 1)) or ((hom_dim > 0) and (num_bars == 0)):
        return dgm, cof
    # These are all the critical pixels
    all_cof = cc.cofaces_of_persistence_pairs()[0][hom_dim]
    # Generate the persistence diagram
    birth_times, death_times = f.flat[all_cof[:, 0]], f.flat[all_cof[:, 1]]
    # Threshold by persistence region if one was provided
    if pers_region is not None:
        lifetimes = death_times - birth_times
        rel_ind = (pers_region[0] < birth_times) & (birth_times < pers_region[1]) & \
                  (pers_region[2] < lifetimes) & (lifetimes < pers_region[3])
        birth_times, death_times, all_cof = birth_times[rel_ind], death_times[rel_ind], all_cof[rel_ind, :]
    min_card = min(len(birth_times), card)
    dgm[:min_card, 0], dgm[:min_card, 1] = birth_times[:min_card], death_times[:min_card]
    cof[:min_card, :] = all_cof[:min_card, :]
    return dgm, cof


def compute_spawn_sw(grad_dgm, dgm, f, card,
                     hom_dim, kernel_size, pool_mode, noise, samples, M,
                     pers_region=None):
    bsm = np.zeros(f.shape, dtype='float32')
    dsm = np.zeros(f.shape, dtype='float32')
    # Find nonzero rows of dgm
    dgm_up_nonzero = dgm.any(axis=1)
    if not np.any(dgm_up_nonzero):
        return bsm, dsm
    dgm_up = dgm[dgm_up_nonzero, :]
    grad_dgm_up = grad_dgm[dgm_up_nonzero, :]
    # Project nonzero rows of dgm to diagonal
    dgm_up_proj = np.column_stack(((dgm_up[:, 0] + dgm_up[:, 1]) / 2, (dgm_up[:, 0] + dgm_up[:, 1]) / 2))
    # For each random sample, compute fuzzy sliced-Wasserstein pairing
    for t in range(samples):
        g = f + np.random.uniform(-noise, noise, size=f.shape)
        x_down, switch = spool(g, kernel_size, pool_mode)
        # Compute persistence diagram and critical pixels.
        dgm_down, cof_down = compute_thresh_dgm(x_down, card, hom_dim, pers_region)
        bsm_down, dsm_down = np.zeros(x_down.shape), np.zeros(x_down.shape)  # Initialize low-res smears.
        # Get nonzero rows of dgm_down
        dgm_down_nonzero = dgm_down.any(axis=1)
        if not np.any(dgm_down_nonzero):  # Skip iteration if downsampled image has no persistent homology.
            continue
        dgm_down = dgm_down[dgm_down_nonzero, :]
        cof_down = cof_down[dgm_down_nonzero, :]
        # Project nonzero rows of downsampled dgm onto diagonal
        dgm_down_proj = np.column_stack(((dgm_down[:, 0] + dgm_down[:, 1]) / 2, (dgm_down[:, 0] + dgm_down[:, 1]) / 2))
        theta = -np.pi / 2
        for i in range(M):
            theta_vec = np.array([np.cos(theta), np.sin(theta)])
            # Symmetrize the pair dgm_up and dgm_down
            V1 = np.concatenate([np.dot(dgm_up, theta_vec), np.dot(dgm_down_proj, theta_vec)])
            V2 = np.concatenate([np.dot(dgm_down, theta_vec), np.dot(dgm_up_proj, theta_vec)])
            V1_sort = V1.argsort()
            V2_sort = V2.argsort()
            for j in range(len(V1)):
                dot1 = V1_sort[j]
                dot2 = V2_sort[j]
                # Check if pair happened between non-diagonal points
                if (dot1 < dgm_up.shape[0]) and (dot2 < dgm_down.shape[0]):
                    bsm_down.ravel()[cof_down[dot2, 0]] += (grad_dgm_up[dot1, 0] / M)
                    dsm_down.ravel()[cof_down[dot2, 1]] += (grad_dgm_up[dot1, 1] / M)
            theta += np.pi / M
        bsm += unspool(bsm_down, kernel_size, switch)
        dsm += unspool(dsm_down, kernel_size, switch)
    bsm, dsm = bsm / samples, dsm / samples
    return bsm, dsm


def robustness_test(f, eps, n, pers_region, p, hom_dim):
    num_eps = len(eps)
    pers_avgs = np.zeros(num_eps)
    pers_mins = np.zeros(num_eps)
    pers_maxs = np.zeros(num_eps)
    for t in range(num_eps):
        S = np.zeros(n)
        for i in range(n):
            g = f + np.random.uniform(low=-eps[t], high=eps[t], size=np.shape(f))
            g = np.clip(g, 0, 255)
            dgm = compute_dgm(g, 10000, hom_dim)[0]
            lifetimes = dgm[:, 1] - dgm[:, 0]
            idx = (pers_region[0] < dgm[:, 0]) & (dgm[:, 0] < pers_region[1]) & \
                  (pers_region[2] < lifetimes) & (lifetimes < pers_region[3])
            S[i] = np.linalg.norm(lifetimes[idx], p)
        pers_avgs[t] = np.average(S)
        pers_mins[t] = np.min(S)
        pers_maxs[t] = np.max(S)
    return pers_avgs, pers_mins, pers_maxs


def spool(f, kernel_size, pool_mode):
    """
    Stochastically pools an image.
    :param f: image
    :param kernel_size: integer kernel size
    :param pool_mode: 'max', 'min', 'uniform', 'simplex'
    :return: downsampled image, switch for unspooling
    """
    # Set stride to kernel size
    stride = kernel_size
    # Check that pool_mode is valid
    assert pool_mode in ['max', 'min', 'uniform', 'simplex']
    # Reshape image according to kernel size and stride
    assert ~((f.shape[0] - kernel_size) % stride or (f.shape[1] - kernel_size) % stride), \
        'Chosen kernel and stride misses some of the image.'
    downsample_shape = ((f.shape[0] - kernel_size) // stride + 1, (f.shape[1] - kernel_size) // stride + 1)
    f_window = as_strided(f,
                          shape=downsample_shape + (kernel_size, kernel_size),
                          strides=(stride * f.strides[0], stride * f.strides[1]) + f.strides)
    # Reshape f_window so each row corresponds to a window.
    f_window = f_window.reshape(-1, kernel_size ** 2)
    # Choose switch according to pool_mode
    if pool_mode == 'max':
        switch = np.zeros(f_window.shape, dtype=np.float32)
        switch[np.arange(switch.shape[0]), f_window.argmax(1)] = 1
    if pool_mode == 'min':
        switch = np.zeros(f_window.shape, dtype=np.float32)
        switch[np.arange(switch.shape[0]), f_window.argmin(1)] = 1
    if pool_mode == 'uniform':
        switch = np.zeros(f_window.shape, dtype=np.float32)
        switch[np.arange(switch.shape[0]),
               np.random.randint(0, switch.shape[1], switch.shape[0])] = 1
    if pool_mode == 'simplex':
        switch = np.random.uniform(0, 1, f_window.shape).astype('float32')
        switch = switch / switch.sum(axis=1)[:, None]
    # Get corresponding values and reshape to downsampled image size.
    f_down = np.sum(f_window * switch, axis=1).reshape(downsample_shape)
    return f_down, switch


def unspool(f, kernel_size, switch):
    """
    Deterministically un-pools an image using a switch.
    :param f: image
    :param kernel_size: kernel_size used in spool()
    :param switch: switch output by spool()
    :return: upscaled image
    """
    stride = kernel_size
    # Initialize upsampled image.
    f_up = np.zeros(((f.shape[0] - 1) * stride + kernel_size, (f.shape[1] - 1) * stride + kernel_size),
                    dtype=np.float32)
    f_window = as_strided(f_up,
                          shape=f.shape + (kernel_size, kernel_size),
                          strides=(stride * f_up.strides[0], stride * f_up.strides[1]) + f_up.strides)
    f_window[:, :, :, :] = (switch * f.ravel()[:, None]).reshape(f.shape + (kernel_size, kernel_size))
    return f_up


# py_func() and Cubical() are modified from GUDHI tutorials here: https://github.com/GUDHI/TDA-tutorial
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    Wraps Python function as TensorFlow op
    :param func: Python function
    :param inp: inputs to func
    :param Tout: types of func's outputs
    :param stateful:
    :param name:
    :param grad: TensorFlow function computing gradient of func
    :return: TensorFlow wrapper of func
    """
    rnd_name = "PyFuncGrad" + str(np.random.randint(0, 1e+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def Spool(x, kernel_size, pool_mode, name=None):
    """
    TF op that stochastically pools an image.
    :param x: image
    :param kernel_size: integer kernel size
    :param pool_mode: 'max', 'min', 'uniform', 'simplex'
    :param name:
    :return: TF operation
    """

    # Define override gradient
    def _Spool(op, grad_xdown, grad_switch):
        switch = op.outputs[1]
        grad_x = tf.py_func(lambda y, z: unspool(y, kernel_size, z),
                            [grad_xdown, switch],
                            [tf.float32])[0]
        return grad_x

    # Create the operation
    with ops.op_scope([x], name, "Spool") as name:
        return py_func(lambda y: spool(y, kernel_size, pool_mode),
                       [x],
                       [tf.float32, tf.float32],
                       name=name,
                       grad=_Spool)


def Cubical(x, card, hom_dim, update_func, name=None):
    """
    TF op that computes the persistence diagram of an image.
    :param x: image
    :param card: maximum number of bars kept
    :param hom_dim: dimension of homology
    :param update_func: update_func(grad_dgm, dgm, cof, x) gives the direction of update
    :param name:
    :return: TF operation
    """

    # Define override gradient
    def _Cubical(op, grad_dgm, grad_cof):
        dgm, cof = op.outputs[0], op.outputs[1]
        x = op.inputs[0]
        grad_x = tf.py_func(lambda a, b, c, d: update_func(a, b, c, d),
                            [grad_dgm, dgm, cof, x],
                            [tf.float32])[0]
        return grad_x

    # Create the operation
    with ops.op_scope([x], name, "Cubical") as name:
        return py_func(lambda y: compute_dgm(y, card, hom_dim),
                       [x],
                       [tf.float32, tf.int32],
                       name=name,
                       grad=_Cubical)


def UniformNoise(x, eps):
    """
    TF op that adds Uniform noise to an image.
    :param x: image
    :param eps: amount of noise
    :return: TF operation
    """
    noise = tf.random_uniform(shape=tf.shape(x), minval=-eps, maxval=eps, dtype=tf.float32)
    return x + noise


def SqPersInRegion(dgm, pers_region):
    """
    TF op that computes the sum of squared persistence in a region.
    :param dgm: persistence diagram
    :param pers_region: np.array([birth_low, birth_high, lifetime_low, lifetime_high])
    :return: TF operation
    """
    birthtimes = dgm[:, 0]
    lifetimes = dgm[:, 1] - dgm[:, 0]
    idx = tf.where((pers_region[0] < birthtimes) & (birthtimes < pers_region[1]) &
                   (pers_region[2] < lifetimes) & (lifetimes < pers_region[3]))
    return tf.reduce_sum(tf.square(tf.gather(dgm[:, 1], idx) - tf.gather(dgm[:, 0], idx)))


def AbsPersInRegion(dgm, pers_region):
    """
    TF op that computes the sum of absolute persistence in a region.
    :param dgm: persistence diagram
    :param pers_region: np.array([birth_low, birth_high, lifetime_low, lifetime_high])
    :return: TF operation
    """
    lifetimes = dgm[:, 1] - dgm[:, 0]
    idx = tf.where((pers_region[0] < dgm[:, 0]) & (dgm[:, 0] < pers_region[1]) &
                   (pers_region[2] < lifetimes) & (lifetimes < pers_region[3]) &
                   tf.logical_not(tf.equal(dgm[:, 0], dgm[:, 1])))
    return tf.reduce_sum(tf.abs(tf.gather(dgm[:, 1], idx) - tf.gather(dgm[:, 0], idx)))


def TopBackprop(f, TopLoss, a, lr, steps):
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
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(steps):
            sess.run(train)
        print('Computation Time = ' + str(timeit.default_timer() - start_time))
        return sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "X")[0])
