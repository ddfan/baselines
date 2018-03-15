import time
import numpy as np
import tensorflow as tf

# from https://github.com/tensorflow/tensorflow/issues/675#issuecomment-336029045

def jacobian(session, y, x, points, batch_size, as_matrix=True):
    """The Jacobian matrix of `y` w.r.t. `x` at `points`

    Let f(x) be some function that has a Jacobian A at point p
    then, f(p) = y = Ap+b
    where A of shape mxn, p of shape nx1 and b of shape mx1

    Args:
        y: The output tensor
        x: The input tensor
        points: The points of linearization where it can be many points
            of shape [num_points, *self.features_shape]
        batch_size: How many rows of the Jacobian to compute at once
        as_matrix: Whether to return the Jacobian as a matrix or retain
            the shape of the input

    Returns:
        The Jacobian matrices for the given points
        of shape [num_points, *jacobian_shape]
        If `as_matrix`, jacobian_shape is [y.size, *x.shape]
        else, jacobian_shape is [y.size, x.size]
    """
    # add and/or get cached ops to the graph
    if not hasattr(session.graph, "_placeholder"):
        session.graph._placeholder = {}
    if not hasattr(session.graph, "_gradient"):
        session.graph._gradient = {}
    with session.graph.as_default():
        if y.dtype in session.graph._placeholder:
            placeholder = session.graph._placeholder[y.dtype]
        else:
            placeholder = tf.placeholder(y.dtype)
            session.graph._placeholder[y.dtype] = placeholder

        if (y, x) in session.graph._gradient:
            gradient = session.graph._gradient[(y, x)]
        else:
            gradient = tf.gradients(placeholder * y, x)[0]
            session.graph._gradient[(y, x)] = gradient

    # extract the Jacobians for all points
    jacobians_list = []
    for i in range(points.shape[0]):
        # extract the Jacobian matrix for a single point
        partials_list = []
        point = points[i:i + 1, :]
        shape = y.shape.as_list()[1:]
        repeated_point = point
        for mask in masks_batches(shape, batch_size):
            # repeat the point according to the mask's batch_size
            batch_size = mask.shape[0]
            if repeated_point.shape[0] < batch_size:
                repeated_point = np.vstack([point] * batch_size)
            if repeated_point.shape[0] > batch_size:
                repeated_point = repeated_point[:batch_size, :]
            feed = {placeholder: mask, x: repeated_point}
            partial = session.run(gradient, feed_dict=feed)
            partials_list.append(partial)
        jacobian = np.vstack(partials_list)

        # reshape it as a matrix
        if as_matrix:
            jacobian = jacobian.reshape(jacobian.shape[0], -1)

        jacobians_list.append(jacobian)

    # stack Jacobians
    jacobians = np.stack(jacobians_list)

    return jacobians


def masks_batches(shape, batch_size):
    """Batches iterator over all possible masks of the given shape

    A mask is a numpy.ndarray of shape `shape` of all zeros except
    for a single position it is one. It is useful to get those masks
    in batches instead of getting them one by one.

    Args:
        shape: The shape of each mask
        batch_size: How many masks to return in each iteration

    Returns:
        A batch of masks of shape [batch_size, *shape]
    """
    num_rows = np.prod(shape)
    if num_rows < batch_size:
        batch_size = num_rows

    eye = np.eye(batch_size)
    _mask = np.zeros((batch_size, *shape))
    mask = _mask.reshape(batch_size, -1)

    num_batches = -(-num_rows // batch_size)
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_rows)

        # check if last batch is smaller than batch size
        if end - start < batch_size:
            batch_size = end - start
            eye = np.eye(batch_size)
            _mask = np.zeros((batch_size, *shape))
            mask = _mask.reshape(batch_size, -1)

        mask[:, start:end] = eye
        yield _mask
        mask[:, start:end] = 0


if __name__ == '__main__':
    m = 10
    n = 500
    s = 20

    x = tf.placeholder(tf.float32)
    w = tf.Variable(tf.truncated_normal([n, m], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[m]))
    y = tf.matmul(x, w) + b

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    start = time.time()
    j_out = jacobian(sess, y, x, np.random.rand(s, n), m)
    w_out = sess.run(w)
    # they should be equal and error ~ < 1e-6 (single precision)
    error = np.linalg.norm(w_out.T - np.mean(j_out, axis=0))
    if error < 1e-6:
        print("Correct Jacobian!")
    else:
        print("Error was {}".format(error))
    print(str(int(time.time() - start)) + " Seconds: " + str(j_out.shape))
