import numpy as np



def output_transform(s):
    """
        returns output node transformation
    """
    return s
    # return np.tanh(s)


def output_transform_prime(s):
    """
        returns derivative of output node transformation
    """
    return 1
    # return np.subtract(1, np.square(s))


def theta(s):
    """
        returns transform for forward propogation
    """
    return np.tanh(s)

def theta_prime(xl):
    """
        returns inverse of transform
    """
    return np.subtract(1, np.square(xl))


def forward_propogation(data_point, num_layers, weights):
    """
        data_point is np ndarray of form [1, x1, x2], dimensions (3,1)
    """

    x = [None] * num_layers
    s = [None] * num_layers

    x[0] = data_point
    s[0] = 0

    # transform all middle layers with theta
    # add dummy 1 node to x
    for l in range(1, num_layers-1):
        s[l] = np.dot(np.transpose(weights[l]), x[l-1])
        x[l] = np.concatenate((np.ones((1, 1)), theta(s[l])))

    # transform output node with output transform
    # also don't concatenate dummy 1 node to output
    s[-1] = np.dot(np.transpose(weights[-1]), x[-2])
    x[-1] = output_transform(s[-1])

    return x


def backpropogation(data_point, num_layers, weights, x):
    # delta[0] will be empty
    delta = [None] * num_layers

    # first delta (for layer L) uses derivative of output transform, not theta
    delta[-1] = 2*(x[-1]-data_point[1])*output_transform_prime(x[-1])

    for l in range(num_layers-2, 0, -1):
        delta[l] = np.multiply(theta_prime(x[l]), weights[l+1]*delta[l+1])

    # delete the first element (placeholder)
    for l in range(num_layers-2, 0, -1):
        delta[l] = delta[l][1:]

    return delta


def neural_network(data, num_layers, weights):
    e_in = 0.0
    N = float(len(data))
    G = [None]*num_layers

    # G[0] empty
    for l in range(1, num_layers):
        G[l] = np.dot(0, weights[l])

    for data_point in data:
        x = forward_propogation(data_point[0], num_layers, weights)
        delta = backpropogation(data_point, num_layers, weights, x)

        e_in += (1/(4*N))*(x[-1]-data_point[1])**2

        # gradient stuff
        for l in range(1, num_layers):
            G_l_point = np.dot(x[l-1], np.transpose(delta[l]))
            G[l] += (1/N)*G_l_point


    print("X:")
    for x_ in x:
        print("  ", x_)

    print()
    print("Delta:")
    for d in delta:
        print("  ", d)
    print()
    print("G:")
    for g in G:
        print("  ", g)

if __name__ == '__main__':
    # one 2d point
    data = [(np.reshape(np.array([1, 1, 1]), (3,1)), 1)]

    
    # one hidden layer
    num_layers = 3
    # two nodes in hidden layer
    M = 2
    
    # 2d np array
    weights = [None]*num_layers
    weights[1] = np.ndarray((3, 2))
    weights[1].fill(0.2501)
    weights[2] = np.ndarray((3, 1))
    weights[2].fill(0.2499)

    neural_network(data, num_layers, weights)



