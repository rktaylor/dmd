from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Plotting Functions


def plot_surf(frames, t, output_file=False):
    """

    :param frames:
    :param t:
    :param output_file:
    :return:
    """
    if len(frames.shape) != 2:
        raise DMDError("Oops! `frames` must be 2d")
    len_t = len(t)
    # below we grab the non-t dimension of the frames.
    len_f = [k for k in frames.shape if k != len_t][0]
    x = np.arange(0, len_f, 1)
    x, t = np.meshgrid(x, t)
    x, t = x.transpose(), t.transpose()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, t, frames, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf)
    ax.set_xlabel("Spatial Axis")
    ax.set_ylabel("Temporal Axis")
    if not output_file:
        plt.show()
    else:
        plt.savefig(output_file)