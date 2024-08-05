import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

mog_marker_list = ['o', 's', '^', 'x']  # Circle, square, triangle up, x
colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # bright orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # dark orange
    "#e377c2",  # magenta
    "#bcbd22",  # yellowish green
    "#dbdb8d",  # light beige
    "#17becf",  # blue-green
    "#9edae5",  # light blue
    "#ffffcc",  # light yellow
    "#c7c7c7",  # dark gray
    "#f7cac9",  # light pink
    "#fc8d59",  # salmon pink
    "#7f7f7f",  # gray (included for completeness, consider using for background)
]

class OnlineToyDataset():
    def __init__(self, data_name):
        super().__init__()
        
        self.data_name = data_name
        self.rng = np.random.RandomState(42)

    def gen_batch(self, batch_size):
        return inf_train_gen(self.data_name, self.rng, batch_size)

    def get_category_sizes(self):
        if self.data_name == "25gaussians_shape":
            return [4, 16]
        elif self.data_name == "25gaussians":
            return [25, 16]
        elif self.data_name == "25circles":
            return [25, 16]
        elif self.data_name == "rings":
            return [4, 16]
        elif self.data_name == "olympic":
            return [5, 5]
    
    def get_numerical_sizes(self):
        return 2

def inf_train_gen(data, rng=None, batch_size=4096):

    if data == "25gaussians_shape":
        
        len_colors = len(colors)
        len_markers = len(mog_marker_list)

        scale = 6.
        centers = [(x/2, y/2) for x in range(-2,3) for y in range(-2,3)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            data_point = np.zeros(4)
            idx = rng.randint(25)

            point = rng.randn(2) * 0.4
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data_point[:2] = point / 1.2
            data_point[2] = int(idx % len_markers)
            data_point[3] = int(idx % len_colors)

            dataset.append(data_point)
        dataset = np.array(dataset)
        return dataset
    
    elif data == "25gaussians":
        def assign_color_labels(x):
            coor = x.copy()
            coor[0] = (coor[0] >= 0)
            coor[1] = (coor[1] >= 0)
            return (coor[0] + 2 * coor[1])

        std = 0.2
        len_colors = len(colors)

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * std
            group_label = rng.randint(25)
            color_label = assign_color_labels(point)
            color_label = (color_label + group_label * 4) % len_colors

            data_point = np.zeros(4)
            data_point[:2] = point
            data_point[2] = group_label
            data_point[3] = color_label

            dataset.append(data_point)
        dataset = np.array(dataset)
        return dataset

    elif data == "25circles":
        def assign_color_labels(x):
            num_classes = 4
            class_width = 1.0 / num_classes
            label = x // class_width
            return label.astype(int)
        
        radius = .4

        rnd_theta = np.random.random(batch_size)
        group_label = np.random.randint(0, 25, batch_size)
        color_label = assign_color_labels(rnd_theta)
        color_label = (color_label + group_label * 4) % len(colors)

        rnd_radius = np.random.random(batch_size) * radius
        sample_thetas = 2 * np.pi * rnd_theta
        sample_x = rnd_radius * np.cos(sample_thetas)
        sample_y = rnd_radius * np.sin(sample_thetas)
        sample_x = sample_x.reshape(-1, 1)
        sample_y = sample_y.reshape(-1, 1)
        sample_group = np.concatenate((sample_x, sample_y), axis=1)

        data = np.zeros((batch_size, 4))
        data[:, 0] = sample_group[:, 0]
        data[:, 1] = sample_group[:, 1]
        data[:, 2] = group_label
        data[:, 3] = color_label
        return data

    elif data == 'checkerboard':
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return data

    elif data == "rings":
        from scipy.stats import truncnorm
        def assign_color_labels(x):
            num_classes = 8
            class_width = 1.0 / num_classes
            label = x // class_width
            return label.astype(int)

        toy_radius = 1.5
        toy_sd = 0.005
        truncnorm_rv = truncnorm(
                a=(0 - toy_radius) / toy_sd,
                b=np.inf,
                loc=toy_radius,
                scale=toy_sd,
            )
        
        rnd_theta = np.random.random(batch_size)
        groups_label = np.random.randint(0, 4, batch_size)
        color_label = assign_color_labels(rnd_theta)
        color_label = (color_label + groups_label * 4) % len(colors)

        sample_radii = truncnorm_rv.rvs(batch_size)
        # sample_radii = sample_radii * (groups_label + 1)
        sample_thetas = 2 * np.pi * rnd_theta
        sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(-1, 1)
        sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(-1, 1)

        sample_group = np.concatenate((sample_x, sample_y), axis=1)

        data = np.zeros((batch_size, 4))
        data[:, 0] = sample_group[:, 0]
        data[:, 1] = sample_group[:, 1]
        data[:, 2] = groups_label
        data[:, 3] = color_label
        return data

    elif data == "olympic":
        

        def circle_generate_sample(N, noise=0.25):
            radius = 2.0
            angle = np.random.uniform(high=2 * np.pi, size=N)
            random_noise = np.random.normal(scale=np.sqrt(0.2), size=(N, 2))
            pos = np.concatenate([np.cos(angle), np.sin(angle)])
            pos = rearrange(pos, "(b c) -> c b", b=2) * radius
            return pos + noise * random_noise

        noise = 0.005
        toy_radius = 2.5

        groups_label = np.random.randint(0, 5, batch_size)
        color_label = groups_label.copy()

        rnd_theta = np.random.random(batch_size)
        sample_thetas = 2 * np.pi * rnd_theta
        sample_x = toy_radius * np.cos(sample_thetas).reshape(-1, 1)
        sample_y = toy_radius * np.sin(sample_thetas).reshape(-1, 1)
        random_noise = np.random.normal(scale=np.sqrt(0.2), size=(batch_size, 2))
        samples = np.concatenate((sample_x, sample_y), axis=1) + noise * random_noise

        data = np.zeros((batch_size, 4))
        data[:, 0] = samples[:, 0]
        data[:, 1] = samples[:, 1]
        data[:, 2] = groups_label
        data[:, 3] = color_label
        return data

    else:
        raise NotImplementedError

def plot_25_gaussian_shape_example(data, save_path='data_samples.png'):
    left_bound = -7
    right_bound = 7
    
    for i in range(len(mog_marker_list)):
        marker = mog_marker_list[i]
        samples = data[data[:, 2] == i]
        plt.scatter(samples[:, 0], samples[:, 1], s=1, c=[colors[int(i)] for i in samples[:, 3]], marker=marker)

    plt.axis('square')
    plt.axis('off')
    # plt.title('data samples')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()

def plot_25_gaussian_example(data, save_path='data_samples.png'):
    left_bound = -7
    right_bound = 7

    scale = 6.
    centers = [(x/2, y/2) for x in range(-2,3) for y in range(-2,3)]
    centers = [(scale * x, scale * y) for x, y in centers]
    centers = np.array(centers)

    inds = data[:, 2].astype(int)
    plt.scatter(data[:, 0]+centers[inds][:,0], data[:, 1]+centers[inds][:,1], s=0.2, c=[colors[int(i)] for i in data[:, 3]], marker='o')

    plt.axis('square')
    plt.axis('off')
    # plt.title('data samples')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()

def plot_25_circles_example(data, save_path='data_samples.png'):
    left_bound = -7
    right_bound = 7

    scale = 6.
    centers = [(x/2, y/2) for x in range(-2,3) for y in range(-2,3)]
    centers = [(scale * x, scale * y) for x, y in centers]
    centers = np.array(centers)

    inds = data[:, 2].astype(int)
    plt.scatter(data[:, 0]+centers[inds][:,0], data[:, 1]+centers[inds][:,1], s=0.2, c=[colors[int(i)] for i in data[:, 3]], marker='o')

    plt.axis('square')
    plt.axis('off')
    # plt.title('data samples')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()

def plot_checkerboard_example(data, save_path='data_samples.png'):
    left_bound = -4
    right_bound = 4

    plt.scatter(data[:, 0], data[:, 1], s=0.2, marker='o')
    plt.axis('square')
    plt.axis('off')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()

def plot_rings_example(data, save_path='data_samples.png'):
    left_bound = -7
    right_bound = 7
    
    group_label = data[:, 2]
    color_label = data[:, 3]
    plt.scatter(data[:, 0]*(group_label+1), data[:, 1]*(group_label+1), s=0.2, c=[colors[int(i)] for i in color_label], marker='o')

    plt.axis('square')
    plt.axis('off')
    # plt.title('data samples')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()

def plot_olympic_example(data, save_path='data_samples.png'):
    left_bound = -7
    right_bound = 7

    w = 8.5
    h = 4.
    centers = np.array([[-w, h], [0.0, h], [w, h], [-w * 0.6, -h], [w * 0.6, -h]])
    inds = data[:, 2].astype(int)
    colors = ['blue', 'black', 'red', 'yellow', 'green']

    plt.scatter(data[:, 0]+centers[inds][:,0]/2, data[:, 1]+centers[inds][:,1]/2, s=0.2, c=[colors[int(i)] for i in data[:, 3]], marker='o')

    plt.axis('square')
    plt.axis('off')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()


if __name__ == '__main__':
    # rng = np.random.RandomState(42)
    # data = inf_train_gen("25gaussians_shape", rng, 20000)
    # plot_25_gaussian_shape_example(data, '25gaussians_data.png')

    # rng = np.random.RandomState(42)
    # data = inf_train_gen("rings", rng, 10096)
    # plot_rings_example(data)
    
    # rng = np.random.RandomState()
    # data = inf_train_gen("25gaussians", rng, 20000)
    # plot_25_gaussian_example(data, '25gaussian_data.png')
    
    # rng = np.random.RandomState()
    # data = inf_train_gen("checkerboard", rng, 20000)
    # plot_checkerboard_example(data, 'checkerboard_data.png')

    # rng = np.random.RandomState()
    # data = inf_train_gen("25circles", rng, 20000)
    # plot_25_circles_example(data, '25circles_data.png')

    rng = np.random.RandomState()
    data = inf_train_gen("olympic", rng, 20000)
    plot_olympic_example(data, 'olympic_data.png')
