import numpy as np

class OnlineToyDataset():
    def __init__(self, data_name):
        super().__init__()
        
        self.data_name = data_name
        self.rng = np.random.RandomState(42)

    def gen_batch(self, batch_size):
        return inf_train_gen(self.data_name, self.rng, batch_size)
    
    def get_category_sizes(self):
        return [10] * 6

    def get_numerical_sizes(self):
        return 11
    
    def evaluate(self, x_gen):
        n_num = self.get_numerical_sizes()
        x_gen_num = x_gen[:, :n_num]
        x_gen_cat = x_gen[:, n_num:]

        x_cat_true = np.stack([gen_cat_data(x_gen_num, i) for i in range(6)], axis=1)
        acc = np.mean(x_gen_cat == x_cat_true)
        return acc
    
def inf_train_gen(data, rng=None, batch_size=4096):
    x_num = np.random.randn(batch_size, 11)
    x_cat = np.stack([gen_cat_data(x_num, i) for i in range(6)], axis=1)
    return np.concatenate([x_num, x_cat], axis=1)
    

def gen_cat_data(x_num, idx):
    
    syn1 = lambda x_num=x_num: x_num[:, 0] * x_num[:, 1]
    syn2 = lambda x_num=x_num: x_num[:, 2]**2 + x_num[:, 3]**2 + x_num[:, 4]**2 + x_num[:, 5]**2 - 4.
    syn3 = lambda x_num=x_num: -10 * np.sin(2*x_num[:, 6]) + 2*np.abs(x_num[:, 7]) + x_num[:, 8] - np.exp(-x_num[:, 9])

    options = {
        0: syn1(),
        1: syn2(),
        2: syn3(),
        3: (x_num[:, 9] < 0) * syn1() + (1 - (x_num[:, 9] < 0)) * syn2(),
        4: (x_num[:, 9] < 0) * syn1() + (1 - (x_num[:, 9] < 0)) * syn3(),
        5: (x_num[:, 9] < 0) * syn2() + (1 - (x_num[:, 9] < 0)) * syn3(),
    }

    eps = np.random.randn(x_num.shape[0]) * .0
    logit = np.tanh(options[idx] + eps)

    num_bins = 10
    ranges = np.linspace(-1, 1, num_bins+1)
    categories = np.digitize(logit, ranges) - 1

    # make sure the number of categories is less than num_bins
    categories[categories >= num_bins] = num_bins-1
    
    return categories

