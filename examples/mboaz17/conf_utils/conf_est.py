import torch
import torch.nn.functional as F
import numpy as np

class ConfEst:

    def __init__(self):
        self.histogram = 0.0
        self.histogram_1d = 0.0
        self.quantiles = 0.0
        self.samples_num = 0
        # self.edges_list = [-1e6] + list(np.linspace(-5,5,11)) + [1e6]
        self.edges_list = [-1e6] + [-10.5, -7, -5.5, -4.5, -2, 0, 3, 5, 7.5] + [1e6]
        # [-19.3564, -10.7485, -7.1572, -5.7219, -4.4656, -1.9554, 0.1445,
        #  2.7789, 4.8235, 7.5348, 19.4049]

    def calc_hist(self, x):
        # x is a tensor with shape [channels_num, samples_num]

        edges = torch.tensor(self.edges_list, device=x.device)
        edges_num = len(edges)
        dims_num = x.shape[0]
        hist_indices_map = torch.zeros(size=(dims_num, x.size(1)), device=x.device, dtype=torch.int)
        hist_indices_map_1d = torch.zeros(x.size(1), device=x.device, dtype=torch.int)

        for ch in range(dims_num):
            total = 0
            for ind in range(edges_num - 1):
                low = edges[ind]
                high = edges[ind + 1]
                x_curr = x[ch]
                mask_curr = (x_curr >= low) & (x_curr < high)
                indices = mask_curr.nonzero()
                total += len(indices)
                hist_indices_map[ch, indices] = ind
            hist_indices_map_1d += ((edges_num-1)**ch) * hist_indices_map[ch]

        histogram_1d = torch.zeros((edges_num-1)**dims_num, device=x.device, dtype=torch.int)
        for ind in range(hist_indices_map_1d.min(), hist_indices_map_1d.max()+1, 1):
            indices = (hist_indices_map_1d == ind).nonzero()
            if len(indices):
                histogram_1d[ind] = len(indices)

        return histogram_1d


    def compare_hist_to_model(self, x):

        edges = self.edges_list
        edges_num = len(edges)
        dims_num = x.shape[1]
        hist_indices_map = torch.zeros(size=(dims_num, x.size(2), x.size(3)), device=x.device, dtype=torch.int)
        hist_indices_map_1d = torch.zeros(size=(x.size(2), x.size(3)), device=x.device, dtype=torch.int)
        score_map = torch.zeros(size=(x.size(2), x.size(3)), device=x.device, dtype=torch.float32)

        for ch in range(dims_num):
            total = 0
            for ind in range(edges_num - 1):
                low = edges[ind]
                high = edges[ind + 1]
                x_curr = x[0, ch]
                mask_curr = (x_curr >= low) & (x_curr < high)
                indices = mask_curr.nonzero()
                total += len(indices[:, 0])
                hist_indices_map[ch, indices[:, 0], indices[:, 1]] = ind
            hist_indices_map_1d += ((edges_num-1)**ch) * hist_indices_map[ch]

        for ind in range(hist_indices_map_1d.min(), hist_indices_map_1d.max()+1, 1):
            indices = (hist_indices_map_1d == ind).nonzero()
            score_map[indices[:, 0], indices[:, 1]] = self.histogram_1d[ind]

        return score_map

    def normalize_hist_after_batch(self, iterations_num=1):
        self.histogram_1d /= self.samples_num
        self.quantiles /= iterations_num
