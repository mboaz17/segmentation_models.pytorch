import torch
import torch.nn.functional as F
import numpy as np

class ConfEst:

    def __init__(self, classes_num=1, channels_num=None):
        self.classes_num = classes_num
        if not channels_num:
            self.channels_num = classes_num
        else:
            self.channels_num = channels_num

        self.histogram = [0.0 for i in range(self.classes_num)]
        self.histogram_1d = [0.0 for i in range(self.classes_num)]
        self.quantiles = [0.0 for i in range(self.classes_num)]
        self.samples_num = [0.0 for i in range(self.classes_num)]

        # self.edges_list = [[-1e6] + list(np.linspace(-8,8,9)) + [1e6] for i in range(self.classes_num)]
        self.edges_list = [
         torch.tensor([-10.8035, -8.0361, -6.7754, -5.1637, -3.2113, -2.4877, -1.4862,
                 2.6298, 3.9689, 4.6755, 6.4872], device='cuda:0'),
         torch.tensor([-13.6183, -7.9322, -6.7876, -5.9239, -5.0735, -4.1047, -2.7102,
                 2.4256, 4.4332, 5.8189, 9.5293], device='cuda:0'),
         torch.tensor([-19.7921, -12.1041, -9.9203, -8.5043, -7.3701, -6.3719, -4.9884,
                 4.3564, 6.6058, 8.0787, 11.7919], device='cuda:0')]
        self.edges_list = [[torch.tensor(-1e6, device='cuda:0')] + list(curr_list[1:-1]) + [torch.tensor(1e6, device='cuda:0')] for curr_list in self.edges_list]


    def calc_hist(self, x, cls):
        # x is a tensor with shape [channels_num, samples_num]

        edges = torch.tensor(self.edges_list[cls], device=x.device)
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

        return histogram_1d.float()


    def compare_hist_to_model(self, x, cls):

        edges = self.edges_list[cls]
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
            score_map[indices[:, 0], indices[:, 1]] = self.histogram_1d[cls][ind]

        return score_map

    def normalize_hist_after_batch(self, iterations_num=1):
        for cls in range(self.classes_num):
            self.histogram_1d[cls] /= self.samples_num[cls]
            self.quantiles[cls] /= self.samples_num[cls]
