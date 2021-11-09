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
            torch.linspace(-0.2, 0.2, 6, device='cuda:0'),
            torch.linspace(-0.2, 0.2, 6, device='cuda:0'),
            torch.linspace(-0.2, 0.2, 6, device='cuda:0')]
            # torch.tensor([-0.0098, 0.0022, 0.0196, 0.0281, 0.1082, 0.1910], device='cuda:0'),
            #  torch.tensor([-0.0070, 0.0117, 0.0188, 0.0276, 0.0795, 0.1570], device='cuda:0'),
            #  torch.tensor([-0.0227, 0.0020, 0.0061, 0.0108, 0.0471, 0.1100], device='cuda:0')]
        # self.edges_list = [[torch.tensor(-1e6, device='cuda:0')] + list(curr_list[1:-1]) + [torch.tensor(1e6, device='cuda:0')] for curr_list in self.edges_list]
        self.edges_list = [[torch.tensor(-1e6, device='cuda:0')] + list(curr_list) + [torch.tensor(1e6, device='cuda:0')] for curr_list in self.edges_list]


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
            self.histogram_1d[cls] /= (self.samples_num[cls] + 1e-10)
            self.quantiles[cls] /= (self.samples_num[cls] + 1e-10)
