import torch
import torch.nn.functional as F

class ConfEst:

    def __init__(self):
        self.dims_num = 2
        self.stages_valid = [0, 1, 1, 1, 1, 1]
        self.stage_end = len(self.stages_valid)
        self.hist_list = []
        for ind in range(self.stage_end):
            self.hist_list.append(0)

        self.edges_list = []
        for d in range(self.dims_num):
            self.edges_list.append([0, 1e-6, 1e-2, 1e-1, 1e0, 1e1, 1e10])

    def compare_hist_to_model(self, hist_list=[], image_size=(0, 0)):
        assert len(self.hist_list) == len(hist_list)

        score_map = 0
        total_weight = 0
        for ind in range(len(hist_list)):
            if self.stages_valid[ind]:
                hist_norm1 = torch.norm(self.hist_list[ind])
                hist_norm2 = torch.norm(hist_list[ind], dim=1).unsqueeze(dim=1)
                total_weight += self.stages_valid[ind]
                if self.criterion == 'inner_product':
                    score_curr = torch.sum(self.hist_list[ind].unsqueeze(dim=2).unsqueeze(dim=3) * hist_list[ind], dim=1).unsqueeze(dim=1)
                    score_curr /= (hist_norm1 * hist_norm2)
                    score_curr = F.pad(score_curr, (self.win_half_size_list[ind][1], self.win_half_size_list[ind][1],
                                       self.win_half_size_list[ind][0], self.win_half_size_list[ind][0]), 'constant', 0)
                    score_map += self.stages_valid[ind] * F.interpolate(score_curr, image_size, mode="bilinear", align_corners=True)
        score_map /= total_weight
        return score_map

    def normalize_hist_after_batch(self, iterations_num=1):
        for ind in range(self.stage_end):
            self.hist_list[ind] /= iterations_num


    def calc_hist(self, x):

        assert x.shape[1] == self.dims_num


