import torch

def calc_hist(x, win_size_half=(0,0), edges = []):

    win_size = (2*win_size_half[0] + 1, 2*win_size_half[1] + 1)
    dims_num = x.size(1)

    edges_num = len(edges)
    hist_tensor = torch.zeros(size=(x.size(0), edges_num - 1, x.size(2)+1-win_size[0], x.size(3)+1-win_size[1]), device=x.device)
    for ind in range(edges_num-1):
        low = edges[ind]
        high = edges[ind+1]

        mask_ext = torch.zeros(size=(x.size(0), x.size(1), x.size(2)+1, x.size(3)+1), device=x.device)
        mask_ext[:, :, 1:, 1:] = (x >= low) & (x < high)

        integral_img = torch.sum(torch.cumsum(torch.cumsum(mask_ext, dim=2), dim=3), dim=1)

        hist_val = integral_img[:, win_size[0]:, win_size[1]:] + integral_img[:, :-win_size[0], :-win_size[1]] - \
                   integral_img[:, :-win_size[0], win_size[1]:] - integral_img[:, win_size[0]:, :-win_size[1]]

        hist_tensor[:, ind, :, :] = torch.unsqueeze(hist_val, dim=1)

    hist_tensor /= (win_size[0]*win_size[1]*dims_num)

    return hist_tensor
