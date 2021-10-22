import torch
from . import initialization as init
from examples.mboaz17.conf_utils.hist import calc_hist
from examples.mboaz17.conf_utils.histogramdd import histogramdd
import torch.nn.functional as F

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x, y=None, conf_obj=None, mode=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        if conf_obj:
            hist_list = []
            for ind in range(conf_obj.stage_end):
                if conf_obj.stages_valid[ind]:
                    b, ch, r, c = features[ind].shape
                    dims_num = 4
                    features_reshaped = features[ind].view((b, dims_num, int(ch/dims_num), r, c))
                    features_projected = torch.mean(features_reshaped, dim=2)
                    features_projected_flattened = features_projected.view((dims_num, -1))

                    edges = torch.tensor((0, 1e-6, 1e-2, 1e-1, 1e0, 1e1, 1e10), device=features_projected.device)
                    tmp = histogramdd(features_projected_flattened, edges=edges)

                    hist_tensor = calc_hist(features[ind], win_size_half=win_size_half, edges=[0, 1e-6, 1e-2, 1e-1, 1e0, 1e1, 1e10])
                    hist_list.append((hist_tensor))
                    if mode == 'update':
                        if y is not None:
                            y_downscaled = F.interpolate(y, (hist_tensor.shape[2], hist_tensor.shape[3]), mode='nearest')
                            class_indices = (y_downscaled[0, 1, :, :] < torch.tensor(0.5, device='cuda')).nonzero()
                            hist_mean = torch.mean(hist_tensor[:, :, class_indices[:,0], class_indices[:,1]], dim=-1)
                        else:
                            hist_mean = torch.mean(hist_tensor, dim=(2, 3))
                        conf_obj.hist_list[ind] += hist_mean
                else:
                    hist_list.append(0)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if conf_obj and mode == 'compare':
            score_map = conf_obj.compare_hist_to_model(hist_list, image_size=(x.size(2), x.size(3)))
            return masks, score_map

        return masks

    def predict(self, x, conf_obj=None, mode=None):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x, conf_obj=conf_obj, mode=mode)

        return x

