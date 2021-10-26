import torch
from . import initialization as init
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

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if conf_obj and mode == 'update':
            b, dims_num, r, c = masks.shape
            for cls in range(conf_obj.classes_num):

                indices_curr = (y[0, cls]).nonzero()
                if len(indices_curr):
                    # features_flattened = masks.view((dims_num, -1))
                    features_flattened = masks[:, :, indices_curr[:, 0], indices_curr[:, 1]].squeeze()
                    histogram_1d = conf_obj.calc_hist(features_flattened, cls)
                    conf_obj.histogram_1d[cls] += histogram_1d
                    conf_obj.samples_num[cls] += features_flattened.shape[1]
                    quantile_edges = torch.linspace(0, 1, 11, device=features_flattened.device)
                    quantiles = torch.quantile(features_flattened, quantile_edges)
                    conf_obj.quantiles[cls] += features_flattened.shape[1] * quantiles

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if conf_obj and mode == 'compare':
            score_map = torch.zeros(size=(conf_obj.classes_num, masks.size(2), masks.size(3)), device=masks.device)
            for cls in range(conf_obj.classes_num):
                score_map[cls] = conf_obj.compare_hist_to_model(masks, cls)
                # in case of Identity() activation
            masks = torch.nn.functional.softmax(masks, dim=1)
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

