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
            max_vals, _ = torch.max(y.squeeze(), dim=0)
            tagged_indices = (max_vals).nonzero()
            if len(tagged_indices):
                b, ch, r, c = masks.shape
                dims_num = ch
                # features_flattened = masks.view((dims_num, -1))
                features_flattened = masks[:, :, tagged_indices[:, 0], tagged_indices[:, 1]].squeeze()
                edges = torch.tensor(conf_obj.edges_list, device=features_flattened.device)
                # histogram, _ = histogramdd(features_flattened, edges=edges)
                histogram = conf_obj.calc_hist(features_flattened)
                histogram = histogram.float()
                conf_obj.histogram_1d += histogram
                conf_obj.samples_num += features_flattened.shape[1]
                quantiles = torch.quantile(features_flattened, torch.linspace(0, 1, 11, device=features_flattened.device))
                conf_obj.quantiles += quantiles

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if conf_obj and mode == 'compare':
            score_map = conf_obj.compare_hist_to_model(masks)
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

