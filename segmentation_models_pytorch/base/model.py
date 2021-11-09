import torch
from . import initialization as init
import torch.nn.functional as F
import examples.mboaz17.conf_utils.vae as vae

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x, y=None, conf_obj=None, vae_obj=None, mode=None):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        if vae_obj:
            vae_input = decoder_output
            vae_output, vae_mean, vae_logvar = vae_obj(vae_input)
            if y is not None:
                y_downscaled = F.interpolate(y, (vae_input.shape[2], vae_input.shape[3]))
                indices_pos = (y_downscaled[0].sum(dim=0)).nonzero()
                loss_vae = vae.vae_loss(vae_output, vae_input, vae_mean, vae_logvar, indices_pos)
            else:
                loss_vae = None

            if 0:  # conf_obj and mode == 'update':
                vae_mean_interpolated = F.interpolate(vae_mean, (y.shape[2], y.shape[3]))
                b, dims_num, r, c = vae_mean_interpolated.shape
                for cls in range(conf_obj.classes_num):

                    indices_curr = (y[0, cls]).nonzero()
                    if len(indices_curr):
                        # features_flattened = masks.view((dims_num, -1))
                        features_flattened = vae_mean_interpolated[:, :, indices_curr[:, 0], indices_curr[:, 1]].squeeze()
                        histogram_1d = conf_obj.calc_hist(features_flattened, cls)
                        conf_obj.histogram_1d[cls] += histogram_1d
                        conf_obj.samples_num[cls] += features_flattened.shape[1]
                        quantile_edges = torch.linspace(0, 1, 6, device=features_flattened.device)
                        quantiles = torch.quantile(features_flattened, quantile_edges)
                        conf_obj.quantiles[cls] += features_flattened.shape[1] * quantiles
                return masks, vae_input, vae_output, vae_mean, vae_logvar, loss_vae

            elif conf_obj and mode == 'compare':
                score_map = torch.zeros(size=(conf_obj.classes_num, masks.size(2), masks.size(3)), device=masks.device)
                for cls in range(conf_obj.classes_num):
                    if 0:
                        res = conf_obj.compare_hist_to_model(vae_mean, cls)
                        score_map[cls] = F.interpolate(res.unsqueeze(dim=0).unsqueeze(dim=0), (masks.size(2), masks.size(3))).squeeze()
                    elif 0:
                        res = torch.mean( vae_mean**2, dim=1)
                        score_map[cls] = F.interpolate(res.unsqueeze(dim=0), (masks.size(2), masks.size(3))).squeeze()
                    elif 1:
                        res = torch.mean( (vae_output - vae_input)**2, dim=1)
                        score_map[cls] = F.interpolate(res.unsqueeze(dim=0), (masks.size(2), masks.size(3))).squeeze()
                    # in case of Identity() activation
                masks = torch.nn.functional.softmax(masks, dim=1)
                return masks, score_map

            return masks, vae_input, vae_output, vae_mean, vae_logvar, loss_vae

        return masks

    def predict(self, x, conf_obj=None, vae_obj=None, mode=None):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x, conf_obj=conf_obj, vae_obj=vae_obj, mode=mode)

        return x

