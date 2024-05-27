import torch
import torch.nn as nn
from loss import second_stage_graph_loss, graph_modeling_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.002)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class Classification(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Classification, self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(input_dim, class_num),
        )

    def forward(self, x):
        out = self.classification(x)
        return out


class VAE1Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VAE1Layer, self).__init__()
        self.input_dim = input_dim
        self.encoder_mean = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.encoder_log_var = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.decoder = nn.Sequential(nn.Linear(output_dim, input_dim), nn.Tanh())
        self.apply(weights_init)

    def reparameterization(self, mean, log_var):
        std = torch.exp(log_var / 2)
        output = mean + std * torch.randn_like(std)
        return output

    def forward(self, x):
        batchsize = x.size(0)
        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        output = self.reparameterization(mean, log_var)
        input_hat = self.decoder(output)

        kld = (
            -0.5
            * torch.sum(1 + log_var - torch.pow(mean, 2) - log_var.exp())
            / (self.input_dim * batchsize)
        )
        return output, input_hat, kld


class VAEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VAEModel, self).__init__()
        self.VAE_firstlayer = VAE1Layer(input_dim, hidden_dim)
        self.VAE_secondlayer = VAE1Layer(hidden_dim, output_dim)

    def forward(self, x):
        hidden1, input_hat_1, kld_1 = self.VAE_firstlayer(x)
        hidden2, input_hat_2, kld_2 = self.VAE_secondlayer(hidden1)
        return hidden1, input_hat_1, kld_1, hidden2, input_hat_2, kld_2


class SpatialModel(nn.Module):
    def __init__(self, input_dim, patch_size, output_dim):
        super(SpatialModel, self).__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(input_dim, 30, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(30, 10, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        self.linear1 = nn.Sequential(
            nn.Linear((patch_size - (3 - 1)) ** 2 * 10, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.conv2d_1(x)
        output = self.conv2d_2(output)

        output = output.reshape(output.shape[0], -1)
        output = self.linear1(output)
        return output


class CDSSnet_LG(nn.Module):
    def __init__(
        self, source_input_dim, target_input_dim, patch_size, hidden_dim, r, class_num
    ):
        super(CDSSnet_LG, self).__init__()
        self.tau = torch.nn.Parameter(torch.tensor([0.8]), requires_grad=True)
        self.source_VAE = VAEModel(source_input_dim, hidden_dim, r)
        self.target_VAE = VAEModel(target_input_dim, hidden_dim, r)
        self.source_spatial = SpatialModel(source_input_dim, patch_size, r)
        self.target_spatial = SpatialModel(target_input_dim, patch_size, r)
        self.classification = Classification(r * 2, class_num)

        self.sigmoid = nn.Sigmoid()

    def loss(self):
        criterion_mse = nn.MSELoss(reduction="mean")
        criterion_crossentropy = nn.CrossEntropyLoss()
        criterion_LG = graph_modeling_loss()
        criterion_second_graph = second_stage_graph_loss()
        return (
            criterion_mse,
            criterion_crossentropy,
            criterion_LG,
            criterion_second_graph,
        )

    def forward(
        self,
        source_data_spectral,
        target_data_spectral,
        source_data_spatial,
        target_data_spatial,
    ):
        tau = self.tau
        (
            source_hidden_spectral_1,
            source_input_hat_1,
            source_kld_1,
            source_hidden_spectral_2,
            source_input_hat_2,
            source_kld_2,
        ) = self.source_VAE(source_data_spectral)

        source_hidden_spectral_2 = self.sigmoid(source_hidden_spectral_2)
        (
            target_hidden_spectral_1,
            target_input_hat_1,
            target_kld_1,
            target_hidden_spectral_2,
            target_input_hat_2,
            target_kld_2,
        ) = self.target_VAE(target_data_spectral)

        target_hidden_spectral_2 = self.sigmoid(target_hidden_spectral_2)
        source_out_spatial = self.source_spatial(source_data_spatial)
        target_out_spatial = self.target_spatial(target_data_spatial)

        source_hidden_spectral_spatial = torch.cat(
            (source_hidden_spectral_2, source_out_spatial), 1
        )
        target_hidden_spectral_spatial = torch.cat(
            (target_hidden_spectral_2, target_out_spatial), 1
        )
        source_out = self.classification(source_hidden_spectral_spatial)
        target_out = self.classification(target_hidden_spectral_spatial)

        return (
            source_hidden_spectral_1,
            source_input_hat_1,
            source_kld_1,
            source_hidden_spectral_spatial,
            source_out_spatial,
            source_hidden_spectral_2,
            source_input_hat_2,
            source_kld_2,
            target_hidden_spectral_1,
            target_input_hat_1,
            target_kld_1,
            target_hidden_spectral_spatial,
            target_out_spatial,
            target_hidden_spectral_2,
            target_input_hat_2,
            target_kld_2,
            tau,
            source_out,
            target_out,
        )


class TDSSnet(nn.Module):
    def __init__(self, target_input_dim, patch_size, hidden_dim, r, class_num):
        super(TDSSnet, self).__init__()
        self.target_VAE = VAEModel(target_input_dim, hidden_dim, r)
        self.target_spatial = SpatialModel(target_input_dim, patch_size, r)
        self.classification = Classification(r * 2, class_num)
        self.sigmoid = nn.Sigmoid()

    def loss(self):
        criterion_mse = nn.MSELoss(reduction="mean")
        criterion_crossentropy = nn.CrossEntropyLoss()
        return criterion_mse, criterion_crossentropy

    def forward(self, target_data, target_data_spatial):
        (
            target_hidden,
            target_input_hat_1,
            target_kld_1,
            target_hidden_2,
            target_input_hat_2,
            target_kld_2,
        ) = self.target_VAE(target_data)
        target_hidden_2 = self.sigmoid(target_hidden_2)
        target_out_spatial = self.target_spatial(target_data_spatial)

        target_hidden_spectral_spatial = torch.cat(
            (target_hidden_2, target_out_spatial * 1), 1
        )
        target_out = self.classification(target_hidden_spectral_spatial)
        return (
            target_hidden,
            target_input_hat_1,
            target_kld_1,
            target_hidden_spectral_spatial,
            target_input_hat_2,
            target_kld_2,
            target_out,
        )
