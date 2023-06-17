import torch
from third_party import dnnlib

import numpy as np
from third_party.ops import bias_act
from third_party.ops import grid_sample_gradfix
from third_party.ops import grid_sample
from models.swin_transformer import WindowAttention, window_partition, window_reverse
from timm.models.layers import to_2tuple

import torch.nn as nn

from models.embedder import get_embedder


def generate_planes():
    return torch.tensor([[[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]],
                         [[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]],
                         [[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = grid_sample.grid_sample_2d(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)

    return output_features

class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        rendering_kwargs    = {},
        triplane_sdf = {},
        triplane_sdf_ini={},
    ):
        super().__init__()
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.rendering_kwargs = rendering_kwargs
        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)  # use Parameter so it could be checkpointed
        self.tritype = 0

        self.sdf_para = SDFNetwork(**triplane_sdf_ini)
        self.decoder = OSG_PE_SDFNetwork(**triplane_sdf, multires=self.rendering_kwargs['PE_res'], geometric_init=self.rendering_kwargs['is_dec_geoinit'])

        self._last_planes = None

        self.plane_axes = generate_planes()

        ini_sdf = torch.randn([3, self.img_channels, self.img_resolution, self.img_resolution])
        xs = (torch.arange(self.img_resolution) - (self.img_resolution / 2 - 0.5)) / (self.img_resolution / 2 - 0.5)
        ys = (torch.arange(self.img_resolution) - (self.img_resolution / 2 - 0.5)) / (self.img_resolution / 2 - 0.5)
        (ys, xs) = torch.meshgrid(-ys, xs)
        N = self.img_resolution
        zs = torch.zeros(N, N)
        inputx = torch.stack([zs, xs, ys]).permute(1, 2, 0).reshape(N ** 2, 3)
        inputy = torch.stack([xs, zs, ys]).permute(1, 2, 0).reshape(N ** 2, 3)
        inputz = torch.stack([xs, ys, zs]).permute(1, 2, 0).reshape(N ** 2, 3)
        ini_sdf[0] = self.sdf_para(inputx).permute(1, 0).reshape(self.img_channels, N, N)
        ini_sdf[1] = self.sdf_para(inputy).permute(1, 0).reshape(self.img_channels, N, N)
        ini_sdf[2] = self.sdf_para(inputz).permute(1, 0).reshape(self.img_channels, N, N)

        self.planes = torch.nn.Parameter(ini_sdf.unsqueeze(0), requires_grad=True)

        self.window_size = self.rendering_kwargs['attention_window_size']
        self.numheads = self.rendering_kwargs['attention_numheads']
        self.attn = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size), num_heads=self.numheads)
        self.window_size4 = self.window_size * 2
        self.attn4 = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size4), num_heads=self.numheads)
        self.window_size2 = self.window_size // 2
        self.attn2 = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size2), num_heads=self.numheads)


    def forward(self, coordinates, directions=None):
        planes = self.planes
        planes = planes.view(len(planes), 3, planes.shape[-3], planes.shape[-2], planes.shape[-1])
        return self.run_model(planes, self.decoder, coordinates.unsqueeze(0), directions, self.rendering_kwargs)

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        img_channels = self.img_channels
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode='zeros',
                                          box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], planes.shape[-2], planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, img_channels)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, img_channels)
        shifted_x = window_reverse(attn_windows, self.window_size, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates, padding_mode='zeros',
                                              box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], planes.shape[-2],
                                                  planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size4)
        x_windows = x_windows.view(-1, self.window_size4 * self.window_size4, img_channels)
        attn_windows = self.attn4(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size4, self.window_size4, img_channels)
        shifted_x = window_reverse(attn_windows, self.window_size4, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention4 = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates,
                                                        padding_mode='zeros',
                                                        box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], planes.shape[-2],
                                                  planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size2)
        x_windows = x_windows.view(-1, self.window_size2 * self.window_size2, img_channels)
        attn_windows = self.attn2(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size2, self.window_size2, img_channels)
        shifted_x = window_reverse(attn_windows, self.window_size2, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention2 = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates,
                                                         padding_mode='zeros',
                                                         box_warp=options['box_warp'])

        sampled_features = torch.cat([sampled_features_attention4, sampled_features_attention, sampled_features_attention2, sampled_features], dim=-1)

        periodic_fns = [torch.sin, torch.cos]
        embed_fn, input_ch = get_embedder(options['multiply_PE_res'], input_dims=3, periodic_fns=periodic_fns)
        sample_PE = embed_fn(sample_coordinates)
        inputs = sample_PE
        d = sampled_features.shape[-1] // (inputs.shape[-1] // 3)
        x = inputs.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 0]
        y = inputs.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 1]
        z = inputs.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 2]
        inputs = torch.cat([z, x, y]).tile(1, 1, d).view(3, inputs.shape[1], -1)
        sampled_features = sampled_features * inputs.unsqueeze(0)
        _, dim, N, nf = sampled_features.shape

        out = decoder(sampled_features, sample_coordinates, sample_directions)
        return out

    def sdf(self, coordinates):
        return self.forward(coordinates)[:, :1]

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def grid_sampling(self, img_resolution):
        xs = (torch.arange(img_resolution) - (img_resolution / 2 - 0.5)) / (img_resolution / 2 - 0.5)
        ys = (torch.arange(img_resolution) - (img_resolution / 2 - 0.5)) / (img_resolution / 2 - 0.5)
        index = torch.stack(torch.meshgrid(ys, xs)).permute(1, 2, 0).unsqueeze(0).tile(3, 1, 1, 1)
        return index



class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 8,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter((torch.ones([out_features, in_features]) + torch.randn([out_features, in_features]) * 0.01) / lr_multiplier / self.in_features)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

class FullyConnectedLayer2(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 8,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier) 
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features) 
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.multires = multires

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 1:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


class OSG_PE_SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(10,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False):
        super(OSG_PE_SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.multires = multires
        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)  # use Parameter so it could be checkpointed

        d_PE = 3
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_PE)  # d_in
            self.embed_fn_fine = embed_fn
            self.num_eoc = int((input_ch - d_PE) / 2)   # d_in
            dims[0] = d_in + self.num_eoc + d_PE
        else:
            dims[0] = d_in + d_PE 

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, inputs_PE, sample_directions):
        _, dim, N, nf = inputs.shape
        inputs = inputs.squeeze(0).permute(1, 2, 0).reshape(N, nf*dim)
        inputs_PE = inputs_PE.squeeze(0)
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            input_enc = self.embed_fn_fine(inputs_PE)
            nfea_eachband = int(input_enc.shape[1] / self.multires) 
            N = int(self.multires / 2)
            inputs_enc, weight = coarse2fine(0.5 * (self.progress.data-0.1), input_enc, self.multires)
            inputs_enc = inputs_enc.view(-1, self.multires, nfea_eachband)[:, :N, :].view([-1, self.num_eoc])
            input_enc = input_enc.view(-1, self.multires, nfea_eachband)[:, :N, :].view([-1, self.num_eoc]).contiguous()
            input_enc = (input_enc.view(-1, N) * weight[:N]).view([-1, self.num_eoc])
            flag = weight[:N].tile(input_enc.shape[0], nfea_eachband,1).transpose(1,2).contiguous().view([-1, self.num_eoc])
            inputs_enc = torch.where(flag > 0.01, inputs_enc, input_enc)

            inputs_PE = torch.cat([inputs_PE, inputs_enc], dim=-1)

        inputs = torch.cat([inputs_PE, inputs], dim=-1)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)


def coarse2fine(progress_data, inputs, L):
    barf_c2f = [0.1, 0.5]
    if barf_c2f is not None:
        start, end = barf_c2f
        alpha = (progress_data - start) / (end - start) * L
        k = torch.arange(L, dtype=torch.float32, device=inputs.device)
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        shape = inputs.shape
        input_enc = (inputs.view(-1, L, int(shape[1]/L)) * weight.tile(int(shape[1]/L),1).T).view(*shape)
    return input_enc, weight
