import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv, WeightAndSum
from dgllife.model import MLPPredictor

from scipy.sparse import load_npz
import pickle



class WeightedSumAndMax(nn.Module):
    # Apply weighted sum and max pooling to the node
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            # 将节点特征赋值给图的节点数据
            bg.ndata['node_pka'] = feats
            h_g_max = dgl.max_nodes(bg, 'node_pka')
        # 按列拼接
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)

        return h_g

class MLP(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_feats):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_feats)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class FPN(nn.Module):
    def __init__(self, dropout_fpn, hidden_dim, fp_2_dim, device):
        super(FPN, self).__init__()

        self.dropout_fpn = dropout_fpn
        self.hidden_dim = hidden_dim
        self.fp_2_dim = fp_2_dim


        self.fp_dim = 2304

        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

        self.device = device

    def forward(self, kpgt_features):
        kpgt_features = torch.Tensor(kpgt_features).to(self.device)

        fpn_out = self.fc1(kpgt_features)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)

        return fpn_out


class GATLayer(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop,
        attn_drop,
        alpha=0.2,
        residual=True,
        agg_mode="flatten",
        activation=None,
        bias=True,
        allow_zero_in_degree=False,
    ):
        super(GATLayer, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gat_conv = GATConv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=alpha,
            residual=residual,
            bias=bias,
            allow_zero_in_degree=allow_zero_in_degree,
        ).to(device)

        assert agg_mode in ["flatten", "mean"]
        self.agg_mode = agg_mode
        self.activation = activation

    def forward(self, bg, feats):
        feats = self.gat_conv(bg, feats)
        if self.agg_mode == "flatten":
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)

        if self.activation is not None:
            feats = self.activation(feats)

        return feats


class GAT(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_dim,
        num_heads,
        feat_drops,
        attn_drops,
        alphas,
        residuals,
        agg_modes,
        activations,
        biases,
        allow_zero_in_degree=False,
    ):
        super(GAT, self).__init__()

        if hidden_dim is None:
            # 默认为[64， 64]
            hidden_dim = [256, 256, 256]

        n_layers = len(hidden_dim)
        if num_heads is None:
            num_heads = [8 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0.1 for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0.1 for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ["flatten" for _ in range(n_layers - 1)]
            agg_modes.append("mean")
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        lengths = [
            len(hidden_dim),
            len(num_heads),
            len(feat_drops),
            len(attn_drops),
            len(alphas),
            len(residuals),
            len(agg_modes),
            len(activations),
            len(biases),
        ]
        assert len(set(lengths)) == 1, (
            "Expect the lengths of hidden_dim, num_heads, "
            "feat_drops, attn_drops, alphas, residuals, "
            "agg_modes, activations, and biases to be the same, "
            "got {}".format(lengths)
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i in range(n_layers):
            self.gnn_layers.append(
                GATLayer(
                    in_feats,
                    hidden_dim[i],
                    num_heads[i],
                    feat_drops[i],
                    attn_drops[i],
                    alphas[i],
                    residuals[i],
                    agg_modes[i],
                    activations[i],
                    biases[i],
                    allow_zero_in_degree=allow_zero_in_degree,
                ).to(device)
            )
            if agg_modes[i] == "flatten":
                in_feats = hidden_dim[i] * num_heads[i]
            else:
                in_feats = hidden_dim[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 将输入数据移动到该设备上
        g = g.to(device)
        feats = feats.to(device)
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats


class GATPredictor(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_dim,
                 num_heads,
                 feat_drops,
                 attn_drops,
                 alphas=None,
                 residuals=None,
                 agg_modes=None,
                 activations=None,
                 biases=None):
        super(GATPredictor, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gnn = GAT(in_feats=in_feats,
                       hidden_dim=hidden_dim,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations,
                       biases=biases).to(device)
        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_dim[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_dim[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats).to(device)

    def forward(self, bg, feats):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)

        return graph_feats


class PGnT(nn.Module):
    def __init__(self, is_classif, in_feats, args):
        super(PGnT, self).__init__()
        self.gat_scale = args.gat_scale

        self.is_classif = is_classif
        self.in_feats = in_feats
        self.drops = args.drops
        self.dropout_fpn = args.dropout
        self.hidden_dim = args.hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.linear_dim = int(self.hidden_dim[0])

        if self.is_classif:
              self.sigmoid = nn.Sigmoid()


        self.gat = GATPredictor(in_feats=in_feats,
                                hidden_dim=self.hidden_dim,
                                num_heads=args.num_heads,
                                feat_drops=self.drops,
                                attn_drops=self.drops)

        self.fpn = FPN(dropout_fpn=self.dropout_fpn,
                       hidden_dim=self.linear_dim,
                       fp_2_dim=args.fp_2_dim,
                       device=self.device)

        self.mlp = MLP(in_feats=self.linear_dim * 2, hidden_dim=self.linear_dim * 4, out_feats=self.linear_dim* 2)

        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=self.linear_dim * 2, out_features=self.linear_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=self.linear_dim, out_features=1, bias=True)
        )

        self.gat_dim = int((self.linear_dim * 2 * self.gat_scale) // 1)
        self.fc_gat = nn.Linear(self.linear_dim * 2, self.gat_dim).cuda()
        self.fc_fpn = nn.Linear(self.linear_dim, self.linear_dim * 2 - self.gat_dim).cuda()
        self.act_func = nn.ReLU()



    def forward(self, bg, kpgt_features):

        feats = bg.ndata['node_pka']
        gat_out = self.gat(bg, feats)
        fpn_out = self.fpn(kpgt_features)

        gat_out = self.fc_gat(gat_out)
        gat_out = self.act_func(gat_out)

        fpn_out = self.fc_fpn(fpn_out)
        fpn_out = self.act_func(fpn_out)

        output = torch.cat([gat_out, fpn_out], axis=1)
        output = self.mlp(output)
        out = self.ffn(output)


        return output, out











