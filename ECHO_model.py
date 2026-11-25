import os
import torch
import dmercator
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.nn import TransformerConv

from index_calculate import get_evaluation_results


def atanh(x, eps=1e-5):
    x = torch.clamp(x, min=-1. + eps, max=1. - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class PoincareManifold:

    @staticmethod
    def poincare_grad(euclidean_grad, x, c=-1, eps=1e-7):
        squared_norm = torch.sum(x * x, dim=-1, keepdim=True)
        squared_norm = torch.clamp(squared_norm, max=1.0 - eps)
        conformal_factor = ((1.0 + c * squared_norm) / 2.0) ** 2
        return conformal_factor * euclidean_grad

    @staticmethod
    def log_map_zero(x, c=-1, eps=1e-7):
        norm_diff = torch.norm(x, p=2, dim=1, keepdim=True)
        norm_diff = torch.clamp(norm_diff, min=eps)
        sqrt_c = torch.sqrt(torch.abs(torch.tensor(c, dtype=x.dtype, device=x.device)))
        arg = torch.clamp(sqrt_c * norm_diff, max=1.0 - eps)
        atanh_x = atanh(arg)
        lam_zero = 2.0
        scale = (2.0 / (sqrt_c * lam_zero)) * atanh_x / norm_diff
        return scale * x


class HyperbolicEmbedding:

    def __init__(self, args: dict, view_id=None):
        self.args = args
        self.view_idx = view_id

    def fit_transform(self, *, quiet_mode=True, fast_mode=True,
                      validation_mode=True, post_kappa=True):
        inf_coord_path = f'{self.args.log_dir}/HE_view{self.view_idx}/he.inf_coord'
        edgelist_path = f'{self.args.data_dir}/view{self.view_idx}_graph.txt'

        if not os.path.exists(inf_coord_path):
            os.makedirs(f'{self.args.log_dir}/HE_view{self.view_idx}', exist_ok=True)
            edges = []
            nodes_in_edgelist = set()
            with open(edgelist_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        u = int(parts[0]);
                        v = int(parts[1])
                        edges.append((u, v))
                        nodes_in_edgelist.add(u)
                        nodes_in_edgelist.add(v)

            all_nodes = set(range(1, self.args.num_nodes + 1))
            missing_nodes = all_nodes - nodes_in_edgelist
            if missing_nodes:
                for node in sorted(missing_nodes):
                    edges.append((node, node))

            aug_path = f'{self.args.log_dir}/HE_view{self.view_idx}/edgelist_aug.txt'
            with open(aug_path, 'w') as f:
                for (u, v) in edges:
                    f.write(f"{u} {v} 1\n")

            dmercator.embed(
                edgelist_filename=aug_path,
                quiet_mode=quiet_mode,
                fast_mode=fast_mode,
                output_name=f'{self.args.log_dir}/HE_view{self.view_idx}/he',
                validation_mode=validation_mode,
                post_kappa=post_kappa
            )
        return self._parse_mercator_output()

    def _parse_mercator_output(self):
        with open(f'{self.args.log_dir}/HE_view{self.view_idx}/he.inf_coord', 'r') as f:
            lines = f.readlines()

        node_num = int(lines[7].split()[-1])
        beta = float(lines[8].split()[-1])
        mu = float(lines[9].split()[-1])

        node_ids = np.zeros(node_num)
        kappa = np.zeros(node_num)
        angular = np.zeros(node_num)
        radius = np.zeros(node_num)
        for i in range(15, 15 + node_num):
            node_ids[i - 15] = int(lines[i].split()[0])
            kappa[i - 15] = float(lines[i].split()[1])
            angular[i - 15] = float(lines[i].split()[2])
            radius[i - 15] = float(lines[i].split()[3])

        actual_node_ids = set(node_ids)
        all_node_ids = set(range(0, self.args.num_nodes))

        missing_nodes = all_node_ids - actual_node_ids

        if missing_nodes:
            for node in sorted(missing_nodes):
                node_ids = np.append(node_ids, node)
                kappa = np.append(kappa, 0.0)
                angular = np.append(angular, 0.0)
                radius = np.append(radius, 0.0)
        return kappa, angular, radius


class CrossViewRefiner(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super(CrossViewRefiner, self).__init__()
        self.scale = dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.fusion_gate = nn.Linear(dim * 2, 1)

    def forward(self, x_consensus, x_views_list):
        Q = self.q_proj(x_consensus).unsqueeze(1)
        K = torch.stack([self.k_proj(v) for v in x_views_list], dim=1)
        V = torch.stack([self.v_proj(v) for v in x_views_list], dim=1)

        attn_logits = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = torch.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn, V).squeeze(1)

        fusion_weight = torch.sigmoid(self.fusion_gate(torch.cat([x_consensus, context], dim=-1)))
        refined_output = fusion_weight * context + (1 - fusion_weight) * x_consensus

        return self.norm(self.dropout(self.out_proj(refined_output)) + x_consensus)


class MultiViewNet(nn.Module):
    def __init__(self, args, adj_list, *, gate_lr_multiplier=100.0):
        super(MultiViewNet, self).__init__()
        self.args = args
        self.device = args.device
        self.dropout_rate = args.dropout_rate
        self.num_views = args.num_views
        self.common_dim = args.common_feature_dim
        self.num_layers = args.num_layers
        self.poincare_dim = 2  # _init_poincare constructs x/y coordinates.
        self.gate_lr_multiplier = gate_lr_multiplier

        self.adj_list = []
        for adj in adj_list:
            if not torch.is_tensor(adj):
                adj = torch.tensor(adj)
            indices = torch.nonzero(adj).t()
            values = adj[indices[0], indices[1]]
            shape = adj.shape
            adj_sparse = torch.sparse_coo_tensor(indices, values, shape).to(self.device)
            adj_csr = adj_sparse.to_sparse_csr()
            self.adj_list.append(adj_csr)


        self.features_linear_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.common_dim),
                nn.ReLU(),
                nn.LayerNorm(self.common_dim),
                nn.Dropout(args.dropout_rate)
            ) for dim in args.num_view_feature_dims
        ])

        self.view_nodes_net_layers = nn.ModuleList()
        self.supernode_net_layers = nn.ModuleList()
        self.rgt_layers = nn.ModuleList()
        self.refiners = nn.ModuleList()
        self.gates = nn.ParameterList([nn.Parameter(torch.tensor(0.0)) for _ in
                                       range(self.num_layers)])

        self.super_interaction = nn.ModuleList([nn.Linear(self.common_dim, self.common_dim) for _ in range(
            self.num_layers)])

        for l_id in range(self.num_layers):
            v_nodes = nn.ModuleList()
            s_nodes = nn.ModuleList()
            for view_id in range(self.num_views):
                v_nodes.append(nn.Sequential(nn.Linear(self.poincare_dim, args.repr_hid_dim), nn.Tanh(),
                                             nn.Linear(args.repr_hid_dim, args.repr_hid_dim),
                                             nn.LayerNorm(args.repr_hid_dim)))
                s_nodes.append(nn.Sequential(nn.Linear(self.poincare_dim, args.repr_hid_dim), nn.ReLU(),
                                             nn.Linear(args.repr_hid_dim, args.repr_hid_dim),
                                             nn.LayerNorm(args.repr_hid_dim)))
            self.view_nodes_net_layers.append(v_nodes)
            self.supernode_net_layers.append(s_nodes)

            rgt_list = nn.ModuleList()
            for view_id in range(self.num_views):
                rgt_list.append(
                    TransformerConv(in_channels=self.common_dim, out_channels=self.common_dim, heads=args.num_heads,
                                    dropout=self.dropout_rate * 3, concat=False))
            self.rgt_layers.append(rgt_list)

            self.refiners.append(CrossViewRefiner(self.common_dim, self.dropout_rate))


        self.output_linear = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.common_dim // 2, args.num_classes),
        )

        self.softmax = nn.Softmax(dim=1)
        self.CELoss = nn.CrossEntropyLoss()


        self.node_embedding_list = []
        self.kmeans_labels_list = []
        self.supernode_embedding_list = nn.ParameterList()


        for view_id in range(self.num_views):
            node_embedding, angular = self._init_poincare(view_id)
            self.node_embedding_list.append(node_embedding)
            supernode_embedding, _, k_labels = self._init_super_node_simplified(view_id)
            self.supernode_embedding_list.append(supernode_embedding)
            self.kmeans_labels_list.append(k_labels)

        self.to(args.device)

    def _init_poincare(self, view_id):
        _, angular, radius = HyperbolicEmbedding(self.args, view_id).fit_transform()
        radius_max = radius.max() + 1e-8
        radius /= radius_max
        r = np.tanh(radius / 2)
        x, y = r * np.cos(angular), r * np.sin(angular)
        poincare_embedding = torch.from_numpy(np.stack([x, y], axis=1)).float().to(self.args.device)
        return self._check_norm(poincare_embedding), angular

    def _init_super_node_simplified(self, view_id):
        if self.args.supernodes_ratio < 1:
            num = int(self.args.supernodes_ratio * self.args.num_nodes)
        else:
            num = int(self.args.supernodes_ratio)

        node_emb = self.node_embedding_list[view_id].cpu().detach().numpy()

        kmeans = KMeans(n_clusters=num, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(node_emb)
        labels = kmeans.labels_
        kmeans_labels = torch.tensor(labels, device=self.args.device).long()

        assignment_matrix = torch.zeros(num, self.args.num_nodes).to(self.device)
        assignment_matrix.scatter_(0, kmeans_labels.unsqueeze(0), 1.0)

        adj = self.adj_list[view_id]
        ones = torch.ones(adj.shape[1], 1).to(self.device)
        degree = torch.sparse.mm(adj, ones)

        denominator = (assignment_matrix @ degree).sum(dim=-1, keepdim=True) + 1e-8
        super_node_vec = assignment_matrix @ (self.node_embedding_list[view_id] * degree) / denominator
        super_node_embedding = nn.Parameter(self._check_norm(super_node_vec))

        return super_node_embedding, assignment_matrix, kmeans_labels

    def _check_norm(self, embedding, eps=1e-5):
        norm = torch.norm(embedding, dim=-1)
        idx = norm > 1
        if idx.sum() > 0:
            embedding[idx] = embedding[idx] / (norm[idx].unsqueeze(-1) + eps)
        return embedding

    def forward(self, X_list, relation_list):
        aligned_X = []
        for view_id in range(self.num_views):
            aligned_X.append(self.features_linear_list[view_id](X_list[view_id]))

        edge_index_list = []
        for rel in relation_list:
            rel = rel.long().to(self.device)
            if rel.dim() == 2 and rel.shape[0] != 2 and rel.shape[1] == 2:
                rel = rel.t()
            edge_index_list.append(rel)

        curr_state = torch.mean(torch.stack(aligned_X), dim=0)
        view_dynamic_P_list_all = []

        for l_id in range(self.num_layers):
            layer_P_list = []
            layer_view_outputs = []

            for view_id in range(self.num_views):
                h_local = torch.zeros_like(curr_state).to(self.device)
                h_global = torch.zeros_like(curr_state).to(self.device)
                P = None

                n_emb = PoincareManifold.log_map_zero(self.node_embedding_list[view_id])
                s_emb = PoincareManifold.log_map_zero(self.supernode_embedding_list[view_id])
                n_repr = self.view_nodes_net_layers[l_id][view_id](n_emb)
                s_repr = self.supernode_net_layers[l_id][view_id](s_emb)
                P = self.softmax((s_repr @ n_repr.T))
                layer_P_list.append(P)

                h_super = P @ curr_state
                h_super = self.super_interaction[l_id](h_super)
                h_global = P.T @ h_super

                h_in = aligned_X[view_id] + curr_state
                h_local = self.rgt_layers[l_id][view_id](h_in, edge_index_list[view_id])

                beta = torch.sigmoid(self.gates[l_id])
                h_view = beta * h_local + (1 - beta) * h_global

                layer_view_outputs.append(h_view)


            if layer_P_list:
                view_dynamic_P_list_all.append(layer_P_list)

            consensus = torch.mean(torch.stack(layer_view_outputs), dim=0)
            refined = self.refiners[l_id](consensus, layer_view_outputs)
            curr_state = curr_state + refined

        prediction = self.output_linear(curr_state)
        return prediction, curr_state, view_dynamic_P_list_all

    def fit(self, X_list, relation_list, labels, labeled_id, unlabeled_id):
        os.makedirs(f'{self.args.log_dir}', exist_ok=True)
        print(f'Training model ...')

        gate_params = []
        gate_params = list(self.gates.parameters())

        supernode_params = []
        supernode_params = list(self.supernode_embedding_list.parameters())

        base_params = []
        base_params = [
            p for n, p in self.named_parameters()
            if 'supernode_embedding_list' not in n and 'gates' not in n
        ]

        optimizer_params = [{'params': base_params, 'lr': self.args.lr, 'weight_decay': 1e-3}]
        if gate_params:
            optimizer_params.append(
                {'params': gate_params, 'lr': self.args.lr * self.gate_lr_multiplier, 'weight_decay': 0.0})

        optimizer = torch.optim.Adam(optimizer_params, lr=self.args.lr)

        scheduler = StepLR(optimizer, step_size=self.args.lr_step, gamma=self.args.lr_decay)

        self.CELoss = nn.CrossEntropyLoss()
        labels = labels.to(self.args.device).long()

        for epoch in range(1, self.args.max_epoch + 1):
            self.train()
            optimizer.zero_grad()

            prediction, _, _ = self(X_list, relation_list)
            ce_loss = self.CELoss(prediction[labeled_id], labels[labeled_id])
            total_losses = ce_loss
            total_losses.backward()

            current_lr = optimizer.param_groups[0]['lr']
            with torch.no_grad():
                for param in self.supernode_embedding_list.parameters():
                    if param.grad is not None:
                        p_grad = PoincareManifold.poincare_grad(param.grad, param)
                        param.copy_(self._check_norm(param - current_lr * 0.1 * p_grad))
                        param.grad.zero_()


            optimizer.step()
            scheduler.step()

        self.eval()
        with torch.no_grad():
            prediction, _, _ = self(X_list, relation_list)

        output_tes = prediction[unlabeled_id]
        pred_labels = torch.argmax(output_tes, dim=1)
        true_labels = labels[unlabeled_id]
        ACC, Precision, Recall, MAF1, MIF1 = get_evaluation_results(true_labels.cpu().numpy(),
                                                                    pred_labels.cpu().numpy())

        gate_val = torch.sigmoid(self.gates[0]).item()

        return ACC, Precision, Recall, MAF1, MIF1
