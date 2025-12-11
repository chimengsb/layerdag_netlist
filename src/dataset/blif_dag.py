import torch
from collections import defaultdict
from torch.utils.data import Dataset

__all__ = ['LayerDAGNodeCountDataset',
           'LayerDAGNodePredDataset',
           'LayerDAGEdgePredDataset',
           'collate_node_count',
           'collate_node_pred',
           'collate_edge_pred']

class LayerDAGBaseDataset(Dataset):
    def __init__(self, conditional=False):
        self.input_src = []
        self.input_dst = []
        self.input_x_n = []
        self.input_level = []
        self.input_e_start = []
        self.input_e_end = []
        self.input_n_start = []
        self.input_n_end = []
        self.conditional = conditional
        if conditional:
            self.input_y = []
            self.input_g = []

    def get_in_deg(self, dst, num_nodes):
        # 增加安全性，防止 dst 为空时报错
        if isinstance(dst, torch.Tensor) and dst.numel() == 0:
            return [0] * num_nodes
        elif isinstance(dst, list) and not dst:
            return [0] * num_nodes
        return torch.bincount(dst, minlength=num_nodes).tolist()

    def get_out_adj_list(self, src, dst):
        out_adj_list = defaultdict(list)
        num_edges = len(src)
        for i in range(num_edges):
            out_adj_list[src[i]].append(dst[i])
        return out_adj_list

    def get_in_adj_list(self, src, dst):
        in_adj_list = defaultdict(list)
        num_edges = len(src)
        for i in range(num_edges):
            in_adj_list[dst[i]].append(src[i])
        return in_adj_list

    def base_postprocess(self):
        self.input_src = torch.LongTensor(self.input_src)
        self.input_dst = torch.LongTensor(self.input_dst)
        self.input_x_n = torch.LongTensor(self.input_x_n)
        self.input_level = torch.LongTensor(self.input_level)
        self.input_e_start = torch.LongTensor(self.input_e_start)
        self.input_e_end = torch.LongTensor(self.input_e_end)
        self.input_n_start = torch.LongTensor(self.input_n_start)
        self.input_n_end = torch.LongTensor(self.input_n_end)
        if self.conditional:
            self.input_y = torch.tensor(self.input_y) if isinstance(self.input_y, list) else self.input_y
            self.input_g = torch.LongTensor(self.input_g)

class LayerDAGNodeCountDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.label = []
        for i in range(len(dag_dataset)):
            try:
                data_i = dag_dataset[i]
                if conditional:
                    src, dst, x_n, y = data_i
                    input_g = len(self.input_y)
                    self.input_y.append(y)
                else:
                    src, dst, x_n = data_i

                input_n_start = len(self.input_x_n)
                input_n_end = len(self.input_x_n)
                input_e_start = len(self.input_src)
                input_e_end = len(self.input_src)

                # Dummy Node
                self.input_x_n.append(dag_dataset.dummy_category)
                input_n_end += 1
                src = src + 1
                dst = dst + 1
                level = 0
                self.input_level.append(level)
                
                num_nodes = len(x_n) + 1
                in_deg = self.get_in_deg(dst, num_nodes)
                src_list, dst_list = src.tolist(), dst.tolist()
                x_n_list = x_n.tolist()
                out_adj_list = self.get_out_adj_list(src_list, dst_list)
                in_adj_list = self.get_in_adj_list(src_list, dst_list)

                frontiers = [u for u in range(1, num_nodes) if in_deg[u] == 0]
                frontier_size = len(frontiers)
                
                while frontier_size > 0:
                    level += 1
                    self.input_e_start.append(input_e_start)
                    self.input_e_end.append(input_e_end)
                    self.input_n_start.append(input_n_start)
                    self.input_n_end.append(input_n_end)
                    if conditional:
                        self.input_g.append(input_g)
                    self.label.append(frontier_size)

                    next_frontiers = []
                    for u in frontiers:
                        self.input_x_n.append(x_n_list[u - 1])
                        self.input_level.append(level)
                        for t in in_adj_list[u]:
                            self.input_src.append(t)
                            self.input_dst.append(u)
                            input_e_end += 1
                        for v in out_adj_list[u]:
                            in_deg[v] -= 1
                            if in_deg[v] == 0:
                                next_frontiers.append(v)
                    input_n_end += frontier_size
                    frontiers = next_frontiers
                    frontier_size = len(frontiers)

                self.input_e_start.append(input_e_start)
                self.input_e_end.append(input_e_end)
                self.input_n_start.append(input_n_start)
                self.input_n_end.append(input_n_end)
                if conditional:
                    self.input_g.append(input_g)
                self.label.append(frontier_size)
            except Exception as e:
                print(f"Error in NodeCountDataset index {i}: {e}")
                continue

        self.base_postprocess()
        self.label = torch.LongTensor(self.label)
        self.max_layer_size = self.label.max().item() if self.label.numel() > 0 else 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        input_e_start, input_e_end = self.input_e_start[index], self.input_e_end[index]
        input_n_start, input_n_end = self.input_n_start[index], self.input_n_end[index]
        input_abs_level = self.input_level[input_n_start:input_n_end]
        input_rel_level = input_abs_level.max() - input_abs_level if input_abs_level.numel() > 0 else input_abs_level

        if self.conditional:
            input_g = self.input_g[index]
            input_y = self.input_y[input_g].item() if self.input_y.ndim > 1 else self.input_y[input_g]
            return self.input_src[input_e_start:input_e_end], self.input_dst[input_e_start:input_e_end], self.input_x_n[input_n_start:input_n_end], input_abs_level, input_rel_level, input_y, self.label[index]
        else:
            return self.input_src[input_e_start:input_e_end], self.input_dst[input_e_start:input_e_end], self.input_x_n[input_n_start:input_n_end], input_abs_level, input_rel_level, self.label[index]

class LayerDAGNodePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False, get_marginal=True):
        super().__init__(conditional)
        self.label_start = []
        self.label_end = []
        
        for i in range(len(dag_dataset)):
            try:
                data_i = dag_dataset[i]
                if conditional:
                    src, dst, x_n, y = data_i
                    input_g = len(self.input_y)
                    self.input_y.append(y)
                else:
                    src, dst, x_n = data_i

                input_n_start = len(self.input_x_n)
                input_n_end = len(self.input_x_n)
                input_e_start = len(self.input_src)
                input_e_end = len(self.input_src)

                self.input_x_n.append(dag_dataset.dummy_category)
                input_n_end += 1
                src = src + 1
                dst = dst + 1
                label_start = len(self.input_x_n)
                level = 0
                self.input_level.append(level)

                num_nodes = len(x_n) + 1
                in_deg = self.get_in_deg(dst, num_nodes)
                src_list, dst_list = src.tolist(), dst.tolist()
                x_n_list = x_n.tolist()
                out_adj_list = self.get_out_adj_list(src_list, dst_list)
                in_adj_list = self.get_in_adj_list(src_list, dst_list)

                frontiers = [u for u in range(1, num_nodes) if in_deg[u] == 0]
                frontier_size = len(frontiers)
                
                while frontier_size > 0:
                    level += 1
                    self.input_e_start.append(input_e_start)
                    self.input_e_end.append(input_e_end)
                    self.input_n_start.append(input_n_start)
                    self.input_n_end.append(input_n_end)
                    if conditional:
                        self.input_g.append(input_g)
                    
                    self.label_start.append(label_start)
                    label_end = label_start + frontier_size
                    self.label_end.append(label_end)
                    label_start = label_end

                    next_frontiers = []
                    for u in frontiers:
                        self.input_x_n.append(x_n_list[u - 1])
                        self.input_level.append(level)
                        for t in in_adj_list[u]:
                            self.input_src.append(t)
                            self.input_dst.append(u)
                            input_e_end += 1
                        for v in out_adj_list[u]:
                            in_deg[v] -= 1
                            if in_deg[v] == 0:
                                next_frontiers.append(v)
                    input_n_end += frontier_size
                    frontiers = next_frontiers
                    frontier_size = len(frontiers)
            except Exception as e:
                print(f"Error in NodePredDataset index {i}: {e}")
                continue

        self.base_postprocess()
        self.label_start = torch.LongTensor(self.label_start)
        self.label_end = torch.LongTensor(self.label_end)

        # --- FIX: Robust Marginal Calculation for BLIF/Sparse types ---
        if get_marginal:
            input_x_n = self.input_x_n
            if input_x_n.ndim == 1:
                input_x_n = input_x_n.unsqueeze(-1)

            num_feats = input_x_n.shape[-1]
            x_n_marginal = []
            
            # 使用 dag_dataset.num_categories 来确定类别总数
            # 如果没有定义，回退到数据推断，但避免 assertion error
            total_categories = getattr(dag_dataset, 'num_categories', None)
            dummy_val = getattr(dag_dataset, 'dummy_category', -1)

            for f in range(num_feats):
                input_x_n_f = input_x_n[:, f]
                unique_x_n_f, x_n_count_f = input_x_n_f.unique(return_counts=True)
                
                if total_categories is None:
                    # Fallback logic: assume dense or handle sparse as best effort
                     num_x_n_types_f = unique_x_n_f.max().item() + 1
                else:
                    # 使用定义的总类别数（不包括 Dummy，或者 Dummy 是最后一个）
                    # 这里的逻辑假设 Dummy 是最后一个，Marginal 只需要前面的真实类别
                    num_x_n_types_f = total_categories - 1 if total_categories > 0 else 1

                x_n_marginal_f = torch.zeros(num_x_n_types_f)

                for c in range(len(x_n_count_f)):
                    x_n_type_f_c = unique_x_n_f[c].item()
                    # Skip dummy category (assuming it's the largest ID or explicitly the dummy val)
                    if x_n_type_f_c != dummy_val and x_n_type_f_c < num_x_n_types_f:
                        x_n_marginal_f[x_n_type_f_c] = x_n_count_f[c].item()

                sum_val = x_n_marginal_f.sum()
                if sum_val > 0:
                    x_n_marginal_f /= sum_val
                else:
                    # Uniform fallback
                    x_n_marginal_f = torch.ones(num_x_n_types_f) / num_x_n_types_f
                    
                x_n_marginal.append(x_n_marginal_f)

            self.x_n_marginal = x_n_marginal

    def __len__(self):
        return len(self.label_start)

    def __getitem__(self, index):
        input_e_start, input_e_end = self.input_e_start[index], self.input_e_end[index]
        input_n_start, input_n_end = self.input_n_start[index], self.input_n_end[index]
        label_start, label_end = self.label_start[index], self.label_end[index]
        input_abs_level = self.input_level[input_n_start:input_n_end]
        input_rel_level = input_abs_level.max() - input_abs_level if input_abs_level.numel() > 0 else input_abs_level

        z = self.input_x_n[label_start:label_end]
        # Ensure z is not empty before applying noise
        if z.numel() > 0:
            t, z_t = self.node_diffusion.apply_noise(z)
        else:
            t, z_t = torch.tensor(0), z

        if self.conditional:
            input_g = self.input_g[index]
            input_y = self.input_y[input_g].item() if self.input_y.ndim > 1 else self.input_y[input_g]
            return self.input_src[input_e_start:input_e_end], self.input_dst[input_e_start:input_e_end], self.input_x_n[input_n_start:input_n_end], input_abs_level, input_rel_level, z_t, t, input_y, z
        else:
            return self.input_src[input_e_start:input_e_end], self.input_dst[input_e_start:input_e_end], self.input_x_n[input_n_start:input_n_end], input_abs_level, input_rel_level, z_t, t, z

class LayerDAGEdgePredDataset(LayerDAGBaseDataset):
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.query_src = []
        self.query_dst = []
        self.query_start = []
        self.query_end = []
        self.label = []

        for i in range(len(dag_dataset)):
            try:
                data_i = dag_dataset[i]
                if conditional:
                    src, dst, x_n, y = data_i
                    input_g = len(self.input_y)
                    self.input_y.append(y)
                else:
                    src, dst, x_n = data_i

                input_n_start = len(self.input_x_n)
                input_n_end = len(self.input_x_n)
                input_e_start = len(self.input_src)
                input_e_end = len(self.input_src)
                query_start = len(self.query_src)
                query_end = len(self.query_src)

                self.input_x_n.append(dag_dataset.dummy_category)
                input_n_end += 1
                src = src + 1
                dst = dst + 1
                level = 0
                self.input_level.append(level)

                num_nodes = len(x_n) + 1
                in_deg = self.get_in_deg(dst, num_nodes)
                src_list, dst_list = src.tolist(), dst.tolist()
                x_n_list = x_n.tolist()
                out_adj_list = self.get_out_adj_list(src_list, dst_list)
                in_adj_list = self.get_in_adj_list(src_list, dst_list)

                prev_frontiers = [u for u in range(1, num_nodes) if in_deg[u] == 0]
                current_frontiers = []
                level += 1

                for u in prev_frontiers:
                    self.input_x_n.append(x_n_list[u - 1])
                    self.input_level.append(level)
                    for v in out_adj_list[u]:
                        in_deg[v] -= 1
                        if in_deg[v] == 0:
                            current_frontiers.append(v)
                input_n_end += len(prev_frontiers)
                src_candidates = prev_frontiers

                while len(current_frontiers) > 0:
                    level += 1
                    next_frontiers = []
                    temp_edge_count = 0
                    for u in current_frontiers:
                        self.input_x_n.append(x_n_list[u - 1])
                        self.input_level.append(level)
                        self.query_src.extend(src_candidates)
                        self.query_dst.extend([u] * len(src_candidates))
                        query_end += len(src_candidates)
                        for t in src_candidates:
                            if t in in_adj_list[u]:
                                self.input_src.append(t)
                                self.input_dst.append(u)
                                temp_edge_count += 1
                                self.label.append(1)
                            else:
                                self.label.append(0)
                        for v in out_adj_list[u]:
                            in_deg[v] -= 1
                            if in_deg[v] == 0:
                                next_frontiers.append(v)
                    input_n_end += len(current_frontiers)

                    self.input_e_start.append(input_e_start)
                    self.input_e_end.append(input_e_end)
                    self.input_n_start.append(input_n_start)
                    self.input_n_end.append(input_n_end)
                    if conditional:
                        self.input_g.append(input_g)
                    self.query_start.append(query_start)
                    self.query_end.append(query_end)

                    src_candidates.extend(current_frontiers)
                    prev_frontiers = current_frontiers
                    current_frontiers = next_frontiers
                    input_e_end += temp_edge_count
                    query_start = query_end
            except Exception as e:
                print(f"Error in EdgePredDataset index {i}: {e}")
                continue

        self.base_postprocess()
        self.query_src = torch.LongTensor(self.query_src)
        self.query_dst = torch.LongTensor(self.query_dst)
        self.query_start = torch.LongTensor(self.query_start)
        self.query_end = torch.LongTensor(self.query_end)
        self.label = torch.LongTensor(self.label)

    def __len__(self):
        return len(self.query_start)

    def __getitem__(self, index):
        input_e_start, input_e_end = self.input_e_start[index], self.input_e_end[index]
        input_n_start, input_n_end = self.input_n_start[index], self.input_n_end[index]
        query_start, query_end = self.query_start[index], self.query_end[index]

        input_abs_level = self.input_level[input_n_start:input_n_end]
        input_rel_level = input_abs_level.max() - input_abs_level if input_abs_level.numel() > 0 else input_abs_level

        query_src = self.query_src[query_start:query_end]
        query_dst = self.query_dst[query_start:query_end]
        label = self.label[query_start:query_end]

        unique_src = torch.unique(query_src, sorted=False)
        unique_dst = torch.unique(query_dst, sorted=False)
        
        # Safe reshaping for label_adj
        if unique_dst.numel() > 0 and unique_src.numel() > 0:
            label_adj = label.reshape(len(unique_dst), len(unique_src))
            t, label_t = self.edge_diffusion.apply_noise(label_adj)
            mask = (label_t == 1)
            # Map back to noisy src/dst indices (simplified assumption: label_t structure matches meshgrid)
            # Warning: This assumes apply_noise preserves structure, which is standard
            # But we need noisy_src/dst as vectors. 
            # Original code logic: noisy_src = query_src[mask] is incorrect if mask is 2D and query_src is 1D repeated
            # Correct logic:
            # Reconstruct indices from mask
            row_idx, col_idx = torch.where(mask)
            noisy_dst = unique_dst[row_idx]
            noisy_src = unique_src[col_idx]
        else:
            t = torch.tensor(0)
            noisy_src = torch.LongTensor([])
            noisy_dst = torch.LongTensor([])

        if self.conditional:
            input_g = self.input_g[index]
            input_y = self.input_y[input_g].item() if self.input_y.ndim > 1 else self.input_y[input_g]
            return self.input_src[input_e_start:input_e_end], self.input_dst[input_e_start:input_e_end], noisy_src, noisy_dst, self.input_x_n[input_n_start:input_n_end], input_abs_level, input_rel_level, t, input_y, query_src, query_dst, label
        else:
            return self.input_src[input_e_start:input_e_end], self.input_dst[input_e_start:input_e_end], noisy_src, noisy_dst, self.input_x_n[input_n_start:input_n_end], input_abs_level, input_rel_level, t, query_src, query_dst, label

# Collate functions match the return signatures
def collate_common(src, dst, x_n, abs_level, rel_level):
    num_nodes_cumsum = torch.cumsum(torch.tensor([0] + [len(x_n_i) for x_n_i in x_n]), dim=0)
    batch_size = len(x_n)
    src_list, dst_list = [], []
    for i in range(batch_size):
        offset = num_nodes_cumsum[i]
        if src[i].numel() > 0: src_list.append(src[i] + offset)
        if dst[i].numel() > 0: dst_list.append(dst[i] + offset)
    
    src = torch.cat(src_list) if src_list else torch.LongTensor([])
    dst = torch.cat(dst_list) if dst_list else torch.LongTensor([])
    edge_index = torch.stack([dst, src]) if src.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
    
    x_n = torch.cat(x_n, dim=0).long()
    abs_level = torch.cat(abs_level, dim=0).float().unsqueeze(-1)
    rel_level = torch.cat(rel_level, dim=0).float().unsqueeze(-1)

    nids, gids = [], []
    for i in range(batch_size):
        nids.append(torch.arange(num_nodes_cumsum[i], num_nodes_cumsum[i+1]).long())
        gids.append(torch.ones(num_nodes_cumsum[i+1] - num_nodes_cumsum[i]).fill_(i).long())
    n2g_index = torch.stack([torch.cat(gids), torch.cat(nids)]) if nids else torch.empty((2,0), dtype=torch.long)
    return batch_size, edge_index, x_n, abs_level, rel_level, n2g_index

def collate_node_count(data):
    data = [d for d in data if d is not None]
    if not data: return None
    if len(data[0]) == 7:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_y, batch_label = map(list, zip(*data))
        y_list = []
        for i, x_n_i in enumerate(batch_x_n):
            y_val = batch_y[i]
            y_list.extend([y_val] * len(x_n_i))
        batch_y = torch.tensor(y_list).unsqueeze(-1) if not isinstance(y_list[0], torch.Tensor) else torch.stack(y_list) # Simple fix logic
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_label = map(list, zip(*data))
        batch_y = None

    batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level, batch_n2g_index = collate_common(batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)
    batch_label = torch.stack(batch_label)
    
    if batch_y is not None:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level, batch_y, batch_n2g_index, batch_label
    else:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level, batch_n2g_index, batch_label

def collate_node_pred(data):
    data = [d for d in data if d is not None]
    if not data: return None
    if len(data[0]) == 8:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_z_t, batch_t, batch_z = map(list, zip(*data))
        batch_y = None
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_z_t, batch_t, batch_y, batch_z = map(list, zip(*data))
        y_list = []
        for i, x_n_i in enumerate(batch_x_n):
            y_list.extend([batch_y[i]] * len(x_n_i))
        batch_y = torch.tensor(y_list).unsqueeze(-1)

    batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level, batch_n2g_index = collate_common(batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level)

    num_query_cumsum = torch.cumsum(torch.tensor([0] + [len(z_t_i) for z_t_i in batch_z_t]), dim=0)
    query2g = []
    for i in range(batch_size):
        query2g.append(torch.ones(num_query_cumsum[i+1] - num_query_cumsum[i]).fill_(i).long())
    query2g = torch.cat(query2g) if query2g else torch.LongTensor([])

    batch_z_t = torch.cat(batch_z_t)
    batch_t = torch.cat(batch_t).unsqueeze(-1)
    batch_z = torch.cat(batch_z)
    if batch_z.ndim == 1: batch_z = batch_z.unsqueeze(-1)

    if batch_y is None:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level, batch_n2g_index, batch_z_t, batch_t, query2g, num_query_cumsum, batch_z
    else:
        return batch_size, batch_edge_index, batch_x_n, batch_abs_level, batch_rel_level, batch_n2g_index, batch_z_t, batch_t, batch_y, query2g, num_query_cumsum, batch_z

def collate_edge_pred(data):
    data = [d for d in data if d is not None]
    if not data: return None
    if len(data[0]) == 11:
        batch_src, batch_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_t, batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        batch_y = None
    else:
        batch_src, batch_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_t, batch_y, batch_query_src, batch_query_dst, batch_label = map(list, zip(*data))
        y_list = []
        for i, x_n_i in enumerate(batch_x_n):
            y_list.extend([batch_y[i]] * len(x_n_i))
        batch_y = torch.tensor(y_list).unsqueeze(-1)

    num_nodes_cumsum = torch.cumsum(torch.tensor([0] + [len(x_n_i) for x_n_i in batch_x_n]), dim=0)
    batch_size = len(batch_x_n)
    src_list, dst_list = [], []
    noisy_src_list, noisy_dst_list = [], []
    query_src_list, query_dst_list = [], []
    t_list = []

    for i in range(batch_size):
        offset = num_nodes_cumsum[i]
        if batch_src[i].numel(): src_list.append(batch_src[i] + offset)
        if batch_dst[i].numel(): dst_list.append(batch_dst[i] + offset)
        if batch_noisy_src[i].numel(): noisy_src_list.append(batch_noisy_src[i] + offset)
        if batch_noisy_dst[i].numel(): noisy_dst_list.append(batch_noisy_dst[i] + offset)
        if batch_query_src[i].numel(): query_src_list.append(batch_query_src[i] + offset)
        if batch_query_dst[i].numel(): query_dst_list.append(batch_query_dst[i] + offset)
        
        num_q = len(batch_query_src[i])
        t_val = batch_t[i]
        if t_val.numel() == 1:
            t_list.append(t_val.expand(num_q, -1))
        else:
            t_list.append(t_val) # Assuming it's already sized or scalar

    src = torch.cat(src_list) if src_list else torch.LongTensor([])
    dst = torch.cat(dst_list) if dst_list else torch.LongTensor([])
    edge_index = torch.stack([dst, src]) if src.numel() > 0 else torch.empty((2,0), dtype=torch.long)

    noisy_src = torch.cat(noisy_src_list) if noisy_src_list else torch.LongTensor([])
    noisy_dst = torch.cat(noisy_dst_list) if noisy_dst_list else torch.LongTensor([])
    noisy_edge_index = torch.stack([noisy_dst, noisy_src]) if noisy_src.numel() > 0 else torch.empty((2,0), dtype=torch.long)

    query_src = torch.cat(query_src_list) if query_src_list else torch.LongTensor([])
    query_dst = torch.cat(query_dst_list) if query_dst_list else torch.LongTensor([])
    t = torch.cat(t_list) if t_list else torch.LongTensor([])

    batch_x_n = torch.cat(batch_x_n, dim=0).long()
    batch_abs_level = torch.cat(batch_abs_level, dim=0).float().unsqueeze(-1)
    batch_rel_level = torch.cat(batch_rel_level, dim=0).float().unsqueeze(-1)
    batch_label = torch.cat(batch_label)

    if batch_y is None:
        return edge_index, noisy_edge_index, batch_x_n, batch_abs_level, batch_rel_level, t, query_src, query_dst, batch_label
    else:
        return edge_index, noisy_edge_index, batch_x_n, batch_abs_level, batch_rel_level, t, batch_y, query_src, query_dst, batch_label