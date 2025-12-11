import torch

from collections import defaultdict
from torch.utils.data import Dataset

__all__ = ['LayerDAGNodeCountDataset',
           'LayerDAGNodePredDataset',
           'LayerDAGEdgePredDataset',
           'collate_node_count',
           'collate_node_pred',
           'collate_edge_pred']

# ... (Helper function remap_indices remains the same) ...
def remap_indices(src, dst, node_indices_global):
# ... (代码保持不变) ...
    """Remaps global edge indices to local indices based on the node subset."""
    if src.numel() == 0: # Handle empty edges
        return src, dst
    node_indices_global_cpu = node_indices_global.cpu() if isinstance(node_indices_global, torch.Tensor) else node_indices_global
    global_to_local_map = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(node_indices_global_cpu)}
    src_cpu = src.cpu() if isinstance(src, torch.Tensor) and src.is_cuda else src
    dst_cpu = dst.cpu() if isinstance(dst, torch.Tensor) and dst.is_cuda else dst
    local_src_list = [global_to_local_map.get(s.item(), -1) for s in src_cpu]
    local_dst_list = [global_to_local_map.get(d.item(), -1) for d in dst_cpu]
    local_src = torch.tensor(local_src_list, dtype=torch.long)
    local_dst = torch.tensor(local_dst_list, dtype=torch.long)
    valid_edge_mask = (local_src != -1) & (local_dst != -1)
    return local_src[valid_edge_mask], local_dst[valid_edge_mask]

# ... (LayerDAGBaseDataset remains the same) ...
class LayerDAGBaseDataset(Dataset):
# ... (代码保持不变) ...
    def __init__(self, conditional=False):
        self.input_src = []
        self.input_dst = []
        self.input_x_n = [] # Can be list of ints, list of tensors, or tensor after postprocess
        self.input_level = []
        self.input_e_start = []
        self.input_e_end = []
        self.input_n_start = []
        self.input_n_end = []
        self.conditional = conditional
        if conditional:
            self.input_y = []
            self.input_g = [] # Stores index mapping y for each processing step
    def get_in_deg(self, dst, num_nodes):
        dst_cpu = dst.cpu() if isinstance(dst, torch.Tensor) and dst.is_cuda else dst
        minlength_val = max(1, num_nodes)
        if isinstance(dst_cpu, torch.Tensor) and dst_cpu.numel() == 0:
             return [0] * minlength_val
        elif isinstance(dst_cpu, list) and not dst_cpu:
             return [0] * minlength_val
        try:
             max_idx = -1
             if isinstance(dst_cpu, torch.Tensor) and dst_cpu.numel() > 0:
                  max_idx = dst_cpu.max().item()
             elif isinstance(dst_cpu, list) and dst_cpu:
                   max_idx = max(dst_cpu)
             if max_idx >= minlength_val:
                      minlength_val = max_idx + 1
             if isinstance(dst_cpu, list):
                  dst_tensor = torch.tensor(dst_cpu, dtype=torch.long)
             else:
                  dst_tensor = dst_cpu
             return torch.bincount(dst_tensor, minlength=minlength_val).tolist()
        except RuntimeError as e:
             max_val_str = f"{dst_cpu.max().item()}" if isinstance(dst_cpu, torch.Tensor) and dst_cpu.numel() > 0 else 'N/A'
             return [0] * minlength_val
    def get_out_adj_list(self, src, dst):
        out_adj_list = defaultdict(list)
        src_list = src.tolist() if isinstance(src, torch.Tensor) else src
        dst_list = dst.tolist() if isinstance(dst, torch.Tensor) else dst
        num_edges = len(src_list)
        for i in range(num_edges):
            out_adj_list[int(src_list[i])].append(int(dst_list[i]))
        return out_adj_list
    def get_in_adj_list(self, src, dst):
        in_adj_list = defaultdict(list)
        src_list = src.tolist() if isinstance(src, torch.Tensor) else src
        dst_list = dst.tolist() if isinstance(dst, torch.Tensor) else dst
        num_edges = len(src_list)
        for i in range(num_edges):
             in_adj_list[int(dst_list[i])].append(int(src_list[i]))
        return in_adj_list
    def base_postprocess(self):
        try:
            self.input_src = torch.LongTensor(self.input_src) if self.input_src else torch.LongTensor([])
            self.input_dst = torch.LongTensor(self.input_dst) if self.input_dst else torch.LongTensor([])
            if self.input_x_n:
                if all(isinstance(x, torch.Tensor) for x in self.input_x_n):
                    try: self.input_x_n = torch.stack(self.input_x_n).long()
                    except RuntimeError: pass # Keep as list
                elif all(isinstance(x, (int, float)) for x in self.input_x_n): self.input_x_n = torch.LongTensor(self.input_x_n)
            self.input_level = torch.LongTensor(self.input_level) if self.input_level else torch.LongTensor([])
            self.input_e_start = torch.LongTensor(self.input_e_start) if self.input_e_start else torch.LongTensor([])
            self.input_e_end = torch.LongTensor(self.input_e_end) if self.input_e_end else torch.LongTensor([])
            self.input_n_start = torch.LongTensor(self.input_n_start) if self.input_n_start else torch.LongTensor([])
            self.input_n_end = torch.LongTensor(self.input_n_end) if self.input_n_end else torch.LongTensor([])
            if self.conditional: self.input_g = torch.LongTensor(self.input_g) if self.input_g else torch.LongTensor([])
        except Exception as e:
            print(f"Error during base_postprocess: {e}. Data might be inconsistent.")
            if not isinstance(self.input_src, torch.Tensor): self.input_src = torch.LongTensor([])
            if not isinstance(self.input_dst, torch.Tensor): self.input_dst = torch.LongTensor([])
            if not isinstance(self.input_x_n, (torch.Tensor, list)): self.input_x_n = []
            if not isinstance(self.input_level, torch.Tensor): self.input_level = torch.LongTensor([])
            if not isinstance(self.input_e_start, torch.Tensor): self.input_e_start = torch.LongTensor([])
            if not isinstance(self.input_e_end, torch.Tensor): self.input_e_end = torch.LongTensor([])
            if not isinstance(self.input_n_start, torch.Tensor): self.input_n_start = torch.LongTensor([])
            if not isinstance(self.input_n_end, torch.Tensor): self.input_n_end = torch.LongTensor([])
            if self.conditional and not isinstance(self.input_g, torch.Tensor): self.input_g = torch.LongTensor([])

# ... (LayerDAGNodeCountDataset remains the same) ...
class LayerDAGNodeCountDataset(LayerDAGBaseDataset):
# ... (代码保持不变) ...
    def __init__(self, dag_dataset, conditional=False):
        super().__init__(conditional)
        self.label = []
        for i in range(len(dag_dataset)):
            try:
                data_i = dag_dataset[i];
                if not data_i: continue
                if conditional:
                    if len(data_i) != 4: continue
                    src, dst, x_n, y = data_i;
                    if not isinstance(y, torch.Tensor): y = torch.tensor(y)
                    input_g = len(self.input_y); self.input_y.append(y)
                else:
                    if len(data_i) != 3: continue
                    src, dst, x_n = data_i; input_g = -1
                if not (isinstance(src, torch.Tensor) and src.dtype == torch.long and isinstance(dst, torch.Tensor) and dst.dtype == torch.long and isinstance(x_n, torch.Tensor) and x_n.dtype == torch.long): continue
                current_graph_nodes, current_graph_levels, current_graph_src, current_graph_dst = [], [], [], []
                current_n_start_global, current_e_start_global = len(self.input_x_n), len(self.input_src)
                dummy_val = dag_dataset.dummy_category;
                if isinstance(dummy_val, torch.Tensor): dummy_val = dummy_val.item()
                feature_dim = x_n.shape[1] if x_n.ndim > 1 and x_n.shape[0]>0 else 1
                if feature_dim > 1: dummy_feature = torch.full((feature_dim,), int(dummy_val), dtype=torch.long)
                else: dummy_feature = torch.tensor(int(dummy_val), dtype=torch.long)
                current_graph_nodes.append(dummy_feature); current_graph_levels.append(0)
                num_orig_nodes = len(x_n); orig_src, orig_dst = src.clone() + 1, dst.clone() + 1
                orig_num_nodes_shifted = num_orig_nodes + 1; orig_x_n_list = x_n.tolist()
                if orig_num_nodes_shifted <= 1 and orig_src.numel() > 0: continue
                try: in_deg = self.get_in_deg(orig_dst, orig_num_nodes_shifted)
                except Exception as e: continue
                out_adj_list, in_adj_list = self.get_out_adj_list(orig_src, orig_dst), self.get_in_adj_list(orig_src, orig_dst)
                frontiers = [u for u in range(1, orig_num_nodes_shifted) if in_deg[u] == 0]
                level = 0; node_map_orig_to_cumulative = {0: current_n_start_global}
                while frontiers:
                    nodes_processed_count = sum(len(n) if isinstance(n, list) else 1 for n in current_graph_nodes)
                    current_nodes_offset = len(self.input_x_n) - nodes_processed_count
                    self.input_n_start.append(current_n_start_global)
                    self.input_n_end.append(len(self.input_x_n))
                    self.input_e_start.append(current_e_start_global)
                    self.input_e_end.append(len(self.input_src))
                    if conditional: self.input_g.append(input_g)
                    self.label.append(len(frontiers))
                    level += 1; next_frontiers_set = set()
                    new_nodes_in_layer, new_levels_in_layer = [], []
                    new_edges_src_global, new_edges_dst_global = [], []
                    current_global_node_idx_base = len(self.input_x_n)
                    for idx_in_layer, u_orig in enumerate(frontiers):
                         current_node_global_idx = current_global_node_idx_base + idx_in_layer
                         node_map_orig_to_cumulative[u_orig] = current_node_global_idx
                         orig_node_feature_index = u_orig - 1
                         if 0 <= orig_node_feature_index < len(orig_x_n_list):
                             node_feat = orig_x_n_list[orig_node_feature_index]
                             if feature_dim > 1 and not isinstance(node_feat, torch.Tensor): node_feat = torch.tensor(node_feat, dtype=torch.long)
                             elif feature_dim == 1 and isinstance(node_feat, list): node_feat = node_feat[0]
                             new_nodes_in_layer.append(node_feat); new_levels_in_layer.append(level)
                         else: continue
                         if u_orig in in_adj_list:
                             for t_orig in in_adj_list[u_orig]:
                                 if t_orig in node_map_orig_to_cumulative: t_global = node_map_orig_to_cumulative[t_orig]; new_edges_src_global.append(t_global); new_edges_dst_global.append(current_node_global_idx)
                         if u_orig in out_adj_list:
                             for v_orig in out_adj_list[u_orig]:
                                 if v_orig < len(in_deg): in_deg[v_orig] -= 1;
                                 if in_deg[v_orig] == 0: next_frontiers_set.add(v_orig)
                    self.input_x_n.extend(new_nodes_in_layer); self.input_level.extend(new_levels_in_layer)
                    self.input_src.extend(new_edges_src_global); self.input_dst.extend(new_edges_dst_global)
                    current_graph_nodes.extend(new_nodes_in_layer)
                    frontiers = sorted(list(next_frontiers_set))
                self.input_n_start.append(current_n_start_global); self.input_n_end.append(len(self.input_x_n))
                self.input_e_start.append(current_e_start_global); self.input_e_end.append(len(self.input_src))
                if conditional: self.input_g.append(input_g)
                self.label.append(0)
            except Exception as e: print(f"Error processing item {i} in NodeCountDataset __init__: {e}. Skipping.")
        self.base_postprocess()
        self.label = torch.LongTensor(self.label) if self.label else torch.LongTensor([])
        self.max_layer_size = self.label.max().item() if self.label.numel() > 0 else 0
    def __len__(self): return len(self.label)
    def __getitem__(self, index):
# ... (代码保持不变) ...
        if not (0 <= index < len(self.label)):
             raise IndexError(f"Index {index} out of bounds...")
        input_n_start_global = self.input_n_start[index]; input_n_end_global = self.input_n_end[index]
        input_e_start_global = self.input_e_start[index]; input_e_end_global = self.input_e_end[index]
        slice_src_global = self.input_src[input_e_start_global:input_e_end_global]; slice_dst_global = self.input_dst[input_e_start_global:input_e_end_global]
        if isinstance(self.input_x_n, torch.Tensor): slice_x_n = self.input_x_n[input_n_start_global:input_n_end_global]
        else:
             items_in_slice = self.input_x_n[input_n_start_global:input_n_end_global]
             if items_in_slice:
                  if all(isinstance(x, torch.Tensor) for x in items_in_slice):
                      try: slice_x_n = torch.stack(items_in_slice).long()
                      except RuntimeError: slice_x_n = items_in_slice[0].unsqueeze(0).long()
                  elif all(isinstance(x, (int, float)) for x in items_in_slice): slice_x_n = torch.LongTensor(items_in_slice)
                  else: slice_x_n = torch.LongTensor([])
             else: slice_x_n = torch.LongTensor([])
        slice_level = self.input_level[input_n_start_global:input_n_end_global]
        num_nodes_in_slice = input_n_end_global - input_n_start_global
        if num_nodes_in_slice > 0: node_indices_global_in_slice = torch.arange(input_n_start_global, input_n_end_global); slice_src_local, slice_dst_local = remap_indices(slice_src_global, slice_dst_global, node_indices_global_in_slice)
        else: slice_src_local, slice_dst_local = torch.LongTensor([]), torch.LongTensor([])
        if slice_level.numel() == 0: input_rel_level = slice_level
        else: input_rel_level = slice_level.max() - slice_level
        if self.conditional:
            input_g_idx = -1
            if not (0 <= index < len(self.input_g)):
                 raise IndexError(f"Error in NodeCount __getitem__({index}): index out of bounds for self.input_g (len={len(self.input_g)})")
            input_g_idx = self.input_g[index].item()
            if not (0 <= input_g_idx < len(self.input_y)):
                 raise IndexError(f"Error in NodeCount __getitem__({index}): Retrieved graph index {input_g_idx} out of bounds for self.input_y (len={len(self.input_y)})")
            input_y_tensor = self.input_y[input_g_idx]
            return slice_src_local, slice_dst_local, slice_x_n, slice_level, input_rel_level, input_y_tensor, self.label[index]
        else:
            return slice_src_local, slice_dst_local, slice_x_n, slice_level, input_rel_level, self.label[index]

# ... (LayerDAGNodePredDataset remains the same) ...
class LayerDAGNodePredDataset(LayerDAGBaseDataset):
# ... (代码保持不变) ...
    def __init__(self, dag_dataset, conditional=False, get_marginal=True):
        super().__init__(conditional)
        self.label_start = [] # Global start index of predicted nodes in self.input_x_n
        self.label_end = []   # Global end index
        for i in range(len(dag_dataset)):
            try:
                data_i = dag_dataset[i];
                if not data_i: continue
                if conditional:
                    if len(data_i) != 4: continue
                    src, dst, x_n, y = data_i; input_g = len(self.input_y); self.input_y.append(y)
                else:
                    if len(data_i) != 3: continue
                    src, dst, x_n = data_i; input_g = -1
                if not (isinstance(src, torch.Tensor) and src.dtype == torch.long and isinstance(dst, torch.Tensor) and dst.dtype == torch.long and isinstance(x_n, torch.Tensor) and x_n.dtype == torch.long): continue
                current_graph_nodes, current_graph_levels, current_graph_src, current_graph_dst = [], [], [], []
                current_n_start_global, current_e_start_global = len(self.input_x_n), len(self.input_src)
                dummy_val = dag_dataset.dummy_category;
                if isinstance(dummy_val, torch.Tensor): dummy_val = dummy_val.item()
                feature_dim = x_n.shape[1] if x_n.ndim > 1 and x_n.shape[0]>0 else 1
                if feature_dim > 1: dummy_feature = torch.full((feature_dim,), int(dummy_val), dtype=torch.long)
                else: dummy_feature = torch.tensor(int(dummy_val), dtype=torch.long)
                current_graph_nodes.append(dummy_feature); current_graph_levels.append(0)
                num_orig_nodes = len(x_n); orig_src, orig_dst = src.clone() + 1, dst.clone() + 1
                orig_num_nodes_shifted = num_orig_nodes + 1; orig_x_n_list = x_n.tolist()
                if orig_num_nodes_shifted <= 1 and orig_src.numel() > 0: continue
                try: in_deg = self.get_in_deg(orig_dst, orig_num_nodes_shifted)
                except Exception as e: continue
                out_adj_list, in_adj_list = self.get_out_adj_list(orig_src, orig_dst), self.get_in_adj_list(orig_src, orig_dst)
                frontiers = [u for u in range(1, orig_num_nodes_shifted) if in_deg[u] == 0]
                level = 0; node_map_orig_to_cumulative = {0: current_n_start_global}
                while frontiers:
                    self.input_n_start.append(current_n_start_global)
                    self.input_n_end.append(len(self.input_x_n))
                    self.input_e_start.append(current_e_start_global)
                    self.input_e_end.append(len(self.input_src))
                    if conditional: self.input_g.append(input_g)
                    self.label_start.append(len(self.input_x_n))
                    level += 1; next_frontiers_set = set()
                    new_nodes_in_layer, new_levels_in_layer = [], []
                    new_edges_src_global, new_edges_dst_global = [], []
                    current_global_node_idx_base = len(self.input_x_n)
                    for idx_in_layer, u_orig in enumerate(frontiers):
                         current_node_global_idx = current_global_node_idx_base + idx_in_layer
                         node_map_orig_to_cumulative[u_orig] = current_node_global_idx
                         orig_node_feature_index = u_orig - 1
                         if 0 <= orig_node_feature_index < len(orig_x_n_list):
                             node_feat = orig_x_n_list[orig_node_feature_index]
                             if feature_dim > 1 and not isinstance(node_feat, torch.Tensor): node_feat = torch.tensor(node_feat, dtype=torch.long)
                             elif feature_dim == 1 and isinstance(node_feat, list): node_feat = node_feat[0]
                             new_nodes_in_layer.append(node_feat); new_levels_in_layer.append(level)
                         else: continue
                         if u_orig in in_adj_list:
                             for t_orig in in_adj_list[u_orig]:
                                 if t_orig in node_map_orig_to_cumulative: t_global = node_map_orig_to_cumulative[t_orig]; new_edges_src_global.append(t_global); new_edges_dst_global.append(current_node_global_idx)
                         if u_orig in out_adj_list:
                             for v_orig in out_adj_list[u_orig]:
                                 if v_orig < len(in_deg): in_deg[v_orig] -= 1;
                                 if in_deg[v_orig] == 0: next_frontiers_set.add(v_orig)
                    self.input_x_n.extend(new_nodes_in_layer); self.input_level.extend(new_levels_in_layer)
                    self.input_src.extend(new_edges_src_global); self.input_dst.extend(new_edges_dst_global)
                    self.label_end.append(len(self.input_x_n))
                    frontiers = sorted(list(next_frontiers_set))
            except Exception as e: print(f"Error processing item {i} in NodePredDataset __init__: {e}. Skipping.")
        self.base_postprocess()
        self.label_start = torch.LongTensor(self.label_start) if self.label_start else torch.LongTensor([])
        self.label_end = torch.LongTensor(self.label_end) if self.label_end else torch.LongTensor([])
        if get_marginal:
            all_node_features_flat = []
            if isinstance(self.input_x_n, torch.Tensor): input_x_n_tensor = self.input_x_n
            elif isinstance(self.input_x_n, list):
                 if self.input_x_n:
                      if all(isinstance(x, (int,float)) for x in self.input_x_n): input_x_n_tensor = torch.LongTensor(self.input_x_n)
                      elif all(isinstance(x, torch.Tensor) for x in self.input_x_n):
                           try: input_x_n_tensor = torch.stack(self.input_x_n).long()
                           except: input_x_n_tensor = torch.LongTensor([])
                      else: input_x_n_tensor = torch.LongTensor([])
                 else: input_x_n_tensor = torch.LongTensor([])
            else: input_x_n_tensor = torch.LongTensor([])
            if input_x_n_tensor.numel() > 0:
                if input_x_n_tensor.ndim == 1: input_x_n_tensor = input_x_n_tensor.unsqueeze(-1)
                num_feats = input_x_n_tensor.shape[-1]; x_n_marginal = []
                dummy_category_val = dag_dataset.dummy_category;
                if isinstance(dummy_category_val, torch.Tensor): dummy_category_val = dummy_category_val.item()
                dummy_category_val = int(dummy_category_val)
                expected_num_real = dag_dataset.num_categories - 1
                for f in range(num_feats):
                    input_x_n_f = input_x_n_tensor[:, f]
                    real_nodes_mask = (input_x_n_f != dummy_category_val)
                    if not real_nodes_mask.any(): x_n_marginal_f = torch.ones(expected_num_real) / expected_num_real
                    else:
                         input_x_n_f_real = input_x_n_f[real_nodes_mask]; unique_x_n_f, x_n_count_f = input_x_n_f_real.unique(return_counts=True)
                         x_n_marginal_f = torch.zeros(expected_num_real)
                         valid_scatter_indices = unique_x_n_f[unique_x_n_f < expected_num_real]
                         valid_counts = x_n_count_f[unique_x_n_f < expected_num_real]
                         if valid_scatter_indices.numel() > 0: x_n_marginal_f.scatter_(0, valid_scatter_indices, valid_counts.float())
                         total_real_nodes = x_n_marginal_f.sum()
                         if total_real_nodes > 0: x_n_marginal_f /= total_real_nodes
                         else: x_n_marginal_f = torch.ones(expected_num_real) / expected_num_real
                    x_n_marginal.append(x_n_marginal_f)
                self.x_n_marginal = x_n_marginal
            else: print("Warning: No node features found to calculate marginal distribution."); self.x_n_marginal = []
    def __len__(self): return len(self.label_start)
    def __getitem__(self, index):
# ... (代码保持不变) ...
        if not (0 <= index < len(self.label_start)): raise IndexError("...")
        input_n_start_global = self.input_n_start[index]; input_n_end_global = self.input_n_end[index]
        input_e_start_global = self.input_e_start[index]; input_e_end_global = self.input_e_end[index]
        slice_src_global = self.input_src[input_e_start_global:input_e_end_global]; slice_dst_global = self.input_dst[input_e_start_global:input_e_end_global]
        if isinstance(self.input_x_n, torch.Tensor): slice_x_n = self.input_x_n[input_n_start_global:input_n_end_global]
        else:
             items_in_slice = self.input_x_n[input_n_start_global:input_n_end_global]
             try: slice_x_n = torch.stack(items_in_slice).long() if items_in_slice else torch.LongTensor([])
             except: slice_x_n = torch.LongTensor([])
        slice_level = self.input_level[input_n_start_global:input_n_end_global]
        num_nodes_in_slice = input_n_end_global - input_n_start_global
        if num_nodes_in_slice > 0: node_indices_global_in_slice = torch.arange(input_n_start_global, input_n_end_global); slice_src_local, slice_dst_local = remap_indices(slice_src_global, slice_dst_global, node_indices_global_in_slice)
        else: slice_src_local, slice_dst_local = torch.LongTensor([]), torch.LongTensor([])
        if slice_level.numel() == 0: input_rel_level = slice_level
        else: input_rel_level = slice_level.max() - slice_level
        label_start_global = self.label_start[index]; label_end_global = self.label_end[index]
        if isinstance(self.input_x_n, torch.Tensor): z = self.input_x_n[label_start_global:label_end_global]
        else:
             items_in_z = self.input_x_n[label_start_global:label_end_global]
             try: z = torch.stack(items_in_z).long() if items_in_z else torch.LongTensor([])
             except: z = torch.LongTensor([])
        if not hasattr(self, 'node_diffusion') or self.node_diffusion is None: raise RuntimeError("Node diffusion model not initialized")
        if z.numel() == 0: t, z_t = torch.zeros(1, dtype=torch.long), torch.LongTensor([])
        else:
             try:
                 t_scalar, z_t = self.node_diffusion.apply_noise(z); t = torch.tensor([t_scalar], dtype=torch.long)
             except IndexError as e:
                 print(f"Error applying noise in NodePred __getitem__({index}): {e}. z shape: {z.shape}")
                 t, z_t = torch.zeros(1, dtype=torch.long), torch.LongTensor([])
        if self.conditional:
            input_g_idx = -1
            if not (0 <= index < len(self.input_g)):
                 raise IndexError(f"Error in NodePred __getitem__({index}): index out of bounds for self.input_g (len={len(self.input_g)})")
            input_g_idx = self.input_g[index].item()
            if not (0 <= input_g_idx < len(self.input_y)):
                 raise IndexError(f"Error in NodePred __getitem__({index}): Retrieved graph index {input_g_idx} out of bounds for self.input_y (len={len(self.input_y)})")
            input_y_tensor = self.input_y[input_g_idx]
            return slice_src_local, slice_dst_local, slice_x_n, slice_level, input_rel_level, z_t, t, input_y_tensor, z
        else:
            return slice_src_local, slice_dst_local, slice_x_n, slice_level, input_rel_level, z_t, t, z

# ... (LayerDAGEdgePredDataset remains the same) ...
class LayerDAGEdgePredDataset(LayerDAGBaseDataset):
# ... (代码保持不变) ...
    def __init__(self, dag_dataset, conditional=False):
         super().__init__(conditional)
         self.query_src = [] # Global indices
         self.query_dst = [] # Global indices
         self.query_start = [] # Start index in the query lists for each step
         self.query_end = []   # End index
         self.label = []       # Edge labels (0 or 1) corresponding to queries
         num_edges = 0
         num_nonsrc_nodes = 0
         for i in range(len(dag_dataset)):
            try:
                data_i = dag_dataset[i]
                if not data_i: continue
                if conditional:
                    if len(data_i) != 4: continue
                    src, dst, x_n, y = data_i; input_g = len(self.input_y); self.input_y.append(y)
                else:
                    if len(data_i) != 3: continue
                    src, dst, x_n = data_i; input_g = -1
                if not (isinstance(src, torch.Tensor) and src.dtype == torch.long and isinstance(dst, torch.Tensor) and dst.dtype == torch.long and isinstance(x_n, torch.Tensor) and x_n.dtype == torch.long): continue
                current_graph_nodes, current_graph_levels = [], []
                current_n_start_global, current_e_start_global = len(self.input_x_n), len(self.input_src)
                dummy_val = dag_dataset.dummy_category
                if isinstance(dummy_val, torch.Tensor): dummy_val = dummy_val.item()
                feature_dim = x_n.shape[1] if x_n.ndim > 1 and x_n.shape[0]>0 else 1
                if feature_dim > 1: dummy_feature = torch.full((feature_dim,), int(dummy_val), dtype=torch.long)
                else: dummy_feature = torch.tensor(int(dummy_val), dtype=torch.long)
                current_graph_nodes.append(dummy_feature); current_graph_levels.append(0)
                num_orig_nodes = len(x_n); orig_src, orig_dst = src.clone() + 1, dst.clone() + 1
                orig_num_nodes_shifted = num_orig_nodes + 1; orig_x_n_list = x_n.tolist()
                if orig_num_nodes_shifted <= 1 and orig_src.numel() > 0: continue
                try: in_deg = self.get_in_deg(orig_dst, orig_num_nodes_shifted)
                except Exception as e: continue
                out_adj_list, in_adj_list = self.get_out_adj_list(orig_src, orig_dst), self.get_in_adj_list(orig_src, orig_dst)
                actual_edges = set((s.item(), d.item()) for s, d in zip(orig_src, orig_dst))
                frontiers = [u for u in range(1, orig_num_nodes_shifted) if in_deg[u] == 0]
                level = 0
                node_map_orig_to_cumulative = {0: current_n_start_global}
                processed_nodes_orig_indices = {0}
                src_candidates_global = [current_n_start_global]
                while frontiers:
                    level += 1
                    next_frontiers_set = set()
                    new_nodes_in_layer, new_levels_in_layer = [], []
                    new_edges_src_global, new_edges_dst_global = [], []
                    query_src_step_global, query_dst_step_global, label_step = [], [], []
                    current_global_node_idx_base = len(self.input_x_n)
                    current_layer_global_indices = []
                    for idx_in_layer, u_orig in enumerate(frontiers):
                         current_node_global_idx = current_global_node_idx_base + idx_in_layer
                         node_map_orig_to_cumulative[u_orig] = current_node_global_idx
                         current_layer_global_indices.append(current_node_global_idx)
                         processed_nodes_orig_indices.add(u_orig)
                         orig_node_feature_index = u_orig - 1
                         if 0 <= orig_node_feature_index < len(orig_x_n_list):
                             node_feat = orig_x_n_list[orig_node_feature_index]
                             if feature_dim > 1 and not isinstance(node_feat, torch.Tensor): node_feat = torch.tensor(node_feat, dtype=torch.long)
                             elif feature_dim == 1 and isinstance(node_feat, list): node_feat = node_feat[0]
                             new_nodes_in_layer.append(node_feat); new_levels_in_layer.append(level)
                         else: continue
                         for dst_node_global_idx in [current_node_global_idx]:
                             for src_node_global_idx in src_candidates_global:
                                 src_orig = -1; dst_orig = u_orig
                                 for orig_idx_lookup, global_idx_lookup in node_map_orig_to_cumulative.items():
                                     if global_idx_lookup == src_node_global_idx: src_orig = orig_idx_lookup; break
                                 if src_orig != -1: query_src_step_global.append(src_node_global_idx); query_dst_step_global.append(dst_node_global_idx); label_step.append(1 if (src_orig, dst_orig) in actual_edges else 0)
                         if u_orig in in_adj_list:
                             for t_orig in in_adj_list[u_orig]:
                                 if t_orig in node_map_orig_to_cumulative: t_global = node_map_orig_to_cumulative[t_orig]; new_edges_src_global.append(t_global); new_edges_dst_global.append(current_node_global_idx)
                         if u_orig in out_adj_list:
                             for v_orig in out_adj_list[u_orig]:
                                 if v_orig < len(in_deg): in_deg[v_orig] -= 1;
                                 if in_deg[v_orig] == 0: next_frontiers_set.add(v_orig)
                    if query_src_step_global:
                        self.input_n_start.append(current_n_start_global)
                        self.input_n_end.append(len(self.input_x_n))
                        self.input_e_start.append(current_e_start_global)
                        self.input_e_end.append(len(self.input_src))
                        if conditional: self.input_g.append(input_g)
                        self.query_start.append(len(self.query_src))
                        self.query_src.extend(query_src_step_global)
                        self.query_dst.extend(query_dst_step_global)
                        self.label.extend(label_step)
                        self.query_end.append(len(self.query_src))
                    self.input_x_n.extend(new_nodes_in_layer); self.input_level.extend(new_levels_in_layer)
                    self.input_src.extend(new_edges_src_global); self.input_dst.extend(new_edges_dst_global)
                    src_candidates_global.extend(current_layer_global_indices)
                    frontiers = sorted(list(next_frontiers_set))
                num_edges += src.numel()
                orig_source_nodes = {u for u in range(1, orig_num_nodes_shifted) if u not in in_adj_list}
                num_nonsrc_nodes += (orig_num_nodes_shifted - 1 - len(orig_source_nodes))
            except Exception as e: print(f"Error processing item {i} in EdgePredDataset __init__: {e}. Skipping.")
         self.base_postprocess()
         self.query_src = torch.LongTensor(self.query_src) if self.query_src else torch.LongTensor([]); self.query_dst = torch.LongTensor(self.query_dst) if self.query_dst else torch.LongTensor([])
         self.query_start = torch.LongTensor(self.query_start) if self.query_start else torch.LongTensor([]); self.query_end = torch.LongTensor(self.query_end) if self.query_end else torch.LongTensor([])
         self.label = torch.LongTensor(self.label) if self.label else torch.LongTensor([])
         self.avg_in_deg = num_edges / num_nonsrc_nodes if num_nonsrc_nodes > 0 else 0
    def __len__(self): return len(self.query_start)
    def __getitem__(self, index):
# ... (代码保持不变) ...
        if not (0 <= index < len(self.query_start)): raise IndexError("...")

        input_n_start_global = self.input_n_start[index]; input_n_end_global = self.input_n_end[index]
        input_e_start_global = self.input_e_start[index]; input_e_end_global = self.input_e_end[index]
        slice_src_global = self.input_src[input_e_start_global:input_e_end_global]; slice_dst_global = self.input_dst[input_e_start_global:input_e_end_global]
        if isinstance(self.input_x_n, torch.Tensor): slice_x_n = self.input_x_n[input_n_start_global:input_n_end_global]
        else:
            items_in_slice = self.input_x_n[input_n_start_global:input_n_end_global]
            try: slice_x_n = torch.stack(items_in_slice).long() if items_in_slice else torch.LongTensor([])
            except: slice_x_n = torch.LongTensor([])
        slice_level = self.input_level[input_n_start_global:input_n_end_global]

        num_nodes_in_slice = input_n_end_global - input_n_start_global
        if num_nodes_in_slice > 0: node_indices_global_in_slice = torch.arange(input_n_start_global, input_n_end_global); slice_src_local, slice_dst_local = remap_indices(slice_src_global, slice_dst_global, node_indices_global_in_slice)
        else: slice_src_local, slice_dst_local = torch.LongTensor([]), torch.LongTensor([])

        if slice_level.numel() == 0: input_rel_level = slice_level
        else: input_rel_level = slice_level.max() - slice_level

        query_start_idx = self.query_start[index]; query_end_idx = self.query_end[index]
        query_src_global = self.query_src[query_start_idx:query_end_idx]; query_dst_global = self.query_dst[query_start_idx:query_end_idx]; label = self.label[query_start_idx:query_end_idx]

        if num_nodes_in_slice > 0: query_src_local, query_dst_local = remap_indices(query_src_global, query_dst_global, node_indices_global_in_slice)
        else: query_src_local, query_dst_local = torch.LongTensor([]), torch.LongTensor([])

        if not hasattr(self, 'edge_diffusion') or self.edge_diffusion is None: raise RuntimeError("...")
        if label.numel() == 0 or query_src_local.numel() == 0 or query_dst_local.numel() == 0: t, noisy_src_local, noisy_dst_local = torch.zeros(1, dtype=torch.long), torch.LongTensor([]), torch.LongTensor([])
        else:
             try:
                 unique_qsrc_local, qsrc_map = torch.unique(query_src_local, return_inverse=True); unique_qdst_local, qdst_map = torch.unique(query_dst_local, return_inverse=True)
                 if unique_qdst_local.numel() > 0 and unique_qsrc_local.numel() > 0:
                     label_adj = torch.zeros(len(unique_qdst_local), len(unique_qsrc_local), dtype=label.dtype); label_adj.index_put_((qdst_map.cpu(), qsrc_map.cpu()), label.cpu())
                     t_scalar, label_t = self.edge_diffusion.apply_noise(label_adj); t = torch.tensor([t_scalar], dtype=torch.long)
                     mask = (label_t == 1)
                     if mask.any(): row_idx, col_idx = torch.where(mask); noisy_dst_local = unique_qdst_local[row_idx]; noisy_src_local = unique_qsrc_local[col_idx]
                     else: noisy_src_local, noisy_dst_local = torch.LongTensor([]), torch.LongTensor([])
                 else: t, noisy_src_local, noisy_dst_local = torch.zeros(1, dtype=torch.long), torch.LongTensor([]), torch.LongTensor([])
             except Exception as e: print(f"Error during noise application in EdgePred item {index}: {e}"); t, noisy_src_local, noisy_dst_local = torch.zeros(1, dtype=torch.long), torch.LongTensor([]), torch.LongTensor([])
        
        # --- MODIFICATION START: Get query destination node features ---
        if query_dst_local.numel() > 0:
            # slice_x_n is all node features, query_dst_local are the indices
            max_idx = query_dst_local.max().item() if query_dst_local.numel() > 0 else 0
            if max_idx >= slice_x_n.shape[0]:
                 print(f"Error: query_dst_local index {max_idx} out of bounds for slice_x_n size {slice_x_n.shape[0]}")
                 query_dst_x_n = torch.empty((0, slice_x_n.shape[-1]), dtype=torch.long)
            else:
                 query_dst_x_n = slice_x_n[query_dst_local]
        else:
            feature_dim = slice_x_n.shape[-1] if slice_x_n.ndim > 1 else 1
            query_dst_x_n = torch.empty((0, feature_dim), dtype=torch.long)
        # --- MODIFICATION END ---
        
        if self.conditional:
            input_g_idx = -1
            if not (0 <= index < len(self.input_g)):
                 raise IndexError(f"Error in EdgePred __getitem__({index}): index out of bounds for self.input_g (len={len(self.input_g)})")
            input_g_idx = self.input_g[index].item()
            if not (0 <= input_g_idx < len(self.input_y)):
                 raise IndexError(f"Error in EdgePred __getitem__({index}): Retrieved graph index {input_g_idx} out of bounds for self.input_y (len={len(self.input_y)})")
            input_y_tensor = self.input_y[input_g_idx]
            # Return 13 items
            return (slice_src_local, slice_dst_local, noisy_src_local, noisy_dst_local, slice_x_n,
                    slice_level, input_rel_level, t, input_y_tensor,
                    query_src_local, query_dst_local, label, query_dst_x_n) # <-- Added query_dst_x_n
        else:
            # Return 10 items
            return (slice_src_local, slice_dst_local, noisy_src_local, noisy_dst_local, slice_x_n,
                    slice_level, input_rel_level, t,
                    query_src_local, query_dst_local, label, query_dst_x_n) # <-- Added query_dst_x_n


# ... (collate_common remains the same) ...
def collate_common(batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level):
# ... (代码保持不变) ...
    valid_indices = [i for i, x in enumerate(batch_x_n) if isinstance(x, torch.Tensor) and x.numel() > 0 and isinstance(batch_src[i], torch.Tensor) and isinstance(batch_dst[i], torch.Tensor) and isinstance(batch_abs_level[i], torch.Tensor) and isinstance(batch_rel_level[i], torch.Tensor)]
    if not valid_indices:
        batch_size = 0; edge_index = torch.empty((2, 0), dtype=torch.long); feature_dim = 1
        orig_batch_x_n = batch_x_n
        if orig_batch_x_n and isinstance(orig_batch_x_n[0], torch.Tensor) and orig_batch_x_n[0].ndim > 1: feature_dim = orig_batch_x_n[0].shape[1]
        elif orig_batch_x_n and isinstance(orig_batch_x_n[0], torch.Tensor) and orig_batch_x_n[0].ndim == 1: feature_dim = 1
        x_n = torch.empty((0, feature_dim), dtype=torch.long); abs_level = torch.empty((0, 1), dtype=torch.float); rel_level = torch.empty((0, 1), dtype=torch.float); n2g_index = torch.empty((2, 0), dtype=torch.long)
        return batch_size, edge_index, [], abs_level, rel_level, n2g_index, []
    batch_src = [batch_src[i] for i in valid_indices]; batch_dst = [batch_dst[i] for i in valid_indices]; batch_x_n = [batch_x_n[i] for i in valid_indices]; batch_abs_level = [batch_abs_level[i] for i in valid_indices]; batch_rel_level = [batch_rel_level[i] for i in valid_indices]
    num_nodes_per_graph = [len(x_n_i) for x_n_i in batch_x_n]; num_nodes_cumsum = torch.cumsum(torch.tensor([0] + num_nodes_per_graph), dim=0); total_nodes = num_nodes_cumsum[-1].item()
    batch_size = len(batch_x_n); src_, dst_ = [], []
    for i in range(batch_size):
        if batch_src[i].numel() > 0:
            max_local_src = batch_src[i].max().item(); max_local_dst = batch_dst[i].max().item(); nodes_in_this_graph = num_nodes_per_graph[i]
            if max_local_src >= nodes_in_this_graph or max_local_dst >= nodes_in_this_graph: raise IndexError(f"Collate Error: Local index out of bounds in graph {valid_indices[i]}. Max src {max_local_src}, Max dst {max_local_dst}, Num nodes {nodes_in_this_graph}")
            src_.append(batch_src[i] + num_nodes_cumsum[i])
        if batch_dst[i].numel() > 0: dst_.append(batch_dst[i] + num_nodes_cumsum[i])
    src = torch.cat(src_, dim=0) if src_ else torch.LongTensor([]); dst = torch.cat(dst_, dim=0) if dst_ else torch.LongTensor([])
    if src.numel() > 0 and src.max().item() >= total_nodes: raise IndexError(f"Collate Error: Max source index {src.max().item()} >= total nodes {total_nodes}. Cumsum: {num_nodes_cumsum}")
    if dst.numel() > 0 and dst.max().item() >= total_nodes: raise IndexError(f"Collate Error: Max destination index {dst.max().item()} >= total nodes {total_nodes}. Cumsum: {num_nodes_cumsum}")
    edge_index = torch.stack([dst, src]) if src.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
    x_n = torch.cat(batch_x_n, dim=0).long(); abs_level = torch.cat(batch_abs_level, dim=0).float().unsqueeze(-1) if batch_abs_level else torch.empty((0,1), dtype=torch.float); rel_level = torch.cat(batch_rel_level, dim=0).float().unsqueeze(-1) if batch_rel_level else torch.empty((0,1), dtype=torch.float)
    nids, gids = [], []
    for i in range(batch_size):
        start_node = num_nodes_cumsum[i]; end_node = num_nodes_cumsum[i+1]
        if end_node > start_node: nids.append(torch.arange(start_node, end_node).long()); gids.append(torch.ones(end_node - start_node).fill_(i).long())
    if not nids: n2g_index = torch.empty((2, 0), dtype=torch.long)
    else: nids_cat = torch.cat(nids, dim=0); gids_cat = torch.cat(gids, dim=0); n2g_index = torch.stack([gids_cat, nids_cat])
    return batch_size, edge_index, batch_x_n, abs_level, rel_level, n2g_index, [] # Return list x_n

# ... (collate_node_count remains the same) ...
def collate_node_count(data):
# ... (代码保持不变) ...
    data = [item for item in data if item is not None];
    if not data: return None
    conditional = len(data[0]) == 7
    if conditional:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_y, batch_label = map(list, zip(*data))
        valid_indices = [i for i, x in enumerate(batch_x_n) if isinstance(x, torch.Tensor) and x.numel() > 0 and isinstance(batch_src[i], torch.Tensor) and isinstance(batch_dst[i], torch.Tensor) and isinstance(batch_abs_level[i], torch.Tensor) and isinstance(batch_rel_level[i], torch.Tensor)]
        if not valid_indices: return None
        batch_src = [batch_src[i] for i in valid_indices]; batch_dst = [batch_dst[i] for i in valid_indices]; batch_x_n_list = [batch_x_n[i] for i in valid_indices]
        batch_abs_level = [batch_abs_level[i] for i in valid_indices]; batch_rel_level = [batch_rel_level[i] for i in valid_indices]; batch_y = [batch_y[i] for i in valid_indices]; batch_label = [batch_label[i] for i in valid_indices]
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_label = map(list, zip(*data)); batch_y = None
        valid_indices = [i for i, x in enumerate(batch_x_n) if isinstance(x, torch.Tensor) and x.numel() > 0 and isinstance(batch_src[i], torch.Tensor) and isinstance(batch_dst[i], torch.Tensor) and isinstance(batch_abs_level[i], torch.Tensor) and isinstance(batch_rel_level[i], torch.Tensor)]
        if not valid_indices: return None
        batch_src = [batch_src[i] for i in valid_indices]; batch_dst = [batch_dst[i] for i in valid_indices]; batch_x_n_list = [batch_x_n[i] for i in valid_indices]
        batch_abs_level = [batch_abs_level[i] for i in valid_indices]; batch_rel_level = [batch_rel_level[i] for i in valid_indices]; batch_label = [batch_label[i] for i in valid_indices]
    batch_size, batch_edge_index, _, batch_abs_level_collated, batch_rel_level_collated, batch_n2g_index, _ = collate_common( batch_src, batch_dst, batch_x_n_list, batch_abs_level, batch_rel_level)
    if batch_size == 0: return None
    batch_label = torch.stack(batch_label);
    batch_x_n_collated = torch.cat(batch_x_n_list, dim=0).long()
    if batch_y is not None:
        return batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level_collated, batch_rel_level_collated, batch_y, batch_n2g_index, batch_label, batch_x_n_list
    else:
        return batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level_collated, batch_rel_level_collated, batch_n2g_index, batch_label

# ... (collate_node_pred remains the same) ...
def collate_node_pred(data):
# ... (代码保持不变) ...
    data = [item for item in data if item is not None];
    if not data: return None
    conditional = len(data[0]) == 9
    if conditional:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_z_t, batch_t, batch_y, batch_z = map(list, zip(*data))
        valid_indices = [i for i, x in enumerate(batch_x_n) if isinstance(x, torch.Tensor) and x.numel() > 0 and isinstance(batch_src[i], torch.Tensor) and isinstance(batch_dst[i], torch.Tensor) and isinstance(batch_abs_level[i], torch.Tensor) and isinstance(batch_rel_level[i], torch.Tensor)]
        if not valid_indices: return None
        batch_src=[batch_src[i] for i in valid_indices]; batch_dst=[batch_dst[i] for i in valid_indices]; batch_x_n_list=[batch_x_n[i] for i in valid_indices]
        batch_abs_level=[batch_abs_level[i] for i in valid_indices]; batch_rel_level=[batch_rel_level[i] for i in valid_indices]; batch_z_t=[batch_z_t[i] for i in valid_indices]
        batch_t=[batch_t[i] for i in valid_indices]; batch_y=[batch_y[i] for i in valid_indices]; batch_z=[batch_z[i] for i in valid_indices]
    else:
        batch_src, batch_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_z_t, batch_t, batch_z = map(list, zip(*data)); batch_y = None
        valid_indices = [i for i, x in enumerate(batch_x_n) if isinstance(x, torch.Tensor) and x.numel() > 0 and isinstance(batch_src[i], torch.Tensor) and isinstance(batch_dst[i], torch.Tensor) and isinstance(batch_abs_level[i], torch.Tensor) and isinstance(batch_rel_level[i], torch.Tensor)]
        if not valid_indices: return None
        batch_src=[batch_src[i] for i in valid_indices]; batch_dst=[batch_dst[i] for i in valid_indices]; batch_x_n_list=[batch_x_n[i] for i in valid_indices]
        batch_abs_level=[batch_abs_level[i] for i in valid_indices]; batch_rel_level=[batch_rel_level[i] for i in valid_indices]; batch_z_t=[batch_z_t[i] for i in valid_indices]
        batch_t=[batch_t[i] for i in valid_indices]; batch_z=[batch_z[i] for i in valid_indices]
    batch_size, batch_edge_index, _, batch_abs_level_collated, batch_rel_level_collated, batch_n2g_index, _ = collate_common(batch_src, batch_dst, batch_x_n_list, batch_abs_level, batch_rel_level)
    if batch_size == 0: return None
    batch_x_n_collated = torch.cat(batch_x_n_list, dim=0).long()
    num_query_per_graph = [len(z_t_i) for z_t_i in batch_z_t if isinstance(z_t_i, torch.Tensor)]
    if not num_query_per_graph: num_query_cumsum = torch.LongTensor([0] * (batch_size + 1)); query2g = torch.LongTensor([])
    else:
         num_query_cumsum = torch.cumsum(torch.tensor([0] + num_query_per_graph), dim=0); query2g = []
         for i in range(batch_size):
             if i < len(num_query_per_graph) and num_query_per_graph[i] > 0: query2g.append(torch.ones(num_query_per_graph[i]).fill_(i).long())
         query2g = torch.cat(query2g) if query2g else torch.LongTensor([])
    batch_z_t_filtered = [t for t in batch_z_t if isinstance(t, torch.Tensor) and t.numel() > 0]; batch_t_filtered = [t for t in batch_t if isinstance(t, torch.Tensor) and t.numel() > 0]; batch_z_filtered = [z for z in batch_z if isinstance(z, torch.Tensor) and z.numel() > 0]
    batch_z_t = torch.cat(batch_z_t_filtered) if batch_z_t_filtered else torch.LongTensor([]); batch_t = torch.cat(batch_t_filtered).unsqueeze(-1) if batch_t_filtered else torch.LongTensor([]).unsqueeze(-1); batch_z = torch.cat(batch_z_filtered) if batch_z_filtered else torch.LongTensor([])
    if batch_z.numel() > 0 and batch_z.ndim == 1: batch_z = batch_z.unsqueeze(-1)
    if conditional: return batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level_collated, batch_rel_level_collated, batch_n2g_index, batch_z_t, batch_t, batch_y, query2g, num_query_cumsum, batch_z, batch_x_n_list
    else: return batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level_collated, batch_rel_level_collated, batch_n2g_index, batch_z_t, batch_t, query2g, num_query_cumsum, batch_z, None

# --- collate_edge_pred (MODIFIED) ---
def collate_edge_pred(data):
    data = [item for item in data if item is not None];
    if not data: return None
    
    # --- MODIFICATION START: Handle new item (query_dst_x_n) ---
    # Conditional now returns 13 items, Unconditional returns 10 items
    conditional = len(data[0]) == 13
    
    if conditional:
        # Unpack the new batch_query_dst_x_n
        batch_src, batch_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_t, batch_y, batch_query_src, batch_query_dst, batch_label, batch_query_dst_x_n = map(list, zip(*data))
        # Add query_dst_x_n to validation check
        valid_indices = [i for i, x in enumerate(batch_x_n) if isinstance(x, torch.Tensor) and x.numel() > 0 and isinstance(batch_src[i], torch.Tensor) and isinstance(batch_dst[i], torch.Tensor) and isinstance(batch_noisy_src[i], torch.Tensor) and isinstance(batch_noisy_dst[i], torch.Tensor) and isinstance(batch_abs_level[i], torch.Tensor) and isinstance(batch_rel_level[i], torch.Tensor) and isinstance(batch_t[i], torch.Tensor) and isinstance(batch_query_src[i], torch.Tensor) and isinstance(batch_query_dst[i], torch.Tensor) and isinstance(batch_label[i], torch.Tensor) and isinstance(batch_query_dst_x_n[i], torch.Tensor)]
        if not valid_indices: return None
        # Filter all lists based on valid_indices
        batch_src=[batch_src[i] for i in valid_indices]; batch_dst=[batch_dst[i] for i in valid_indices]; batch_noisy_src=[batch_noisy_src[i] for i in valid_indices]; batch_noisy_dst=[batch_noisy_dst[i] for i in valid_indices]; batch_x_n_list=[batch_x_n[i] for i in valid_indices]
        batch_abs_level=[batch_abs_level[i] for i in valid_indices]; batch_rel_level=[batch_rel_level[i] for i in valid_indices]; batch_t=[batch_t[i] for i in valid_indices]; batch_y=[batch_y[i] for i in valid_indices]
        batch_query_src=[batch_query_src[i] for i in valid_indices]; batch_query_dst=[batch_query_dst[i] for i in valid_indices]; batch_label=[batch_label[i] for i in valid_indices]
        batch_query_dst_x_n = [batch_query_dst_x_n[i] for i in valid_indices] # Filter new item
    else:
        # Unpack the new batch_query_dst_x_n
        batch_src, batch_dst, batch_noisy_src, batch_noisy_dst, batch_x_n, batch_abs_level, batch_rel_level, batch_t, batch_query_src, batch_query_dst, batch_label, batch_query_dst_x_n = map(list, zip(*data)); batch_y = None
        # Add query_dst_x_n to validation check
        valid_indices = [i for i, x in enumerate(batch_x_n) if isinstance(x, torch.Tensor) and x.numel() > 0 and isinstance(batch_src[i], torch.Tensor) and isinstance(batch_dst[i], torch.Tensor) and isinstance(batch_noisy_src[i], torch.Tensor) and isinstance(batch_noisy_dst[i], torch.Tensor) and isinstance(batch_abs_level[i], torch.Tensor) and isinstance(batch_rel_level[i], torch.Tensor) and isinstance(batch_t[i], torch.Tensor) and isinstance(batch_query_src[i], torch.Tensor) and isinstance(batch_query_dst[i], torch.Tensor) and isinstance(batch_label[i], torch.Tensor) and isinstance(batch_query_dst_x_n[i], torch.Tensor)]
        if not valid_indices: return None
        # Filter all lists based on valid_indices
        batch_src=[batch_src[i] for i in valid_indices]; batch_dst=[batch_dst[i] for i in valid_indices]; batch_noisy_src=[batch_noisy_src[i] for i in valid_indices]; batch_noisy_dst=[batch_noisy_dst[i] for i in valid_indices]; batch_x_n_list=[batch_x_n[i] for i in valid_indices]
        batch_abs_level=[batch_abs_level[i] for i in valid_indices]; batch_rel_level=[batch_rel_level[i] for i in valid_indices]; batch_t=[batch_t[i] for i in valid_indices]
        batch_query_src=[batch_query_src[i] for i in valid_indices]; batch_query_dst=[batch_query_dst[i] for i in valid_indices]; batch_label=[batch_label[i] for i in valid_indices]
        batch_query_dst_x_n = [batch_query_dst_x_n[i] for i in valid_indices] # Filter new item
    # --- MODIFICATION END ---

    batch_size = len(batch_x_n_list);
    if batch_size == 0: return None
    num_nodes_per_graph = [len(x_n_i) for x_n_i in batch_x_n_list]; num_nodes_cumsum = torch.cumsum(torch.tensor([0] + num_nodes_per_graph), dim=0); total_nodes = num_nodes_cumsum[-1].item()
    src_, dst_, noisy_src_, noisy_dst_, query_src_, query_dst_, t_ = [], [], [], [], [], [], []
    for i in range(batch_size):
        offset = num_nodes_cumsum[i]
        if batch_src[i].numel() > 0: src_.append(batch_src[i] + offset)
        if batch_dst[i].numel() > 0: dst_.append(batch_dst[i] + offset)
        if batch_noisy_src[i].numel() > 0: noisy_src_.append(batch_noisy_src[i] + offset)
        if batch_noisy_dst[i].numel() > 0: noisy_dst_.append(batch_noisy_dst[i] + offset)
        if batch_query_src[i].numel() > 0: query_src_.append(batch_query_src[i] + offset)
        if batch_query_dst[i].numel() > 0: query_dst_.append(batch_query_dst[i] + offset)
        num_queries_i = len(batch_query_src[i])
        if num_queries_i > 0 and batch_t[i].numel() > 0: t_.append(batch_t[i].expand(num_queries_i, -1))
    src = torch.cat(src_, dim=0) if src_ else torch.LongTensor([]); dst = torch.cat(dst_, dim=0) if dst_ else torch.LongTensor([])
    edge_index = torch.stack([dst, src]) if src.numel() > 0 else torch.empty((2,0), dtype=torch.long)
    noisy_src = torch.cat(noisy_src_, dim=0) if noisy_src_ else torch.LongTensor([]); noisy_dst = torch.cat(noisy_dst_, dim=0) if noisy_dst_ else torch.LongTensor([])
    noisy_edge_index = torch.stack([noisy_dst, noisy_src]) if noisy_src.numel() > 0 else torch.empty((2,0), dtype=torch.long)
    query_src = torch.cat(query_src_) if query_src_ else torch.LongTensor([]); query_dst = torch.cat(query_dst_) if query_dst_ else torch.LongTensor([])
    t = torch.cat(t_) if t_ else torch.LongTensor([]).unsqueeze(-1)

    # ... (Validation check for noisy_edge_index) ...
    if noisy_edge_index.numel() > 0:
        max_noisy_idx_dst = noisy_edge_index[0].max().item()
        max_noisy_idx_src = noisy_edge_index[1].max().item()
        if max_noisy_idx_dst >= total_nodes or max_noisy_idx_src >= total_nodes:
            raise IndexError(f"Collate Error (EdgePred): Noisy edge index out of bounds. Max idx ({max_noisy_idx_dst},{max_noisy_idx_src}) >= total nodes ({total_nodes})")
    
    if edge_index.numel() > 0:
         max_edge_idx_dst = edge_index[0].max().item()
         max_edge_idx_src = edge_index[1].max().item()
         if max_edge_idx_dst >= total_nodes or max_edge_idx_src >= total_nodes:
             raise IndexError(f"Collate Error (EdgePred): Edge index out of bounds. Max edge idx ({max_edge_idx_dst},{max_edge_idx_src}) >= total nodes ({total_nodes})")

    if query_src.numel() > 0 and query_src.max().item() >= total_nodes: raise IndexError(f"Collate Error (EdgePred): Query src index out of bounds. Max idx ({query_src.max().item()}) >= total nodes ({total_nodes})")
    if query_dst.numel() > 0 and query_dst.max().item() >= total_nodes: raise IndexError(f"Collate Error (EdgePred): Query dst index out of bounds. Max idx ({query_dst.max().item()}) >= total nodes ({total_nodes})")

    batch_x_n_collated = torch.cat(batch_x_n_list, dim=0).long(); batch_abs_level_collated = torch.cat(batch_abs_level, dim=0).float().unsqueeze(-1); batch_rel_level_collated = torch.cat(batch_rel_level, dim=0).float().unsqueeze(-1)
    batch_label_collated = torch.cat(batch_label) if batch_label and any(l.numel() > 0 for l in batch_label) else torch.LongTensor([])
    
    # --- MODIFICATION START: Collate the new query_dst_x_n tensor ---
    batch_query_dst_x_n_collated = torch.cat(batch_query_dst_x_n) if batch_query_dst_x_n and any(q.numel() > 0 for q in batch_query_dst_x_n) else torch.LongTensor([])
    # --- MODIFICATION END ---
    
    if conditional: 
        # Return 11 items
        return edge_index, noisy_edge_index, batch_x_n_list, batch_abs_level_collated, batch_rel_level_collated, t, batch_y, query_src, query_dst, batch_label_collated, batch_query_dst_x_n_collated
    else: 
        # Return 10 items
        return edge_index, noisy_edge_index, batch_x_n_list, batch_abs_level_collated, batch_rel_level_collated, t, query_src, query_dst, batch_label_collated, batch_query_dst_x_n_collated
