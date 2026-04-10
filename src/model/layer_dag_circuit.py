# import dgl.sparse as dglsp
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from einops import rearrange

# __all__ = [
#     'LayerDAG'
# ]

# # ... (TruthTableEncoder, SinusoidalPE, BiMPNNLayer, OneHotPE, MultiEmbedding, BiMPNNEncoder, GraphClassifier, TransformerLayer, NodePredModel, EdgePredModel 保持不变) ...
# class TruthTableEncoder(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, output_size):
#         super().__init__()
#         self.output_size = output_size
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # [B, 16, H/2, W/2]
#             nn.GELU(),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, H/4, W/4]
#             nn.GELU(),
#             nn.AdaptiveAvgPool2d((4, 4)), # Pool to a fixed size [B, 32, 4, 4]
#         )
#         self.mlp = nn.Linear(32 * 4 * 4, output_size)
#     def forward(self, y_list, device):
#         if not y_list or all(y.numel() == 0 for y in y_list):
#             return torch.zeros(len(y_list), self.output_size, device=device)
#         max_rows = max(y.shape[0] for y in y_list if y.numel() > 0) if any(y.numel() > 0 for y in y_list) else 1
#         max_cols = max(y.shape[1] for y in y_list if y.numel() > 0) if any(y.numel() > 0 for y in y_list) else 1
#         padded_tensors = []
#         for y in y_list:
#             if y.numel() == 0:
#                 padded_tensors.append(torch.zeros(max_rows, max_cols, device=device, dtype=torch.float))
#                 continue
#             padded = F.pad(y.float(), (0, max_cols - y.shape[1], 0, max_rows - y.shape[0]))
#             padded_tensors.append(padded)
#         y_batch = torch.stack(padded_tensors).float()
#         y_batch = y_batch.unsqueeze(1)
#         y_batch = y_batch.to(device)
#         cnn_out = self.cnn(y_batch)
#         cnn_flat = cnn_out.view(cnn_out.size(0), -1)
#         y_emb = self.mlp(cnn_flat)
#         return y_emb
# class SinusoidalPE(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, pe_size):
#         super().__init__()
#         self.pe_size = pe_size
#         if pe_size > 0:
#             self.div_term = torch.exp(torch.arange(0, pe_size, 2) *
#                                       (-math.log(10000.0) / pe_size))
#             self.div_term = nn.Parameter(self.div_term, requires_grad=False)
#     def forward(self, position):
#         if self.pe_size == 0:
#             return torch.zeros(len(position), 0).to(position.device)
#         position = position.float()
#         return torch.cat([
#             torch.sin(position * self.div_term),
#             torch.cos(position * self.div_term)
#         ], dim=-1)
# class BiMPNNLayer(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, in_size, out_size):
#         super().__init__()
#         self.W = nn.Linear(in_size, out_size)
#         self.W_trans = nn.Linear(in_size, out_size)
#         self.W_self = nn.Linear(in_size, out_size)
#     def forward(self, A, A_T, h_n):
#         num_nodes_h = h_n.shape[0]; num_nodes_A = A.shape[0]
#         if num_nodes_A != num_nodes_h:
#              raise ValueError(f"Shape mismatch detected before SpMM: A has {num_nodes_A} nodes, h_n has {num_nodes_h} nodes.")
#         if A.nnz > 0:
#             A_row, A_col = A.coo()
#             max_A_row = A_row.max().item() if A_row.numel() > 0 else -1
#             max_A_col = A_col.max().item() if A_col.numel() > 0 else -1
#             if max_A_row >= num_nodes_A or max_A_col >= num_nodes_A:
#                  raise IndexError(f"Index out of bounds detected in A matrix before SpMM.")
#         if A_T.nnz > 0:
#             AT_row, AT_col = A_T.coo()
#             max_AT_row = AT_row.max().item() if AT_row.numel() > 0 else -1
#             max_AT_col = AT_col.max().item() if AT_col.numel() > 0 else -1
#             if max_AT_row >= num_nodes_A or max_AT_col >= num_nodes_A:
#                  raise IndexError(f"Index out of bounds detected in A_T matrix before SpMM.")
#         if torch.isnan(h_n).any() or torch.isinf(h_n).any():
#             raise ValueError("NaN/Inf in input features to BiMPNNLayer")
#         try:
#             h_n_w = self.W(h_n); h_n_w_trans = self.W_trans(h_n); h_n_w_self = self.W_self(h_n)
#             if A.nnz == 0: h_n_out = h_n_w_self
#             else: term1 = A @ h_n_w; term2 = A_T @ h_n_w_trans; h_n_out = term1 + term2 + h_n_w_self
#         except RuntimeError as e:
#             print(f"--- Runtime Error likely during SpMM ---"); print(f"A shape: {A.shape}, A nnz: {A.nnz}"); print(f"h_n shape: {h_n.shape}"); print(f"Error Message: {e}"); print("--- End Runtime Error Info ---")
#             raise e
#         if torch.isnan(h_n_out).any() or torch.isinf(h_n_out).any():
#              print("ERROR: NaN or Inf detected in h_n_out after SpMM/Add!")
#         return F.gelu(h_n_out)
# class OneHotPE(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, pe_size):
#         super().__init__()
#         self.pe_size = pe_size
#     def forward(self, position):
#         if self.pe_size == 0:
#             return torch.zeros(len(position), 0).to(position.device)
#         return F.one_hot(position.clamp(max=self.pe_size - 1).long().squeeze(-1),
#                          num_classes=self.pe_size).float() # Convert to float
# class MultiEmbedding(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, num_x_n_cat, hidden_size):
#         super().__init__()
#         if isinstance(num_x_n_cat, torch.Tensor): num_cats = num_x_n_cat.tolist() if num_x_n_cat.numel() > 1 else [num_x_n_cat.item()]
#         elif isinstance(num_x_n_cat, int): num_cats = [num_x_n_cat]
#         elif isinstance(num_x_n_cat, list): num_cats = num_x_n_cat
#         else: raise TypeError(f"Unsupported type for num_x_n_cat: {type(num_x_n_cat)}")
#         self.emb_list = nn.ModuleList([ nn.Embedding(int(num_i), hidden_size) for num_i in num_cats ])
#     def forward(self, x_n_cat):
#         if x_n_cat.numel() == 0:
#             output_dim = len(self.emb_list) * self.emb_list[0].embedding_dim if len(self.emb_list) > 0 else 0
#             if len(self.emb_list) == 1: output_dim = self.emb_list[0].embedding_dim
#             return torch.empty((0, output_dim), device=x_n_cat.device, dtype=torch.float)
#         if len(self.emb_list) == 1:
#              if x_n_cat.ndim == 2 and x_n_cat.shape[1] == 1: x_n_cat = x_n_cat.squeeze(1)
#              if x_n_cat.ndim != 1: raise ValueError(f"MultiEmbedding expected 1D input when single feature, got shape {x_n_cat.shape}")
#              x_n_emb = self.emb_list[0](x_n_cat)
#         else:
#             if x_n_cat.ndim != 2 or x_n_cat.shape[1] != len(self.emb_list): raise ValueError(f"MultiEmbedding expected 2D input with shape (*, {len(self.emb_list)}), got shape {x_n_cat.shape}")
#             embeddings = []
#             for i in range(len(self.emb_list)):
#                  max_idx = self.emb_list[i].num_embeddings; indices = x_n_cat[:, i]
#                  if (indices >= max_idx).any() or (indices < 0).any(): raise IndexError(f"Index out of bounds for embedding layer {i}. Max index allowed: {max_idx-1}, Got indices between {indices.min()} and {indices.max()}")
#                  embeddings.append(self.emb_list[i](indices))
#             x_n_emb = torch.cat(embeddings, dim=1)
#         return x_n_emb
# class BiMPNNEncoder(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, num_x_n_cat, x_n_emb_size, pe_emb_size, hidden_size, num_mpnn_layers, pe=None, y_emb_size=0, pool=None):
#         super().__init__()
#         self.pe = pe; self.pe_emb_size = pe_emb_size
#         if self.pe in ['relative_level', 'abs_level']: self.level_emb = SinusoidalPE(pe_emb_size)
#         elif self.pe == 'relative_level_one_hot': self.level_emb = OneHotPE(pe_emb_size)
#         else: self.level_emb = None
#         if isinstance(num_x_n_cat, torch.Tensor): num_feat_dims = len(num_x_n_cat.tolist()) if num_x_n_cat.numel() > 1 else 1
#         elif isinstance(num_x_n_cat, int): num_feat_dims = 1
#         elif isinstance(num_x_n_cat, list): num_feat_dims = len(num_x_n_cat)
#         else: num_feat_dims = 0
#         total_x_n_emb_size = num_feat_dims * x_n_emb_size
#         self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)
#         actual_input_size = total_x_n_emb_size
#         if self.level_emb is not None: actual_input_size += pe_emb_size
#         actual_input_size += y_emb_size
#         self.proj_input = nn.Sequential( nn.Linear(actual_input_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size) )
#         self.mpnn_layers = nn.ModuleList()
#         for _ in range(num_mpnn_layers): self.mpnn_layers.append(BiMPNNLayer(hidden_size, hidden_size))
#         self.project_output_n = nn.Sequential( nn.Linear((num_mpnn_layers + 1) * hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size) )
#         self.pool = pool
#         if pool is not None: self.bn_g = nn.BatchNorm1d(hidden_size)
#     def forward(self, A, x_n, abs_level, rel_level, h_y=None, A_n2g=None):
#         if x_n.numel() == 0:
#             output_size = self.project_output_n[-1].out_features
#             if self.pool is None: return torch.empty((0, output_size), device=A.device, dtype=torch.float)
#             else: batch_size = A_n2g.shape[0] if A_n2g is not None else 0; return torch.empty((batch_size, output_size), device=A.device, dtype=torch.float)
#         A_T = A.T; h_n_initial = self.x_n_emb(x_n); node_pe = None
#         if self.level_emb is not None:
#             if self.pe == 'abs_level' and abs_level.numel() > 0: node_pe = self.level_emb(abs_level)
#             elif self.pe in ['relative_level', 'relative_level_one_hot'] and rel_level.numel() > 0: node_pe = self.level_emb(rel_level)
#             if node_pe is not None and node_pe.numel() == 0: node_pe = None
#         features_to_concat = [h_n_initial]
#         if node_pe is not None:
#              if node_pe.shape[0] != h_n_initial.shape[0]:
#                    if node_pe.shape[0] == 1: node_pe = node_pe.expand(h_n_initial.shape[0], -1)
#                    else: raise ValueError(f"Shape mismatch: h_n has {h_n_initial.shape[0]} rows, PE has {node_pe.shape[0]} rows.")
#              features_to_concat.append(node_pe)
#         if h_y is not None:
#              if h_y.shape[0] != h_n_initial.shape[0]: raise ValueError(f"Shape mismatch: h_n has {h_n_initial.shape[0]} rows, h_y (conditional embedding) has {h_y.shape[0]} rows. Check get_batch_y.")
#              features_to_concat.append(h_y)
#         h_n_combined = torch.cat(features_to_concat, dim=-1)
#         h_n = self.proj_input(h_n_combined); h_n_cat = [h_n]
#         for layer in self.mpnn_layers: h_n = layer(A, A_T, h_n); h_n_cat.append(h_n)
#         h_n_final = torch.cat(h_n_cat, dim=-1); h_n_output = self.project_output_n(h_n_final)
#         if self.pool is None: return h_n_output
#         elif A_n2g is None: raise ValueError("A_n2g matrix is required for pooling but was not provided.")
#         elif self.pool == 'sum': h_g = A_n2g @ h_n_output; return self.bn_g(h_g) if h_g.shape[0] > 0 and h_g.shape[1] > 0 else h_g
#         elif self.pool == 'mean':
#             h_g = A_n2g @ h_n_output; num_nodes_per_graph = A_n2g.sum(dim=1).unsqueeze(-1); num_nodes_per_graph = torch.clamp(num_nodes_per_graph, min=1.0)
#             h_g = h_g / num_nodes_per_graph; return self.bn_g(h_g) if h_g.shape[0] > 0 and h_g.shape[1] > 0 else h_g
#         else: raise ValueError(f"Unknown pooling type: {self.pool}")
# class GraphClassifier(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, graph_encoder, emb_size, num_classes):
#         super().__init__()
#         self.graph_encoder = graph_encoder
#         self.predictor = nn.Sequential( nn.Linear(emb_size, emb_size), nn.GELU(), nn.Linear(emb_size, num_classes) )
#     def forward(self, A, x_n, abs_level, rel_level, A_n2g, h_y=None):
#         h_g = self.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y, A_n2g=A_n2g)
#         if h_g.numel() == 0: return torch.empty((0, self.predictor[-1].out_features), device=h_g.device, dtype=h_g.dtype)
#         pred_g = self.predictor(h_g); return pred_g
# class TransformerLayer(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, hidden_size, num_heads, dropout):
#         super().__init__()
#         self.to_v = nn.Linear(hidden_size, hidden_size); self.to_qk = nn.Linear(hidden_size, hidden_size * 2)
#         self._reset_parameters()
#         self.num_heads = num_heads; head_dim = hidden_size // num_heads; assert head_dim * num_heads == hidden_size
#         self.scale = head_dim ** -0.5
#         self.proj_new = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout)); self.norm1 = nn.LayerNorm(hidden_size)
#         self.out_proj = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Linear(4 * hidden_size, hidden_size), nn.Dropout(dropout)); self.norm2 = nn.LayerNorm(hidden_size)
#     def _reset_parameters(self): nn.init.xavier_uniform_(self.to_v.weight); nn.init.xavier_uniform_(self.to_qk.weight)
#     def attn(self, q, k, v, num_query_cumsum):
#         batch_size = len(num_query_cumsum) - 1; num_query_nodes = torch.diff(num_query_cumsum); max_num_nodes = num_query_nodes.max().item() if batch_size > 0 else 0
#         q_padded = q.new_zeros(batch_size, max_num_nodes, q.shape[-1]); k_padded = k.new_zeros(batch_size, max_num_nodes, k.shape[-1]); v_padded = v.new_zeros(batch_size, max_num_nodes, v.shape[-1])
#         pad_mask = q.new_zeros(batch_size, max_num_nodes).bool()
#         for i in range(batch_size):
#             q_padded[i, :num_query_nodes[i]] = q[num_query_cumsum[i]:num_query_cumsum[i + 1]]; k_padded[i, :num_query_nodes[i]] = k[num_query_cumsum[i]:num_query_cumsum[i + 1]]; v_padded[i, :num_query_nodes[i]] = v[num_query_cumsum[i]:num_query_cumsum[i + 1]]
#             pad_mask[i, num_query_nodes[i]:] = True
#         q_padded = rearrange(q_padded, 'b n (h d) -> b h n d', h=self.num_heads); k_padded = rearrange(k_padded, 'b n (h d) -> b h n d', h=self.num_heads); v_padded = rearrange(v_padded, 'b n (h d) -> b h n d', h=self.num_heads)
#         dot = torch.matmul(q_padded, k_padded.transpose(-1, -2)) * self.scale; dot = dot.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
#         attn_scores = F.softmax(dot, dim=-1); h_n_padded = torch.matmul(attn_scores, v_padded); h_n_padded = rearrange(h_n_padded, 'b h n d -> (b n) (h d)')
#         pad_mask = (~pad_mask).reshape(-1); return h_n_padded[pad_mask]
#     def forward(self, h_n, num_query_cumsum):
#         v_n = self.to_v(h_n); q_n, k_n = self.to_qk(h_n).chunk(2, dim=-1); h_n_new = self.attn(q_n, k_n, v_n, num_query_cumsum); h_n_new = self.proj_new(h_n_new)
#         h_n = self.norm1(h_n + h_n_new); h_n = self.norm2(h_n + self.out_proj(h_n)); return h_n
# class NodePredModel(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, graph_encoder, num_x_n_cat, x_n_emb_size, t_emb_size, in_hidden_size, out_hidden_size, num_transformer_layers, num_heads, dropout):
#         super().__init__()
#         self.graph_encoder = graph_encoder
#         if isinstance(num_x_n_cat, torch.Tensor): num_real_classes = num_x_n_cat - 1
#         elif isinstance(num_x_n_cat, int): num_real_classes = torch.LongTensor([num_x_n_cat - 1])
#         elif isinstance(num_x_n_cat, list): num_real_classes = torch.LongTensor(num_x_n_cat) - 1
#         else: raise TypeError(f"Unsupported type for num_x_n_cat: {type(num_x_n_cat)}")
#         num_real_classes = torch.clamp(num_real_classes, min=1)
#         self.x_n_emb = MultiEmbedding(num_real_classes, x_n_emb_size)
#         self.t_emb = SinusoidalPE(t_emb_size)
#         num_feat_dims = len(num_real_classes.tolist()) if num_real_classes.numel() > 1 else 1
#         actual_input_size = in_hidden_size + t_emb_size + num_feat_dims * x_n_emb_size
#         self.project_h_n = nn.Sequential(nn.Linear(actual_input_size, out_hidden_size), nn.GELU())
#         self.trans_layers = nn.ModuleList([TransformerLayer(out_hidden_size, num_heads, dropout) for _ in range(num_transformer_layers)])
#         self.pred_list = nn.ModuleList([])
#         num_real_classes_list = num_real_classes.tolist()
#         for num_classes_f in num_real_classes_list:
#             num_classes_f_valid = max(1, int(num_classes_f))
#             self.pred_list.append(nn.Sequential(nn.Linear(out_hidden_size, out_hidden_size), nn.GELU(), nn.Linear(out_hidden_size, num_classes_f_valid)))
#     def forward_with_h_g(self, h_g, x_n_t, t, query2g, num_query_cumsum):
#         if h_g.numel() == 0 or x_n_t.numel() == 0: return [torch.empty(0)] * len(self.pred_list) # Handle empty input
#         h_t = self.t_emb(t);
#         if h_g.shape[0] == 1 and h_t.shape[0] > 1: h_t = h_t[0].unsqueeze(0)
#         elif h_g.shape[0] > 1 and h_t.shape[0] == 1: h_t = h_t.expand(h_g.shape[0], -1)
#         elif h_g.shape[0] != h_t.shape[0]: raise ValueError(f"Batch size mismatch h_g ({h_g.shape[0]}) vs h_t ({h_t.shape[0]})")
#         h_g = torch.cat([h_g, h_t], dim=1);
#         h_n_t_emb = self.x_n_emb(x_n_t)
#         if query2g.numel() > 0 and query2g.max() >= h_g.shape[0]: raise IndexError(f"query2g index out of bounds: max index {query2g.max()}, h_g size {h_g.shape[0]}")
#         h_n_t = torch.cat([h_n_t_emb, h_g[query2g]], dim=1); h_n_t = self.project_h_n(h_n_t);
#         for trans_layer in self.trans_layers: h_n_t = trans_layer(h_n_t, num_query_cumsum)
#         pred = [pred_layer(h_n_t) for pred_layer in self.pred_list]; return pred
#     def forward(self, A, x_n, abs_level, rel_level, A_n2g, x_n_t, t, query2g, num_query_cumsum, h_y=None):
#         h_g = self.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y, A_n2g=A_n2g)
#         return self.forward_with_h_g(h_g, x_n_t, t, query2g, num_query_cumsum)
# class EdgePredModel(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self, graph_encoder, t_emb_size, in_hidden_size, out_hidden_size): # in_hidden_size is GNN output size
#         super().__init__()
#         self.graph_encoder = graph_encoder
#         self.t_emb = SinusoidalPE(t_emb_size)
#         self.pred = nn.Sequential(
#             nn.Linear(2 * in_hidden_size + t_emb_size, out_hidden_size), # Correct input size
#             nn.GELU(),
#             nn.Linear(out_hidden_size, 2)
#         )
#     def forward(self, A, x_n, abs_level, rel_level, t, query_src, query_dst, h_y=None):
#         h_n = self.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y) # Removed A_n2g as it's not needed if pool=None
#         if h_n.numel() == 0 or query_src.numel() == 0: return torch.empty((0, 2), device=h_n.device, dtype=h_n.dtype)
#         max_idx = max(query_src.max().item() if query_src.numel()>0 else -1, query_dst.max().item() if query_dst.numel()>0 else -1)
#         if max_idx >= h_n.shape[0]: raise IndexError(f"Query index {max_idx} out of bounds for h_n with size {h_n.shape[0]}")
#         h_t = self.t_emb(t); num_queries = query_src.shape[0]
#         if h_t.shape[0] == 1 and num_queries > 1: h_t = h_t.expand(num_queries, -1)
#         elif h_t.shape[0] != num_queries: raise ValueError(f"Shape mismatch between t_emb ({h_t.shape[0]}) and queries ({num_queries})")
#         h_e = torch.cat([h_t, h_n[query_src], h_n[query_dst]], dim=-1); return self.pred(h_e)
# class LayerDAG(nn.Module):
# # ... (代码保持不变) ...
#     def __init__(self,
#                  device,
#                  num_x_n_cat,
#                  node_count_encoder_config,
#                  max_layer_size,
#                  node_diffusion,
#                  node_pred_graph_encoder_config,
#                  node_predictor_config,
#                  edge_diffusion,
#                  edge_pred_graph_encoder_config,
#                  edge_predictor_config,
#                  max_level=None):
#         super().__init__()
#         if isinstance(num_x_n_cat, int): num_x_n_cat = torch.LongTensor([num_x_n_cat])
#         elif isinstance(num_x_n_cat, list): num_x_n_cat = torch.LongTensor(num_x_n_cat)
#         elif not isinstance(num_x_n_cat, torch.Tensor): raise TypeError(f"Unsupported type for num_x_n_cat: {type(num_x_n_cat)}")
#         def get_num_feat_dims(n_cat):
#             if isinstance(n_cat, torch.Tensor): return len(n_cat.tolist()) if n_cat.numel() > 1 else 1
#             elif isinstance(n_cat, int): return 1
#             elif isinstance(n_cat, list): return len(n_cat)
#             return 0
#         num_feat_dims = get_num_feat_dims(num_x_n_cat)
#         nc_y_emb_size = node_count_encoder_config.get('y_emb_size', 0); np_y_emb_size = node_pred_graph_encoder_config.get('y_emb_size', 0); ep_y_emb_size = edge_pred_graph_encoder_config.get('y_emb_size', 0)
#         max_y_emb_size = max(nc_y_emb_size, np_y_emb_size, ep_y_emb_size)
#         self.y_encoder = TruthTableEncoder(max_y_emb_size).to(device) if max_y_emb_size > 0 else None
#         nc_calculated_input_size = (num_feat_dims * node_count_encoder_config['x_n_emb_size'] + node_count_encoder_config.get('pe_emb_size', 0) + max_y_emb_size) # Use max_y_emb_size
#         node_count_encoder = BiMPNNEncoder(num_x_n_cat, x_n_emb_size=node_count_encoder_config['x_n_emb_size'], pe_emb_size=node_count_encoder_config.get('pe_emb_size', 0), hidden_size=nc_calculated_input_size, num_mpnn_layers=node_count_encoder_config['num_mpnn_layers'], pe=node_count_encoder_config.get('pe', None), y_emb_size=max_y_emb_size, pool=node_count_encoder_config.get('pool', None)).to(device)
#         self.node_count_model = GraphClassifier(node_count_encoder, emb_size=nc_calculated_input_size, num_classes=max_layer_size+1).to(device)
#         self.node_diffusion = node_diffusion
#         np_calculated_input_size = (num_feat_dims * node_pred_graph_encoder_config['x_n_emb_size'] + node_pred_graph_encoder_config.get('pe_emb_size', 0) + max_y_emb_size) # Use max_y_emb_size
#         node_pred_graph_encoder = BiMPNNEncoder(num_x_n_cat, x_n_emb_size=node_pred_graph_encoder_config['x_n_emb_size'], pe_emb_size=node_pred_graph_encoder_config.get('pe_emb_size', 0), hidden_size=np_calculated_input_size, num_mpnn_layers=node_pred_graph_encoder_config['num_mpnn_layers'], pe=node_pred_graph_encoder_config.get('pe', None), y_emb_size=max_y_emb_size, pool=node_pred_graph_encoder_config.get('pool', None)).to(device)
#         self.node_pred_model = NodePredModel(node_pred_graph_encoder, num_x_n_cat, node_pred_graph_encoder_config['x_n_emb_size'], in_hidden_size=np_calculated_input_size, **node_predictor_config).to(device)
#         self.edge_diffusion = edge_diffusion
#         ep_calculated_input_size = (num_feat_dims * edge_pred_graph_encoder_config['x_n_emb_size'] + edge_pred_graph_encoder_config.get('pe_emb_size', 0) + max_y_emb_size) # Use max_y_emb_size
#         if edge_pred_graph_encoder_config.get('pool', None) is not None: print("Warning: Edge prediction graph encoder config has pooling enabled...")
#         edge_pred_graph_encoder = BiMPNNEncoder(num_x_n_cat, x_n_emb_size=edge_pred_graph_encoder_config['x_n_emb_size'], pe_emb_size=edge_pred_graph_encoder_config.get('pe_emb_size', 0), hidden_size=ep_calculated_input_size, num_mpnn_layers=edge_pred_graph_encoder_config['num_mpnn_layers'], pe=edge_pred_graph_encoder_config.get('pe', None), y_emb_size=max_y_emb_size, pool=edge_pred_graph_encoder_config.get('pool', None)).to(device)
#         self.edge_pred_model = EdgePredModel(edge_pred_graph_encoder, in_hidden_size=ep_calculated_input_size, **edge_predictor_config).to(device)
#         self.max_level = max_level
#         self.dummy_x_n = num_x_n_cat - 1
#         if len(num_x_n_cat) > 1: self.dummy_x_n = num_x_n_cat - 1
#         elif len(num_x_n_cat) == 1: dummy_val = (num_x_n_cat - 1)[0].item() if num_x_n_cat.numel() > 0 else 0; self.dummy_x_n = int(dummy_val)
#     def get_batch_y(self, y_list, x_n_list, device):
# # ... (代码保持不变) ...
#         if self.y_encoder is None or y_list is None: return None
#         processed_x_n_list = []
#         if x_n_list and not isinstance(x_n_list[0], torch.Tensor):
#              try: processed_x_n_list = [torch.tensor(x, dtype=torch.long, device=device) for x in x_n_list]
#              except Exception as e: print(f"Warning: Could not convert x_n_list in get_batch_y: {e}"); processed_x_n_list = x_n_list
#         else: processed_x_n_list = x_n_list
#         h_y_graphs = self.y_encoder(y_list, device)
#         h_y_per_node_list = []
#         list_len = min(len(y_list), len(processed_x_n_list))
#         if h_y_graphs.shape[0] < list_len: list_len = h_y_graphs.shape[0]
#         for i in range(list_len):
#             num_nodes = len(processed_x_n_list[i])
#             if num_nodes > 0: h_y_per_node_list.append(h_y_graphs[i].expand(num_nodes, -1))
#         if not h_y_per_node_list:
#              encoder_output_size = self.y_encoder.output_size if self.y_encoder else 0
#              return torch.empty((0, encoder_output_size), device=device, dtype=torch.float)
#         else: batch_h_y = torch.cat(h_y_per_node_list, dim=0).to(device); return batch_h_y

#     def posterior(self, Z_t, Q_t, Q_bar_s, Q_bar_t, Z_0):
#         # ... (numerator/denominator calculation) ...
#         left_term = Z_t @ torch.transpose(Q_t, -1, -2); left_term = left_term.unsqueeze(dim=-2); right_term = Q_bar_s.unsqueeze(dim=-3); numerator = left_term * right_term
#         prod = Q_bar_t @ torch.transpose(Z_t, -1, -2); prod = torch.transpose(prod, -1, -2); denominator = prod.unsqueeze(-1); denominator = torch.clamp(denominator, min=1e-8)
#         out = numerator / denominator
#         prob = Z_0.unsqueeze(-1) * out
#         prob = prob.sum(dim=-2)
        
#         # --- FIX: Add normalization and clamping ---
#         if torch.isnan(prob).any():
#             # print("NaN detected in posterior (node) before normalization!")
#             prob = torch.nan_to_num(prob, nan=0.0) # Replace NaNs with 0
#         prob_sum = prob.sum(dim=-1, keepdim=True)
#         prob = prob / (prob_sum + 1e-8) # Normalize
#         prob = torch.clamp(prob, min=0.0, max=1.0) # Clamp
#         # --- End FIX ---

#         return prob
        
#     def posterior_edge(self,Z_t,alpha_t,alpha_bar_s,alpha_bar_t,Z_0,marginal_list,num_new_nodes_list,num_query_list):
# # ... (代码保持不变) ...
#         batch_size = len(num_new_nodes_list)
#         valid_query_indices = [i for i, nq in enumerate(num_query_list) if nq > 0]
#         if not valid_query_indices: return torch.BoolTensor([])
#         split_sizes = [num_query_list[i] for i in valid_query_indices]
#         Z_t_list = torch.split(Z_t, split_sizes, dim=0); Z_0_list = torch.split(Z_0, split_sizes, dim=0)
#         device = Z_t.device; e_mask_list = []
#         for i_split, i_batch in enumerate(valid_query_indices):
#             Z_t_i = Z_t_list[i_split]; Z_0_i = Z_0_list[i_split]; num_new_nodes_i = num_new_nodes_list[i_batch]
#             current_marginal = marginal_list[i_batch] if i_batch < len(marginal_list) else 0.5
#             Q_t_i, Q_bar_s_i, Q_bar_t_i = self.edge_diffusion.get_Qs(alpha_t, alpha_bar_s, alpha_bar_t, current_marginal)
#             Q_t_i=Q_t_i.to(device); Q_bar_s_i=Q_bar_s_i.to(device); Q_bar_t_i=Q_bar_t_i.to(device)
#             left_term_i = Z_t_i @ torch.transpose(Q_t_i, -1, -2); left_term_i = left_term_i.unsqueeze(dim=-2)
#             right_term_i = Q_bar_s_i.unsqueeze(dim=-3); numerator_i = left_term_i * right_term_i
#             prod_i = Q_bar_t_i @ torch.transpose(Z_t_i, -1, -2); prod_i = torch.transpose(prod_i, -1, -2)
#             denominator_i = prod_i.unsqueeze(-1); denominator_i = torch.clamp(denominator_i, min=1e-8)
#             out_i = numerator_i / denominator_i
#             prob_i = Z_0_i.unsqueeze(-1) * out_i; prob_i = prob_i.sum(dim=-2)
            
#             # --- FIX: Add NaN check before normalization ---
#             if torch.isnan(prob_i).any():
#                 print(f"NaN detected in posterior_edge (graph {i_batch}) before normalization!")
#                 prob_i = torch.nan_to_num(prob_i, nan=0.0) # Replace NaNs with 0
#             # --- End FIX ---

#             prob_i = prob_i / (prob_i.sum(dim=-1, keepdim=True) + 1e-6)
#             prob_i = prob_i[:, 1]
#             num_queries_i_actual = split_sizes[i_split]
#             if num_new_nodes_i <= 0 or num_queries_i_actual % num_new_nodes_i != 0:
#                  print(f"Warning: Cannot reshape posterior_edge probability for graph {i_batch}. Num new nodes: {num_new_nodes_i}, Num queries: {num_queries_i_actual}. Appending all-False mask.")
#                  e_mask_list.append(torch.zeros(num_queries_i_actual, dtype=torch.bool, device=device))
#                  continue
#             num_src_candidates = num_queries_i_actual // num_new_nodes_i
#             prob_i = prob_i.reshape(num_new_nodes_i, num_src_candidates)
            
#             # --- FIX: Add clamping ---
#             prob_i = torch.clamp(prob_i, min=0.0, max=1.0)
#             # --- End FIX ---
            
#             e_mask_i = torch.bernoulli(prob_i)
#             isolated_mask_i = (e_mask_i.sum(dim=1) == 0)
#             if isolated_mask_i.any(): highest_prob_idx = prob_i[isolated_mask_i].argmax(dim=1); e_mask_i[isolated_mask_i, highest_prob_idx] = 1
#             e_mask_list.append(e_mask_i.reshape(-1))
#         return torch.cat(e_mask_list).bool() if e_mask_list else torch.BoolTensor([])

#     @torch.no_grad()
#     def sample_node_layer(self,
# # ... (代码保持不变) ...
#                           A,
#                           x_n,
#                           abs_level,
#                           rel_level,
#                           A_n2g,
#                           curr_level=None,
#                           h_y=None, # Expect encoded y here
#                           min_num_steps_n=None,
#                           max_num_steps_n=None):
#         device = A.device
#         node_count_logits = self.node_count_model(A, x_n, abs_level, rel_level, A_n2g=A_n2g, h_y=h_y)
#         if curr_level == 0: node_count_logits[:, 0] = float('-inf')
#         node_count_probs = node_count_logits.softmax(dim=-1); num_new_nodes = node_count_probs.multinomial(1)
#         num_new_nodes_total = num_new_nodes.sum().item(); batch_size = num_new_nodes.shape[0]
#         if num_new_nodes_total == 0: return [torch.LongTensor([]).to(device) for _ in range(batch_size)]
#         num_classes_list = self.node_diffusion.num_classes_list; marginal_list = self.node_diffusion.m_list; D = len(num_classes_list)
#         x_n_t = []
#         for d in range(D):
#              if d >= len(marginal_list) or marginal_list[d].numel() == 0:
#                   print(f"Warning: Marginal missing or empty for dim {d}. Sampling uniformly.")
#                   num_classes_d = num_classes_list[d] if d < len(num_classes_list) else 1
#                   prior_d = torch.ones(num_new_nodes_total, num_classes_d, device=device) / num_classes_d # Ensure on device
#              else:
#                   marginal_d = marginal_list[d]
#                   prior_d = marginal_d[0].to(device).unsqueeze(0).expand(num_new_nodes_total, -1)
#              x_n_t_d = prior_d.multinomial(1).squeeze(-1); x_n_t.append(x_n_t_d)
#         x_n_t = torch.stack(x_n_t, dim=1) # Already on device
#         h_g = self.node_pred_model.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y, A_n2g=A_n2g)
#         num_new_nodes_list_cpu = num_new_nodes.squeeze(-1).cpu().tolist(); num_query_cumsum = torch.cumsum(torch.tensor([0] + num_new_nodes_list_cpu), dim=0).long().to(device)
#         query2g = []
#         for i in range(batch_size):
#             num_nodes_i = num_query_cumsum[i+1] - num_query_cumsum[i]
#             if num_nodes_i > 0: query2g.append(torch.full((num_nodes_i,), i, dtype=torch.long, device=device))
#         query2g = torch.cat(query2g) if query2g else torch.LongTensor([]).to(device)
#         T_x_n = self.node_diffusion.T
#         if max_num_steps_n is not None: T_x_n = min(T_x_n, max_num_steps_n)
#         time_x_n_list = list(reversed(range(0, T_x_n)))
#         if min_num_steps_n is not None and self.max_level is not None and self.max_level > 0:
#             num_steps_n = min_num_steps_n + int((T_x_n - min_num_steps_n) * (curr_level / self.max_level))
#             time_x_n_list = time_x_n_list[-num_steps_n:] if num_steps_n > 0 else []
#         for s_x_n in time_x_n_list:
#             t_x_n = s_x_n + 1
#             alpha_t = self.node_diffusion.alphas[t_x_n]; alpha_bar_s = self.node_diffusion.alpha_bars[s_x_n]; alpha_bar_t = self.node_diffusion.alpha_bars[t_x_n]
#             t_x_n_tensor = torch.full((batch_size, 1), t_x_n, dtype=torch.long, device=device)
#             x_n_0_logits = self.node_pred_model.forward_with_h_g( h_g, x_n_t, t_x_n_tensor, query2g, num_query_cumsum)
#             if not x_n_0_logits or not x_n_0_logits[0].numel(): continue
#             x_n_s = []
#             for d in range(D):
#                 if d >= len(x_n_0_logits): continue

#                 # --- [START] 关键修复：在 Softmax 之前处理 NAN/INF ---
#                 # 检查模型输出的 logits 是否稳定
#                 logits_d = x_n_0_logits[d]
#                 if torch.isnan(logits_d).any() or torch.isinf(logits_d).any():
#                     # print(f"Warning: NaN/Inf detected in node_pred_model logits at step {s_x_n}, dim {d}. Clamping.")
#                     # 1. 用 0 替换 NAN
#                     logits_d = torch.nan_to_num(logits_d, nan=0.0)
#                     # 2. 用一个大/小数值替换 Inf/-Inf
#                     logits_d = torch.clamp(logits_d, min=-1e4, max=1e4)
#                 # --- [END] 关键修复 ---

#                 Q_t_d = self.node_diffusion.get_Q(alpha_t, d).to(device); Q_bar_s_d = self.node_diffusion.get_Q(alpha_bar_s, d).to(device); Q_bar_t_d = self.node_diffusion.get_Q(alpha_bar_t, d).to(device)
                
#                 # [MODIFIED] 使用修复后的 logits
#                 x_n_0_probs_d = logits_d.softmax(dim=-1)
                
#                 if x_n_t.numel() > 0 and d < x_n_t.shape[1]:
#                      num_classes_d = num_classes_list[d]
#                      indices_d = x_n_t[:, d].clamp(0, num_classes_d - 1)
#                      x_n_t_one_hot_d = F.one_hot(indices_d, num_classes=num_classes_d).float()
                     
#                      # posterior 函数内部已经有 nan_to_num，但这里的 x_n_0_probs_d 现在应该是安全的
#                      x_n_s_probs_d = self.posterior(x_n_t_one_hot_d, Q_t_d, Q_bar_s_d, Q_bar_t_d, x_n_0_probs_d)
                     
#                      x_n_s_d = x_n_s_probs_d.multinomial(1).squeeze(-1); x_n_s.append(x_n_s_d)
#                 else:
#                     print(f"Warning: Skipping sampling for dim {d} due to empty x_n_t or dim mismatch.")
#                     num_classes_d = num_classes_list[d] if d < len(num_classes_list) else 1
#                     x_n_s.append(torch.zeros(num_new_nodes_total, dtype=torch.long, device=device))
#             if len(x_n_s) == D: x_n_t = torch.stack(x_n_s, dim=1)
#         return list(torch.split(x_n_t, num_new_nodes_list_cpu))

#     @torch.no_grad()
#     def sample_edge_layer(self,
# # ... (代码保持不变) ...
#                           num_nodes_cumsum, edge_index_list,
#                           batch_x_n, batch_abs_level, batch_rel_level,
#                           num_new_nodes_list, batch_query_src, batch_query_dst,
#                           query_src_list, query_dst_list,
#                           h_y=None, # Expect encoded y here
#                           curr_level=None,
#                           min_num_steps_e=None,
#                           max_num_steps_e=None,
#                           x_n_l_list_active=None): # Accept new node types
#         device = batch_x_n.device
#         e_t_mask_list = []; batch_size = len(num_new_nodes_list); marginal_list = []; num_query_list = []
        
#         # --- FIX: Ensure e_t_mask_list has correct number of elements ---
#         for i in range(batch_size):
#             num_query_i = len(query_src_list[i]) if i < len(query_src_list) else 0; num_query_list.append(num_query_i)
#             if num_query_i == 0:
#                  marginal_list.append(0.5)
#                  e_t_mask_list.append(torch.BoolTensor([])) # Append empty tensor
#                  continue 
            
#             num_new_nodes_i = num_new_nodes_list[i] if i < len(num_new_nodes_list) else 0; num_src_cand_i = num_query_i // num_new_nodes_i if num_new_nodes_i > 0 else 0
#             mean_in_deg_i = min(self.edge_diffusion.avg_in_deg, num_src_cand_i) if num_src_cand_i > 0 else 0; marginal_i = mean_in_deg_i / num_src_cand_i if num_src_cand_i > 0 else 0.0; marginal_list.append(marginal_i)
#             prior_i = torch.full((num_query_i,), marginal_i); e_t_mask_i_flat = torch.bernoulli(prior_i)
#             if num_new_nodes_i > 0 and num_src_cand_i > 0: # Ensure reshape is valid
#                 e_t_mask_i = e_t_mask_i_flat.reshape(num_new_nodes_i, num_src_cand_i); isolated_mask = (e_t_mask_i.sum(dim=1) == 0)
#                 if isolated_mask.any(): 
#                   e_t_mask_i[isolated_mask, 0] = 1 
#                 e_t_mask_list.append(e_t_mask_i.reshape(-1))
#             else: # Append flat mask if reshape is not possible
#                  e_t_mask_list.append(e_t_mask_i_flat)
#         # --- End FIX ---
        
#         e_t_mask = torch.cat(e_t_mask_list).bool().to(device) if e_t_mask_list else torch.BoolTensor([]).to(device)
#         num_nodes = len(batch_x_n); num_queries = len(batch_query_src)
        
#         if num_queries == 0: return edge_index_list
#         if e_t_mask.shape[0] != num_queries:
#              raise ValueError(f"Mismatch e_t_mask size ({e_t_mask.shape[0]}) vs num_queries ({num_queries}) right after init. "
#                              f"num_query_list: {num_query_list}, query_src_list lengths: {[len(q) for q in query_src_list]}")

#         batch_edge_index = self.get_batch_A(num_nodes_cumsum, edge_index_list, device, return_edge_index=True)
#         T_x_e = self.edge_diffusion.T
#         if max_num_steps_e is not None: T_x_e = min(T_x_e, max_num_steps_e)
#         time_x_e_list = list(reversed(range(0, T_x_e)))
#         if min_num_steps_e is not None and self.max_level is not None and self.max_level > 0: num_steps_e = min_num_steps_e + int((T_x_e - min_num_steps_e) * (curr_level / self.max_level)); time_x_e_list = time_x_e_list[-num_steps_e:] if num_steps_e > 0 else []
        
#         for s_x_e in time_x_e_list:
#             t_x_e = s_x_e + 1; alpha_t = self.edge_diffusion.alphas[t_x_e]; alpha_bar_s = self.edge_diffusion.alpha_bars[s_x_e]; alpha_bar_t = self.edge_diffusion.alpha_bars[t_x_e]
#             edge_index_t = torch.empty((2,0), dtype=torch.long, device=device)
            
#             valid_mask = e_t_mask[:num_queries] # Ensure mask is not longer than queries
#             if valid_mask.numel() > 0 and valid_mask.any():
#                  valid_query_src = batch_query_src[:valid_mask.shape[0]]
#                  valid_query_dst = batch_query_dst[:valid_mask.shape[0]]
#                  edge_index_t = torch.stack([
#                     valid_query_dst[valid_mask],
#                     valid_query_src[valid_mask]
#                  ]).to(device)

#             A = dglsp.spmatrix(torch.cat([batch_edge_index, edge_index_t], dim=1), shape=(num_nodes, num_nodes)).to(device)
#             t_x_e_tensor = torch.full((num_queries, 1), t_x_e, dtype=torch.long, device=device)
#             e_0_logits = self.edge_pred_model( A, batch_x_n, batch_abs_level, batch_rel_level, t_x_e_tensor, batch_query_src, batch_query_dst, h_y)
#             if e_0_logits.numel() == 0: continue

#             # --- [START] 关键修复：在 Softmax 之前处理 NAN/INF (同样适用于 EdgePred) ---
#             if torch.isnan(e_0_logits).any() or torch.isinf(e_0_logits).any():
#                 # print(f"Warning: NaN/Inf detected in edge_pred_model logits at step {s_x_e}. Clamping.")
#                 e_0_logits = torch.nan_to_num(e_0_logits, nan=0.0)
#                 e_0_logits = torch.clamp(e_0_logits, min=-1e4, max=1e4)
#             # --- [END] 关键修复 ---

#             # --- START: Syntax Masking ---
#             if x_n_l_list_active is not None:
#                 syntax_mask = torch.zeros_like(e_0_logits[:, 0]) # Shape [num_queries]
#                 query_offset = 0
#                 for i_graph in range(batch_size):
#                     num_new_nodes_i = num_new_nodes_list[i_graph]
#                     if num_new_nodes_i == 0: continue
#                     num_queries_i = num_query_list[i_graph]
#                     if num_queries_i == 0: continue
                    
#                     num_src_cand_i = num_queries_i // num_new_nodes_i
#                     node_types_i = x_n_l_list_active[i_graph]
#                     if node_types_i.ndim > 1: # Handle [N, 1] shape
#                          node_types_i = node_types_i.squeeze(-1)

#                     for j in range(num_new_nodes_i):
#                         node_type = node_types_i[j].item()
#                         q_start = query_offset + j * num_src_cand_i
#                         q_end = q_start + num_src_cand_i
#                         node_logits = e_0_logits[q_start:q_end, 1]

#                         # Rule 1: NOT (2) and BUF (8) must have exactly 1 input
#                         if node_type == 2 or node_type == 8: # Assuming 2 is NOT, 8 is BUF
#                             if node_logits.numel() > 0:
#                                 best_edge_idx = torch.argmax(node_logits)
#                                 syntax_mask[q_start:q_end] = -torch.inf
#                                 syntax_mask[q_start + best_edge_idx] = 0.0
                        
#                         # Rule 2: AND (1) must have exactly 2 inputs
#                         elif node_type == 1: # Assuming 1 is AND
#                             if node_logits.numel() >= 2:
#                                 best_indices = torch.topk(node_logits, 2).indices
#                                 syntax_mask[q_start:q_end] = -torch.inf
#                                 syntax_mask[q_start + best_indices] = 0.0
#                             # else: allow <2 inputs, converter script will handle
                        
#                     query_offset += num_queries_i
                
#                 e_0_logits = e_0_logits + syntax_mask.unsqueeze(-1)
#             # --- END: Syntax Masking ---

#             e_0_probs = e_0_logits.softmax(dim=-1)
#             if e_t_mask.shape[0] != num_queries: print(f"Warning: Mismatch e_t_mask size ({e_t_mask.shape[0]}) vs num_queries ({num_queries}). Skipping posterior."); continue
#             e_t_one_hot = F.one_hot(e_t_mask.long(), num_classes=2).float()
#             e_s_mask = self.posterior_edge(e_t_one_hot, alpha_t, alpha_bar_s, alpha_bar_t, e_0_probs, marginal_list, num_new_nodes_list, num_query_list)
#             e_t_mask = e_s_mask

#         if e_t_mask.numel() != num_queries: print(f"Warning: Final edge mask size ({e_t_mask.numel()}) mismatch with total queries ({num_queries}). Cannot reconstruct."); return edge_index_list
#         num_query_cumsum = torch.cumsum(torch.tensor([0] + num_query_list), dim=0)
#         edge_index_list_ = []
#         for i in range(batch_size):
#             original_edges_i = edge_index_list[i] if i < len(edge_index_list) else torch.empty((2,0), dtype=torch.long, device=device)
#             start_q, end_q = num_query_cumsum[i], num_query_cumsum[i+1]; mask_i = e_t_mask[start_q:end_q]
#             query_src_i = query_src_list[i] if i < len(query_src_list) else torch.LongTensor([]); query_dst_i = query_dst_list[i] if i < len(query_dst_list) else torch.LongTensor([])
#             if mask_i.numel() != query_src_i.numel():
#                  print(f"Warning: Mask/Query mismatch in final step graph {i}. Mask: {mask_i.numel()}, Query: {query_src_i.numel()}. Skipping edges.")
#                  sampled_src_i = torch.LongTensor([])
#                  sampled_dst_i = torch.LongTensor([])
#             else:
#                  sampled_src_i = query_src_i[mask_i]; sampled_dst_i = query_dst_i[mask_i]
#             new_edges_i = torch.stack([sampled_dst_i, sampled_src_i]) if sampled_src_i.numel() > 0 else torch.empty((2,0), dtype=torch.long, device=device)
#             combined_edges_i = torch.cat([original_edges_i, new_edges_i], dim=1); edge_index_list_.append(combined_edges_i)
#         return edge_index_list_

#     def get_batch_A(self, num_nodes_cumsum, edge_index_list, device, return_edge_index=False):
# # ... (代码保持不变) ...
#         batch_size = len(edge_index_list); edge_index_list_ = []
#         for i in range(batch_size):
#             if i < len(num_nodes_cumsum) and isinstance(edge_index_list[i], torch.Tensor): edge_index_list_.append(edge_index_list[i] + num_nodes_cumsum[i])
#         if not edge_index_list_: batch_edge_index = torch.empty((2,0), dtype=torch.long, device=device)
#         else: batch_edge_index = torch.cat(edge_index_list_, dim=1)
#         if return_edge_index: return batch_edge_index
#         N = num_nodes_cumsum[-1].item() if len(num_nodes_cumsum) > 0 else 0; shape = (N, N) if N > 0 else (0, 0)
#         batch_edge_index_dev = batch_edge_index.to(device); batch_A = dglsp.spmatrix(batch_edge_index_dev, shape=shape)
#         return batch_A
#     def get_batch_A_n2g(self, num_nodes_cumsum, device):
# # ... (代码保持不变) ...
#         batch_size = len(num_nodes_cumsum) - 1
#         if batch_size < 0: return dglsp.spmatrix(torch.empty((2,0), dtype=torch.long, device=device), shape=(0, 0))
#         nids, gids = [], []
#         for i in range(batch_size):
#             start_node = num_nodes_cumsum[i]; end_node = num_nodes_cumsum[i+1]
#             if end_node > start_node: nids.append(torch.arange(start_node, end_node).long()); gids.append(torch.ones(end_node - start_node).fill_(i).long())
#         N = num_nodes_cumsum[-1].item() if len(num_nodes_cumsum) > 0 else 0; shape = (batch_size, N) if N > 0 and batch_size >= 0 else (max(0, batch_size), 0)
#         if not nids: n2g_index = torch.empty((2,0), dtype=torch.long)
#         else: nids = torch.cat(nids, dim=0); gids = torch.cat(gids, dim=0); n2g_index = torch.stack([gids, nids])
#         n2g_index_dev = n2g_index.to(device); batch_A_n2g = dglsp.spmatrix(n2g_index_dev, shape=shape)
#         return batch_A_n2g
#     def get_batch_y(self, y_list, x_n_list, device):
# # ... (代码保持不变) ...
#         if self.y_encoder is None or y_list is None: return None
#         processed_x_n_list = []
#         if x_n_list and isinstance(x_n_list, list) and len(x_n_list) > 0 and not isinstance(x_n_list[0], torch.Tensor): # Check if list of non-tensors
#              try: processed_x_n_list = [torch.tensor(x, dtype=torch.long, device=device) for x in x_n_list]
#              except Exception as e: print(f"Warning: Could not convert x_n_list in get_batch_y: {e}"); processed_x_n_list = x_n_list
#         else: processed_x_n_list = x_n_list # Assume it's a list of tensors or empty
#         h_y_graphs = self.y_encoder(y_list, device)
#         h_y_per_node_list = []
#         list_len = min(len(y_list), len(processed_x_n_list))
#         if h_y_graphs.shape[0] < list_len: list_len = h_y_graphs.shape[0]
#         for i in range(list_len):
#             num_nodes = len(processed_x_n_list[i])
#             if num_nodes > 0: h_y_per_node_list.append(h_y_graphs[i].expand(num_nodes, -1))
#         if not h_y_per_node_list:
#              encoder_output_size = self.y_encoder.output_size if self.y_encoder else 0
#              return torch.empty((0, encoder_output_size), device=device, dtype=torch.float)
#         else: batch_h_y = torch.cat(h_y_per_node_list, dim=0).to(device); return batch_h_y

#     @torch.no_grad()
#     def sample(self,
#                device,
#                batch_size=1,
#                y=None, # Expects list of y tensors
#                min_num_steps_n=None,
#                max_num_steps_n=None,
#                min_num_steps_e=None,
#                max_num_steps_e=None):
#         if y is not None:
#             if not isinstance(y, list): raise TypeError("Conditional input 'y' must be a list of tensors.")
#             batch_size = len(y) # Override batch_size if y is provided
#         y_list = y

#         edge_index_list = [torch.empty((2,0), dtype=torch.long, device=device) for _ in range(batch_size)]
#         x_n_list = [] # List of node feature tensors for each graph
#         abs_level_list = [] # List of absolute level tensors

#         # --- FIX: Ensure init_x_n is always 2D ---
#         init_feature_dim = 1 # Default
#         if isinstance(self.dummy_x_n, int):
#              init_feature_dim = 1
#         elif isinstance(self.dummy_x_n, torch.Tensor):
#              if self.dummy_x_n.ndim == 1: init_feature_dim = self.dummy_x_n.shape[0] # [F]
#              elif self.dummy_x_n.ndim > 1: init_feature_dim = self.dummy_x_n.shape[-1] # [?, F]
        
#         for i in range(batch_size):
#              if isinstance(self.dummy_x_n, int):
#                  init_x_n = torch.tensor([[self.dummy_x_n]], dtype=torch.long, device=device) # Shape [1, 1]
#              elif isinstance(self.dummy_x_n, torch.Tensor):
#                  init_x_n = self.dummy_x_n.to(device)
#                  if init_x_n.ndim == 0: init_x_n = init_x_n.view(1, 1) # [1, 1]
#                  elif init_x_n.ndim == 1: init_x_n = init_x_n.unsqueeze(0) # Shape [1, F]
#                  if init_x_n.shape[0] != 1: init_x_n = init_x_n[0].unsqueeze(0) # Take first row
#              else:
#                   init_x_n = torch.tensor([[0]], dtype=torch.long, device=device) # Default dummy [1,1]
#              x_n_list.append(init_x_n); abs_level_list.append(torch.tensor([[0.]], device=device))
#         # --- End FIX ---
        
#         init_feature_dim = x_n_list[0].shape[1] # Get feature dim from the created dummy node


#         level = 0.; edge_index_finished, x_n_finished, y_finished = [], [], []
        
#         # --- FIX: Revert to original list replacement logic ---
#         while True: # Loop until all graphs are finished
#              current_batch_size = len(x_n_list)
#              if current_batch_size == 0:
#                  break # All graphs finished

#              # --- Prepare batch data ---
#              num_nodes_per_active_graph = [len(x) for x in x_n_list]
#              num_nodes_cumsum = torch.cumsum(torch.tensor([0] + num_nodes_per_active_graph), dim=0).to(device)
#              total_active_nodes = num_nodes_cumsum[-1].item()

#              batch_x_n = torch.cat(x_n_list, dim=0).to(device)
#              batch_abs_level = torch.cat(abs_level_list, dim=0).to(device)
#              batch_rel_level = batch_abs_level.max() - batch_abs_level if batch_abs_level.numel() > 0 else batch_abs_level
             
#              batch_A = self.get_batch_A(num_nodes_cumsum, edge_index_list, device)
#              batch_A_n2g = self.get_batch_A_n2g(num_nodes_cumsum, device)
#              batch_h_y = self.get_batch_y(y_list, x_n_list, device) if y_list is not None else None

#              # --- Sample Node Layer ---
#              x_n_l_list_active = self.sample_node_layer(
#                  batch_A, batch_x_n, batch_abs_level, batch_rel_level,
#                  batch_A_n2g, curr_level=level, h_y=batch_h_y,
#                  min_num_steps_n=min_num_steps_n, max_num_steps_n=max_num_steps_n)

#              # --- Prepare lists for next iteration ---
#              next_edge_index_list, next_x_n_list, next_abs_level_list = [], [], []
#              query_src_list_step, query_dst_list_step, num_new_nodes_list_step = [], [], []
#              batch_query_src_step, batch_query_dst_step = [], []
#              node_count_step = 0
#              next_y_list = [] if y_list is not None else None
             
#              # --- MODIFICATION: Track new node types for edge sampling ---
#              x_n_l_list_for_edge_sampling = [] # List of new node type tensors

#              # --- Iterate over *currently active* graphs ---
#              for i_active, x_n_l_i in enumerate(x_n_l_list_active):
#                   num_new_nodes_i = len(x_n_l_i)
                  
#                   if num_new_nodes_i == 0: # Graph finished
#                       if len(x_n_list[i_active]) > 1: # Don't save empty/dummy graph
#                            edge_index_finished.append(edge_index_list[i_active] - 1)
#                            x_n_finished.append(x_n_list[i_active][1:])
#                            if y_list is not None: y_finished.append(y_list[i_active])
                  
#                   else: # Graph continues
#                       N_old_i = len(x_n_list[i_active])
                      
#                       x_n_l_i_dev = x_n_l_i.to(device)
#                       # Ensure 2D
#                       if x_n_l_i_dev.ndim == 1: x_n_l_i_dev = x_n_l_i_dev.unsqueeze(-1)
                      
#                       existing_x_n = x_n_list[i_active]
#                       if existing_x_n.ndim == 1: existing_x_n = existing_x_n.unsqueeze(-1)
                          
#                       if existing_x_n.shape[1] != x_n_l_i_dev.shape[1]:
#                            raise ValueError(f"Feature dimension mismatch in sample loop: "
#                                             f"Existing nodes {existing_x_n.shape} vs New nodes {x_n_l_i_dev.shape}")
                      
#                       next_x_n = torch.cat([existing_x_n, x_n_l_i_dev], dim=0)
#                       next_level = torch.full((num_new_nodes_i, 1), level + 1.0, device=device); next_abs_level = torch.cat([abs_level_list[i_active], next_level])
                      
#                       next_x_n_list.append(next_x_n)
#                       next_abs_level_list.append(next_abs_level)
#                       next_edge_index_list.append(edge_index_list[i_active]) # Carry over old edges
#                       if y_list is not None: next_y_list.append(y_list[i_active]) # Carry over label
                      
#                       # --- MODIFICATION: Store new node types ---
#                       x_n_l_list_for_edge_sampling.append(x_n_l_i_dev) # Store the new types
                      
#                       if level >= 0: # Start sampling edges
#                           src_candidates_i = torch.arange(1, N_old_i, device=device)
#                           dst_nodes_i = torch.arange(N_old_i, N_old_i + num_new_nodes_i, device=device)
                          
#                           if src_candidates_i.numel() > 0:
#                               query_src_i = src_candidates_i.repeat_interleave(num_new_nodes_i)
#                               query_dst_i = dst_nodes_i.repeat(len(src_candidates_i))
                              
#                               query_src_list_step.append(query_src_i) # Local indices
#                               query_dst_list_step.append(query_dst_i) # Local indices
#                               num_new_nodes_list_step.append(num_new_nodes_i)
                              
#                               batch_query_src_step.append(query_src_i + node_count_step) # Batch-global indices
#                               batch_query_dst_step.append(query_dst_i + node_count_step) # Batch-global indices
#                           else:
#                               # Graph has no previous nodes to connect from
#                               query_src_list_step.append(torch.LongTensor([]).to(device))
#                               query_dst_list_step.append(torch.LongTensor([]).to(device))
#                               num_new_nodes_list_step.append(num_new_nodes_i)
                          
#                       node_count_step += N_old_i + num_new_nodes_i

#              # --- Replace the old active lists with the new active lists ---
#              edge_index_list = next_edge_index_list
#              x_n_list = next_x_n_list
#              abs_level_list = next_abs_level_list
#              y_list = next_y_list # Update y_list as well
             
#              level += 1.0
             
#              if len(edge_index_list) == 0: # If new list is empty, all graphs are done
#                  break
             
#              # --- Sample Edge Layer ---
#              if level > 0 and batch_query_src_step: # Only sample if queries were generated
#                  # --- FIX: Filter *all* inputs to sample_edge_layer ---
#                  # We need to find which graphs *in the new lists* correspond to the queries
#                  # This logic is complex. Revert to simpler logic:
#                  # We assume batch_query_src_step only contains queries for graphs that
#                  # *are* in the next_..._lists.
                 
#                  current_batch_size_edge = len(x_n_list) # Use the new batch size
                 
#                  active_edge_index_edge = edge_index_list
#                  active_x_n_edge = x_n_list
#                  active_abs_level_edge = abs_level_list
#                  active_y_edge = y_list if y_list is not None else None

#                  num_nodes_per_active_graph_edge = [len(x) for x in active_x_n_edge]
#                  num_nodes_cumsum_edge = torch.cumsum(torch.tensor([0] + num_nodes_per_active_graph_edge), dim=0).to(device)
                 
#                  if not active_x_n_edge: continue
                 
#                  feature_dim_edge = active_x_n_edge[0].shape[1] if active_x_n_edge[0].ndim > 1 else 1

#                  batch_x_n_edge = torch.cat(active_x_n_edge).to(device)
#                  if batch_x_n_edge.ndim == 1: batch_x_n_edge = batch_x_n_edge.unsqueeze(-1)
                 
#                  batch_abs_level_edge = torch.cat(active_abs_level_edge).to(device)
#                  batch_rel_level_edge = batch_abs_level_edge.max() - batch_abs_level_edge if batch_abs_level_edge.numel() > 0 else batch_abs_level_edge
#                  batch_h_y_edge = self.get_batch_y(active_y_edge, active_x_n_edge, device) if active_y_edge is not None else None
#                  batch_query_src = torch.cat(batch_query_src_step).to(device)
#                  batch_query_dst = torch.cat(batch_query_dst_step).to(device)

#                  if batch_query_src.numel() == 0: continue

#                  updated_active_edge_indices = self.sample_edge_layer(
#                      num_nodes_cumsum_edge, active_edge_index_edge, batch_x_n_edge, batch_abs_level_edge,
#                      batch_rel_level_edge, num_new_nodes_list_step, batch_query_src,
#                      batch_query_dst, query_src_list_step, query_dst_list_step, batch_h_y_edge,
#                      curr_level=level,
#                      min_num_steps_e=min_num_steps_e,
#                      max_num_steps_e=max_num_steps_e,
#                      # --- MODIFICATION: Pass new node types ---
#                      x_n_l_list_active=x_n_l_list_for_edge_sampling
#                  )
                 
#                  # --- Direct Replacement ---
#                  edge_index_list = updated_active_edge_indices
#                  # --- End Replacement ---


#              if self.max_level is not None and level >= self.max_level:
#                  break
        
#         # Add any graphs remaining in the active lists (if max_level reached)
#         for i_active in range(len(x_n_list)): # Use index into the final active lists
#              if len(x_n_list[i_active]) > 1: # Check based on final list state
#                   edge_index_finished.append(edge_index_list[i_active] - 1); x_n_finished.append(x_n_list[i_active][1:])
#                   if y_list is not None: y_finished.append(y_list[i_active])

#         if y is None: return edge_index_finished, x_n_finished
#         else: final_y = y_finished; return edge_index_finished, x_n_finished, final_y
#     # --- End FIX in sample ---
# # 

import dgl.sparse as dglsp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

__all__ = [
    'LayerDAG'
]

# ... (TruthTableEncoder, SinusoidalPE, BiMPNNLayer, OneHotPE, MultiEmbedding, BiMPNNEncoder, GraphClassifier, TransformerLayer, NodePredModel, EdgePredModel 保持不变) ...
class TruthTableEncoder(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # [B, 16, H/2, W/2]
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, H/4, W/4]
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), # Pool to a fixed size [B, 32, 4, 4]
        )
        self.mlp = nn.Linear(32 * 4 * 4, output_size)
    def forward(self, y_list, device):
        if not y_list or all(y.numel() == 0 for y in y_list):
            return torch.zeros(len(y_list), self.output_size, device=device)
        max_rows = max(y.shape[0] for y in y_list if y.numel() > 0) if any(y.numel() > 0 for y in y_list) else 1
        max_cols = max(y.shape[1] for y in y_list if y.numel() > 0) if any(y.numel() > 0 for y in y_list) else 1
        padded_tensors = []
        for y in y_list:
            if y.numel() == 0:
                padded_tensors.append(torch.zeros(max_rows, max_cols, device=device, dtype=torch.float))
                continue
            padded = F.pad(y.float(), (0, max_cols - y.shape[1], 0, max_rows - y.shape[0]))
            padded_tensors.append(padded)
        y_batch = torch.stack(padded_tensors).float()
        y_batch = y_batch.unsqueeze(1)
        y_batch = y_batch.to(device)
        cnn_out = self.cnn(y_batch)
        cnn_flat = cnn_out.view(cnn_out.size(0), -1)
        y_emb = self.mlp(cnn_flat)
        return y_emb
class SinusoidalPE(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, pe_size):
        super().__init__()
        self.pe_size = pe_size
        if pe_size > 0:
            self.div_term = torch.exp(torch.arange(0, pe_size, 2) *
                                      (-math.log(10000.0) / pe_size))
            self.div_term = nn.Parameter(self.div_term, requires_grad=False)
    def forward(self, position):
        if self.pe_size == 0:
            return torch.zeros(len(position), 0).to(position.device)
        position = position.float()
        return torch.cat([
            torch.sin(position * self.div_term),
            torch.cos(position * self.div_term)
        ], dim=-1)
class BiMPNNLayer(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = nn.Linear(in_size, out_size)
        self.W_trans = nn.Linear(in_size, out_size)
        self.W_self = nn.Linear(in_size, out_size)
    def forward(self, A, A_T, h_n):
        num_nodes_h = h_n.shape[0]; num_nodes_A = A.shape[0]
        if num_nodes_A != num_nodes_h:
             raise ValueError(f"Shape mismatch detected before SpMM: A has {num_nodes_A} nodes, h_n has {num_nodes_h} nodes.")
        if A.nnz > 0:
            A_row, A_col = A.coo()
            max_A_row = A_row.max().item() if A_row.numel() > 0 else -1
            max_A_col = A_col.max().item() if A_col.numel() > 0 else -1
            if max_A_row >= num_nodes_A or max_A_col >= num_nodes_A:
                 raise IndexError(f"Index out of bounds detected in A matrix before SpMM.")
        if A_T.nnz > 0:
            AT_row, AT_col = A_T.coo()
            max_AT_row = AT_row.max().item() if AT_row.numel() > 0 else -1
            max_AT_col = AT_col.max().item() if AT_col.numel() > 0 else -1
            if max_AT_row >= num_nodes_A or max_AT_col >= num_nodes_A:
                 raise IndexError(f"Index out of bounds detected in A_T matrix before SpMM.")
        if torch.isnan(h_n).any() or torch.isinf(h_n).any():
            raise ValueError("NaN/Inf in input features to BiMPNNLayer")
        try:
            h_n_w = self.W(h_n); h_n_w_trans = self.W_trans(h_n); h_n_w_self = self.W_self(h_n)
            if A.nnz == 0: h_n_out = h_n_w_self
            else: term1 = A @ h_n_w; term2 = A_T @ h_n_w_trans; h_n_out = term1 + term2 + h_n_w_self
        except RuntimeError as e:
            print(f"--- Runtime Error likely during SpMM ---"); print(f"A shape: {A.shape}, A nnz: {A.nnz}"); print(f"h_n shape: {h_n.shape}"); print(f"Error Message: {e}"); print("--- End Runtime Error Info ---")
            raise e
        if torch.isnan(h_n_out).any() or torch.isinf(h_n_out).any():
             print("ERROR: NaN or Inf detected in h_n_out after SpMM/Add!")
        return F.gelu(h_n_out)
class OneHotPE(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, pe_size):
        super().__init__()
        self.pe_size = pe_size
    def forward(self, position):
        if self.pe_size == 0:
            return torch.zeros(len(position), 0).to(position.device)
        return F.one_hot(position.clamp(max=self.pe_size - 1).long().squeeze(-1),
                         num_classes=self.pe_size).float() # Convert to float
class MultiEmbedding(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, num_x_n_cat, hidden_size):
        super().__init__()
        if isinstance(num_x_n_cat, torch.Tensor): num_cats = num_x_n_cat.tolist() if num_x_n_cat.numel() > 1 else [num_x_n_cat.item()]
        elif isinstance(num_x_n_cat, int): num_cats = [num_x_n_cat]
        elif isinstance(num_x_n_cat, list): num_cats = num_x_n_cat
        else: raise TypeError(f"Unsupported type for num_x_n_cat: {type(num_x_n_cat)}")
        self.emb_list = nn.ModuleList([ nn.Embedding(int(num_i), hidden_size) for num_i in num_cats ])
    def forward(self, x_n_cat):
        if x_n_cat.numel() == 0:
            output_dim = len(self.emb_list) * self.emb_list[0].embedding_dim if len(self.emb_list) > 0 else 0
            if len(self.emb_list) == 1: output_dim = self.emb_list[0].embedding_dim
            return torch.empty((0, output_dim), device=x_n_cat.device, dtype=torch.float)
        if len(self.emb_list) == 1:
             if x_n_cat.ndim == 2 and x_n_cat.shape[1] == 1: x_n_cat = x_n_cat.squeeze(1)
             if x_n_cat.ndim != 1: raise ValueError(f"MultiEmbedding expected 1D input when single feature, got shape {x_n_cat.shape}")
             x_n_emb = self.emb_list[0](x_n_cat)
        else:
            if x_n_cat.ndim != 2 or x_n_cat.shape[1] != len(self.emb_list): raise ValueError(f"MultiEmbedding expected 2D input with shape (*, {len(self.emb_list)}), got shape {x_n_cat.shape}")
            embeddings = []
            for i in range(len(self.emb_list)):
                 max_idx = self.emb_list[i].num_embeddings; indices = x_n_cat[:, i]
                 if (indices >= max_idx).any() or (indices < 0).any(): raise IndexError(f"Index out of bounds for embedding layer {i}. Max index allowed: {max_idx-1}, Got indices between {indices.min()} and {indices.max()}")
                 embeddings.append(self.emb_list[i](indices))
            x_n_emb = torch.cat(embeddings, dim=1)
        return x_n_emb
class BiMPNNEncoder(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, num_x_n_cat, x_n_emb_size, pe_emb_size, hidden_size, num_mpnn_layers, pe=None, y_emb_size=0, pool=None):
        super().__init__()
        self.pe = pe; self.pe_emb_size = pe_emb_size
        if self.pe in ['relative_level', 'abs_level']: self.level_emb = SinusoidalPE(pe_emb_size)
        elif self.pe == 'relative_level_one_hot': self.level_emb = OneHotPE(pe_emb_size)
        else: self.level_emb = None
        if isinstance(num_x_n_cat, torch.Tensor): num_feat_dims = len(num_x_n_cat.tolist()) if num_x_n_cat.numel() > 1 else 1
        elif isinstance(num_x_n_cat, int): num_feat_dims = 1
        elif isinstance(num_x_n_cat, list): num_feat_dims = len(num_x_n_cat)
        else: num_feat_dims = 0
        total_x_n_emb_size = num_feat_dims * x_n_emb_size
        self.x_n_emb = MultiEmbedding(num_x_n_cat, x_n_emb_size)
        actual_input_size = total_x_n_emb_size
        if self.level_emb is not None: actual_input_size += pe_emb_size
        actual_input_size += y_emb_size
        self.proj_input = nn.Sequential( nn.Linear(actual_input_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size) )
        self.mpnn_layers = nn.ModuleList()
        for _ in range(num_mpnn_layers): self.mpnn_layers.append(BiMPNNLayer(hidden_size, hidden_size))
        self.project_output_n = nn.Sequential( nn.Linear((num_mpnn_layers + 1) * hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size) )
        self.pool = pool
        if pool is not None: self.bn_g = nn.BatchNorm1d(hidden_size)
    def forward(self, A, x_n, abs_level, rel_level, h_y=None, A_n2g=None):
        if x_n.numel() == 0:
            output_size = self.project_output_n[-1].out_features
            if self.pool is None: return torch.empty((0, output_size), device=A.device, dtype=torch.float)
            else: batch_size = A_n2g.shape[0] if A_n2g is not None else 0; return torch.empty((batch_size, output_size), device=A.device, dtype=torch.float)
        A_T = A.T; h_n_initial = self.x_n_emb(x_n); node_pe = None
        if self.level_emb is not None:
            if self.pe == 'abs_level' and abs_level.numel() > 0: node_pe = self.level_emb(abs_level)
            elif self.pe in ['relative_level', 'relative_level_one_hot'] and rel_level.numel() > 0: node_pe = self.level_emb(rel_level)
            if node_pe is not None and node_pe.numel() == 0: node_pe = None
        features_to_concat = [h_n_initial]
        if node_pe is not None:
             if node_pe.shape[0] != h_n_initial.shape[0]:
                   if node_pe.shape[0] == 1: node_pe = node_pe.expand(h_n_initial.shape[0], -1)
                   else: raise ValueError(f"Shape mismatch: h_n has {h_n_initial.shape[0]} rows, PE has {node_pe.shape[0]} rows.")
             features_to_concat.append(node_pe)
        if h_y is not None:
             if h_y.shape[0] != h_n_initial.shape[0]: raise ValueError(f"Shape mismatch: h_n has {h_n_initial.shape[0]} rows, h_y (conditional embedding) has {h_y.shape[0]} rows. Check get_batch_y.")
             features_to_concat.append(h_y)
        h_n_combined = torch.cat(features_to_concat, dim=-1)
        h_n = self.proj_input(h_n_combined); h_n_cat = [h_n]
        for layer in self.mpnn_layers: h_n = layer(A, A_T, h_n); h_n_cat.append(h_n)
        h_n_final = torch.cat(h_n_cat, dim=-1); h_n_output = self.project_output_n(h_n_final)
        if self.pool is None: return h_n_output
        elif A_n2g is None: raise ValueError("A_n2g matrix is required for pooling but was not provided.")
        elif self.pool == 'sum': h_g = A_n2g @ h_n_output; return self.bn_g(h_g) if h_g.shape[0] > 0 and h_g.shape[1] > 0 else h_g
        elif self.pool == 'mean':
            h_g = A_n2g @ h_n_output; num_nodes_per_graph = A_n2g.sum(dim=1).unsqueeze(-1); num_nodes_per_graph = torch.clamp(num_nodes_per_graph, min=1.0)
            h_g = h_g / num_nodes_per_graph; return self.bn_g(h_g) if h_g.shape[0] > 0 and h_g.shape[1] > 0 else h_g
        else: raise ValueError(f"Unknown pooling type: {self.pool}")
class GraphClassifier(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, graph_encoder, emb_size, num_classes):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.predictor = nn.Sequential( nn.Linear(emb_size, emb_size), nn.GELU(), nn.Linear(emb_size, num_classes) )
    def forward(self, A, x_n, abs_level, rel_level, A_n2g, h_y=None):
        h_g = self.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y, A_n2g=A_n2g)
        if h_g.numel() == 0: return torch.empty((0, self.predictor[-1].out_features), device=h_g.device, dtype=h_g.dtype)
        pred_g = self.predictor(h_g); return pred_g
class TransformerLayer(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.to_v = nn.Linear(hidden_size, hidden_size); self.to_qk = nn.Linear(hidden_size, hidden_size * 2)
        self._reset_parameters()
        self.num_heads = num_heads; head_dim = hidden_size // num_heads; assert head_dim * num_heads == hidden_size
        self.scale = head_dim ** -0.5
        self.proj_new = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout)); self.norm1 = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Sequential(nn.Linear(hidden_size, 4 * hidden_size), nn.GELU(), nn.Linear(4 * hidden_size, hidden_size), nn.Dropout(dropout)); self.norm2 = nn.LayerNorm(hidden_size)
    def _reset_parameters(self): nn.init.xavier_uniform_(self.to_v.weight); nn.init.xavier_uniform_(self.to_qk.weight)
    def attn(self, q, k, v, num_query_cumsum):
        batch_size = len(num_query_cumsum) - 1; num_query_nodes = torch.diff(num_query_cumsum); max_num_nodes = num_query_nodes.max().item() if batch_size > 0 else 0
        q_padded = q.new_zeros(batch_size, max_num_nodes, q.shape[-1]); k_padded = k.new_zeros(batch_size, max_num_nodes, k.shape[-1]); v_padded = v.new_zeros(batch_size, max_num_nodes, v.shape[-1])
        pad_mask = q.new_zeros(batch_size, max_num_nodes).bool()
        for i in range(batch_size):
            q_padded[i, :num_query_nodes[i]] = q[num_query_cumsum[i]:num_query_cumsum[i + 1]]; k_padded[i, :num_query_nodes[i]] = k[num_query_cumsum[i]:num_query_cumsum[i + 1]]; v_padded[i, :num_query_nodes[i]] = v[num_query_cumsum[i]:num_query_cumsum[i + 1]]
            pad_mask[i, num_query_nodes[i]:] = True
        q_padded = rearrange(q_padded, 'b n (h d) -> b h n d', h=self.num_heads); k_padded = rearrange(k_padded, 'b n (h d) -> b h n d', h=self.num_heads); v_padded = rearrange(v_padded, 'b n (h d) -> b h n d', h=self.num_heads)
        dot = torch.matmul(q_padded, k_padded.transpose(-1, -2)) * self.scale; dot = dot.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_scores = F.softmax(dot, dim=-1); h_n_padded = torch.matmul(attn_scores, v_padded); h_n_padded = rearrange(h_n_padded, 'b h n d -> (b n) (h d)')
        pad_mask = (~pad_mask).reshape(-1); return h_n_padded[pad_mask]
    def forward(self, h_n, num_query_cumsum):
        v_n = self.to_v(h_n); q_n, k_n = self.to_qk(h_n).chunk(2, dim=-1); h_n_new = self.attn(q_n, k_n, v_n, num_query_cumsum); h_n_new = self.proj_new(h_n_new)
        h_n = self.norm1(h_n + h_n_new); h_n = self.norm2(h_n + self.out_proj(h_n)); return h_n
class NodePredModel(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, graph_encoder, num_x_n_cat, x_n_emb_size, t_emb_size, in_hidden_size, out_hidden_size, num_transformer_layers, num_heads, dropout):
        super().__init__()
        self.graph_encoder = graph_encoder
        if isinstance(num_x_n_cat, torch.Tensor): num_real_classes = num_x_n_cat - 1
        elif isinstance(num_x_n_cat, int): num_real_classes = torch.LongTensor([num_x_n_cat - 1])
        elif isinstance(num_x_n_cat, list): num_real_classes = torch.LongTensor(num_x_n_cat) - 1
        else: raise TypeError(f"Unsupported type for num_x_n_cat: {type(num_x_n_cat)}")
        num_real_classes = torch.clamp(num_real_classes, min=1)
        self.x_n_emb = MultiEmbedding(num_real_classes, x_n_emb_size)
        self.t_emb = SinusoidalPE(t_emb_size)
        num_feat_dims = len(num_real_classes.tolist()) if num_real_classes.numel() > 1 else 1
        actual_input_size = in_hidden_size + t_emb_size + num_feat_dims * x_n_emb_size
        self.project_h_n = nn.Sequential(nn.Linear(actual_input_size, out_hidden_size), nn.GELU())
        self.trans_layers = nn.ModuleList([TransformerLayer(out_hidden_size, num_heads, dropout) for _ in range(num_transformer_layers)])
        self.pred_list = nn.ModuleList([])
        num_real_classes_list = num_real_classes.tolist()
        for num_classes_f in num_real_classes_list:
            num_classes_f_valid = max(1, int(num_classes_f))
            self.pred_list.append(nn.Sequential(nn.Linear(out_hidden_size, out_hidden_size), nn.GELU(), nn.Linear(out_hidden_size, num_classes_f_valid)))
    def forward_with_h_g(self, h_g, x_n_t, t, query2g, num_query_cumsum):
        if h_g.numel() == 0 or x_n_t.numel() == 0: return [torch.empty(0)] * len(self.pred_list) # Handle empty input
        h_t = self.t_emb(t);
        if h_g.shape[0] == 1 and h_t.shape[0] > 1: h_t = h_t[0].unsqueeze(0)
        elif h_g.shape[0] > 1 and h_t.shape[0] == 1: h_t = h_t.expand(h_g.shape[0], -1)
        elif h_g.shape[0] != h_t.shape[0]: raise ValueError(f"Batch size mismatch h_g ({h_g.shape[0]}) vs h_t ({h_t.shape[0]})")
        h_g = torch.cat([h_g, h_t], dim=1);
        h_n_t_emb = self.x_n_emb(x_n_t)
        if query2g.numel() > 0 and query2g.max() >= h_g.shape[0]: raise IndexError(f"query2g index out of bounds: max index {query2g.max()}, h_g size {h_g.shape[0]}")
        h_n_t = torch.cat([h_n_t_emb, h_g[query2g]], dim=1); h_n_t = self.project_h_n(h_n_t);
        for trans_layer in self.trans_layers: h_n_t = trans_layer(h_n_t, num_query_cumsum)
        pred = [pred_layer(h_n_t) for pred_layer in self.pred_list]; return pred
    def forward(self, A, x_n, abs_level, rel_level, A_n2g, x_n_t, t, query2g, num_query_cumsum, h_y=None):
        h_g = self.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y, A_n2g=A_n2g)
        return self.forward_with_h_g(h_g, x_n_t, t, query2g, num_query_cumsum)
class EdgePredModel(nn.Module):
# ... (代码保持不变) ...
    def __init__(self, graph_encoder, t_emb_size, in_hidden_size, out_hidden_size): # in_hidden_size is GNN output size
        super().__init__()
        self.graph_encoder = graph_encoder
        self.t_emb = SinusoidalPE(t_emb_size)
        self.pred = nn.Sequential(
            nn.Linear(2 * in_hidden_size + t_emb_size, out_hidden_size), # Correct input size
            nn.GELU(),
            nn.Linear(out_hidden_size, 2)
        )
    def forward(self, A, x_n, abs_level, rel_level, t, query_src, query_dst, h_y=None):
        h_n = self.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y) # Removed A_n2g as it's not needed if pool=None
        if h_n.numel() == 0 or query_src.numel() == 0: return torch.empty((0, 2), device=h_n.device, dtype=h_n.dtype)
        max_idx = max(query_src.max().item() if query_src.numel()>0 else -1, query_dst.max().item() if query_dst.numel()>0 else -1)
        if max_idx >= h_n.shape[0]: raise IndexError(f"Query index {max_idx} out of bounds for h_n with size {h_n.shape[0]}")
        h_t = self.t_emb(t); num_queries = query_src.shape[0]
        if h_t.shape[0] == 1 and num_queries > 1: h_t = h_t.expand(num_queries, -1)
        elif h_t.shape[0] != num_queries: raise ValueError(f"Shape mismatch between t_emb ({h_t.shape[0]}) and queries ({num_queries})")
        h_e = torch.cat([h_t, h_n[query_src], h_n[query_dst]], dim=-1); return self.pred(h_e)
class LayerDAG(nn.Module):
# ... (代码保持不变) ...
    def __init__(self,
                 device,
                 num_x_n_cat,
                 node_count_encoder_config,
                 max_layer_size,
                 node_diffusion,
                 node_pred_graph_encoder_config,
                 node_predictor_config,
                 edge_diffusion,
                 edge_pred_graph_encoder_config,
                 edge_predictor_config,
                 max_level=None):
        super().__init__()
        if isinstance(num_x_n_cat, int): num_x_n_cat = torch.LongTensor([num_x_n_cat])
        elif isinstance(num_x_n_cat, list): num_x_n_cat = torch.LongTensor(num_x_n_cat)
        elif not isinstance(num_x_n_cat, torch.Tensor): raise TypeError(f"Unsupported type for num_x_n_cat: {type(num_x_n_cat)}")
        def get_num_feat_dims(n_cat):
            if isinstance(n_cat, torch.Tensor): return len(n_cat.tolist()) if n_cat.numel() > 1 else 1
            elif isinstance(n_cat, int): return 1
            elif isinstance(n_cat, list): return len(n_cat)
            return 0
        num_feat_dims = get_num_feat_dims(num_x_n_cat)
        nc_y_emb_size = node_count_encoder_config.get('y_emb_size', 0); np_y_emb_size = node_pred_graph_encoder_config.get('y_emb_size', 0); ep_y_emb_size = edge_pred_graph_encoder_config.get('y_emb_size', 0)
        max_y_emb_size = max(nc_y_emb_size, np_y_emb_size, ep_y_emb_size)
        self.y_encoder = TruthTableEncoder(max_y_emb_size).to(device) if max_y_emb_size > 0 else None
        nc_calculated_input_size = (num_feat_dims * node_count_encoder_config['x_n_emb_size'] + node_count_encoder_config.get('pe_emb_size', 0) + max_y_emb_size) # Use max_y_emb_size
        node_count_encoder = BiMPNNEncoder(num_x_n_cat, x_n_emb_size=node_count_encoder_config['x_n_emb_size'], pe_emb_size=node_count_encoder_config.get('pe_emb_size', 0), hidden_size=nc_calculated_input_size, num_mpnn_layers=node_count_encoder_config['num_mpnn_layers'], pe=node_count_encoder_config.get('pe', None), y_emb_size=max_y_emb_size, pool=node_count_encoder_config.get('pool', None)).to(device)
        self.node_count_model = GraphClassifier(node_count_encoder, emb_size=nc_calculated_input_size, num_classes=max_layer_size+1).to(device)
        self.node_diffusion = node_diffusion
        np_calculated_input_size = (num_feat_dims * node_pred_graph_encoder_config['x_n_emb_size'] + node_pred_graph_encoder_config.get('pe_emb_size', 0) + max_y_emb_size) # Use max_y_emb_size
        node_pred_graph_encoder = BiMPNNEncoder(num_x_n_cat, x_n_emb_size=node_pred_graph_encoder_config['x_n_emb_size'], pe_emb_size=node_pred_graph_encoder_config.get('pe_emb_size', 0), hidden_size=np_calculated_input_size, num_mpnn_layers=node_pred_graph_encoder_config['num_mpnn_layers'], pe=node_pred_graph_encoder_config.get('pe', None), y_emb_size=max_y_emb_size, pool=node_pred_graph_encoder_config.get('pool', None)).to(device)
        self.node_pred_model = NodePredModel(node_pred_graph_encoder, num_x_n_cat, node_pred_graph_encoder_config['x_n_emb_size'], in_hidden_size=np_calculated_input_size, **node_predictor_config).to(device)
        self.edge_diffusion = edge_diffusion
        ep_calculated_input_size = (num_feat_dims * edge_pred_graph_encoder_config['x_n_emb_size'] + edge_pred_graph_encoder_config.get('pe_emb_size', 0) + max_y_emb_size) # Use max_y_emb_size
        if edge_pred_graph_encoder_config.get('pool', None) is not None: print("Warning: Edge prediction graph encoder config has pooling enabled...")
        edge_pred_graph_encoder = BiMPNNEncoder(num_x_n_cat, x_n_emb_size=edge_pred_graph_encoder_config['x_n_emb_size'], pe_emb_size=edge_pred_graph_encoder_config.get('pe_emb_size', 0), hidden_size=ep_calculated_input_size, num_mpnn_layers=edge_pred_graph_encoder_config['num_mpnn_layers'], pe=edge_pred_graph_encoder_config.get('pe', None), y_emb_size=max_y_emb_size, pool=edge_pred_graph_encoder_config.get('pool', None)).to(device)
        self.edge_pred_model = EdgePredModel(edge_pred_graph_encoder, in_hidden_size=ep_calculated_input_size, **edge_predictor_config).to(device)
        self.max_level = max_level
        self.dummy_x_n = num_x_n_cat - 1
        if len(num_x_n_cat) > 1: self.dummy_x_n = num_x_n_cat - 1
        elif len(num_x_n_cat) == 1: dummy_val = (num_x_n_cat - 1)[0].item() if num_x_n_cat.numel() > 0 else 0; self.dummy_x_n = int(dummy_val)
    def get_batch_y(self, y_list, x_n_list, device):
# ... (代码保持不变) ...
        if self.y_encoder is None or y_list is None: return None
        processed_x_n_list = []
        if x_n_list and not isinstance(x_n_list[0], torch.Tensor):
             try: processed_x_n_list = [torch.tensor(x, dtype=torch.long, device=device) for x in x_n_list]
             except Exception as e: print(f"Warning: Could not convert x_n_list in get_batch_y: {e}"); processed_x_n_list = x_n_list
        else: processed_x_n_list = x_n_list
        h_y_graphs = self.y_encoder(y_list, device)
        h_y_per_node_list = []
        list_len = min(len(y_list), len(processed_x_n_list))
        if h_y_graphs.shape[0] < list_len: list_len = h_y_graphs.shape[0]
        for i in range(list_len):
            num_nodes = len(processed_x_n_list[i])
            if num_nodes > 0: h_y_per_node_list.append(h_y_graphs[i].expand(num_nodes, -1))
        if not h_y_per_node_list:
             encoder_output_size = self.y_encoder.output_size if self.y_encoder else 0
             return torch.empty((0, encoder_output_size), device=device, dtype=torch.float)
        else: batch_h_y = torch.cat(h_y_per_node_list, dim=0).to(device); return batch_h_y

    def posterior(self, Z_t, Q_t, Q_bar_s, Q_bar_t, Z_0):
        # ... (numerator/denominator calculation) ...
        left_term = Z_t @ torch.transpose(Q_t, -1, -2); left_term = left_term.unsqueeze(dim=-2); right_term = Q_bar_s.unsqueeze(dim=-3); numerator = left_term * right_term
        prod = Q_bar_t @ torch.transpose(Z_t, -1, -2); prod = torch.transpose(prod, -1, -2); denominator = prod.unsqueeze(-1); denominator = torch.clamp(denominator, min=1e-8)
        out = numerator / denominator
        prob = Z_0.unsqueeze(-1) * out
        prob = prob.sum(dim=-2)
        
        # --- FIX: Add normalization and clamping ---
        if torch.isnan(prob).any():
            # print("NaN detected in posterior (node) before normalization!")
            prob = torch.nan_to_num(prob, nan=0.0) # Replace NaNs with 0
        prob_sum = prob.sum(dim=-1, keepdim=True)
        prob = prob / (prob_sum + 1e-8) # Normalize
        prob = torch.clamp(prob, min=0.0, max=1.0) # Clamp
        # --- End FIX ---

        return prob
        
    def posterior_edge(self,Z_t,alpha_t,alpha_bar_s,alpha_bar_t,Z_0,marginal_list,num_new_nodes_list,num_query_list):
# ... (代码保持不变) ...
        batch_size = len(num_new_nodes_list)
        valid_query_indices = [i for i, nq in enumerate(num_query_list) if nq > 0]
        if not valid_query_indices: return torch.BoolTensor([])
        split_sizes = [num_query_list[i] for i in valid_query_indices]
        Z_t_list = torch.split(Z_t, split_sizes, dim=0); Z_0_list = torch.split(Z_0, split_sizes, dim=0)
        device = Z_t.device; e_mask_list = []
        for i_split, i_batch in enumerate(valid_query_indices):
            Z_t_i = Z_t_list[i_split]; Z_0_i = Z_0_list[i_split]; num_new_nodes_i = num_new_nodes_list[i_batch]
            current_marginal = marginal_list[i_batch] if i_batch < len(marginal_list) else 0.5
            Q_t_i, Q_bar_s_i, Q_bar_t_i = self.edge_diffusion.get_Qs(alpha_t, alpha_bar_s, alpha_bar_t, current_marginal)
            Q_t_i=Q_t_i.to(device); Q_bar_s_i=Q_bar_s_i.to(device); Q_bar_t_i=Q_bar_t_i.to(device)
            left_term_i = Z_t_i @ torch.transpose(Q_t_i, -1, -2); left_term_i = left_term_i.unsqueeze(dim=-2)
            right_term_i = Q_bar_s_i.unsqueeze(dim=-3); numerator_i = left_term_i * right_term_i
            prod_i = Q_bar_t_i @ torch.transpose(Z_t_i, -1, -2); prod_i = torch.transpose(prod_i, -1, -2)
            denominator_i = prod_i.unsqueeze(-1); denominator_i = torch.clamp(denominator_i, min=1e-8)
            out_i = numerator_i / denominator_i
            prob_i = Z_0_i.unsqueeze(-1) * out_i; prob_i = prob_i.sum(dim=-2)
            
            # --- FIX: Add NaN check before normalization ---
            if torch.isnan(prob_i).any():
                # print(f"NaN detected in posterior_edge (graph {i_batch}) before normalization!")
                prob_i = torch.nan_to_num(prob_i, nan=0.0) # Replace NaNs with 0
            # --- End FIX ---

            prob_i = prob_i / (prob_i.sum(dim=-1, keepdim=True) + 1e-6)
            prob_i = prob_i[:, 1]
            num_queries_i_actual = split_sizes[i_split]
            if num_new_nodes_i <= 0 or num_queries_i_actual % num_new_nodes_i != 0:
                 print(f"Warning: Cannot reshape posterior_edge probability for graph {i_batch}. Num new nodes: {num_new_nodes_i}, Num queries: {num_queries_i_actual}. Appending all-False mask.")
                 e_mask_list.append(torch.zeros(num_queries_i_actual, dtype=torch.bool, device=device))
                 continue
            num_src_candidates = num_queries_i_actual // num_new_nodes_i
            prob_i = prob_i.reshape(num_new_nodes_i, num_src_candidates)
            
            # --- FIX: Add clamping ---
            prob_i = torch.clamp(prob_i, min=0.0, max=1.0)
            # --- End FIX ---
            
            e_mask_i = torch.bernoulli(prob_i)
            isolated_mask_i = (e_mask_i.sum(dim=1) == 0)
            if isolated_mask_i.any(): highest_prob_idx = prob_i[isolated_mask_i].argmax(dim=1); e_mask_i[isolated_mask_i, highest_prob_idx] = 1
            e_mask_list.append(e_mask_i.reshape(-1))
        return torch.cat(e_mask_list).bool() if e_mask_list else torch.BoolTensor([])

    @torch.no_grad()
    def sample_node_layer(self,
# ... (代码保持不变) ...
                          A,
                          x_n,
                          abs_level,
                          rel_level,
                          A_n2g,
                          curr_level=None,
                          h_y=None, # Expect encoded y here
                          min_num_steps_n=None,
                          max_num_steps_n=None):
        device = A.device
        node_count_logits = self.node_count_model(A, x_n, abs_level, rel_level, A_n2g=A_n2g, h_y=h_y)
        if curr_level == 0: node_count_logits[:, 0] = float('-inf')
        node_count_probs = node_count_logits.softmax(dim=-1); num_new_nodes = node_count_probs.multinomial(1)
        num_new_nodes_total = num_new_nodes.sum().item(); batch_size = num_new_nodes.shape[0]
        if num_new_nodes_total == 0: return [torch.LongTensor([]).to(device) for _ in range(batch_size)]
        num_classes_list = self.node_diffusion.num_classes_list; marginal_list = self.node_diffusion.m_list; D = len(num_classes_list)
        x_n_t = []
        for d in range(D):
             if d >= len(marginal_list) or marginal_list[d].numel() == 0:
                  print(f"Warning: Marginal missing or empty for dim {d}. Sampling uniformly.")
                  num_classes_d = num_classes_list[d] if d < len(num_classes_list) else 1
                  prior_d = torch.ones(num_new_nodes_total, num_classes_d, device=device) / num_classes_d # Ensure on device
             else:
                  marginal_d = marginal_list[d]
                  prior_d = marginal_d[0].to(device).unsqueeze(0).expand(num_new_nodes_total, -1)
             x_n_t_d = prior_d.multinomial(1).squeeze(-1); x_n_t.append(x_n_t_d)
        x_n_t = torch.stack(x_n_t, dim=1) # Already on device
        h_g = self.node_pred_model.graph_encoder(A, x_n, abs_level, rel_level, h_y=h_y, A_n2g=A_n2g)
        num_new_nodes_list_cpu = num_new_nodes.squeeze(-1).cpu().tolist(); num_query_cumsum = torch.cumsum(torch.tensor([0] + num_new_nodes_list_cpu), dim=0).long().to(device)
        query2g = []
        for i in range(batch_size):
            num_nodes_i = num_query_cumsum[i+1] - num_query_cumsum[i]
            if num_nodes_i > 0: query2g.append(torch.full((num_nodes_i,), i, dtype=torch.long, device=device))
        query2g = torch.cat(query2g) if query2g else torch.LongTensor([]).to(device)
        T_x_n = self.node_diffusion.T
        if max_num_steps_n is not None: T_x_n = min(T_x_n, max_num_steps_n)
        time_x_n_list = list(reversed(range(0, T_x_n)))
        if min_num_steps_n is not None and self.max_level is not None and self.max_level > 0:
            num_steps_n = min_num_steps_n + int((T_x_n - min_num_steps_n) * (curr_level / self.max_level))
            time_x_n_list = time_x_n_list[-num_steps_n:] if num_steps_n > 0 else []
        for s_x_n in time_x_n_list:
            t_x_n = s_x_n + 1
            alpha_t = self.node_diffusion.alphas[t_x_n]; alpha_bar_s = self.node_diffusion.alpha_bars[s_x_n]; alpha_bar_t = self.node_diffusion.alpha_bars[t_x_n]
            t_x_n_tensor = torch.full((batch_size, 1), t_x_n, dtype=torch.long, device=device)
            x_n_0_logits = self.node_pred_model.forward_with_h_g( h_g, x_n_t, t_x_n_tensor, query2g, num_query_cumsum)
            if not x_n_0_logits or not x_n_0_logits[0].numel(): continue
            x_n_s = []
            for d in range(D):
                if d >= len(x_n_0_logits): continue

                # --- [START] 关键修复：在 Softmax 之前处理 NAN/INF ---
                # 检查模型输出的 logits 是否稳定
                logits_d = x_n_0_logits[d]
                if torch.isnan(logits_d).any() or torch.isinf(logits_d).any():
                    # print(f"Warning: NaN/Inf detected in node_pred_model logits at step {s_x_n}, dim {d}. Clamping.")
                    # 1. 用 0 替换 NAN
                    logits_d = torch.nan_to_num(logits_d, nan=0.0)
                    # 2. 用一个大/小数值替换 Inf/-Inf
                    logits_d = torch.clamp(logits_d, min=-1e4, max=1e4)
                # --- [END] 关键修复 ---

                Q_t_d = self.node_diffusion.get_Q(alpha_t, d).to(device); Q_bar_s_d = self.node_diffusion.get_Q(alpha_bar_s, d).to(device); Q_bar_t_d = self.node_diffusion.get_Q(alpha_bar_t, d).to(device)
                
                # [MODIFIED] 使用修复后的 logits
                x_n_0_probs_d = logits_d.softmax(dim=-1)
                
                if x_n_t.numel() > 0 and d < x_n_t.shape[1]:
                     num_classes_d = num_classes_list[d]
                     indices_d = x_n_t[:, d].clamp(0, num_classes_d - 1)
                     x_n_t_one_hot_d = F.one_hot(indices_d, num_classes=num_classes_d).float()
                     
                     # posterior 函数内部已经有 nan_to_num，但这里的 x_n_0_probs_d 现在应该是安全的
                     x_n_s_probs_d = self.posterior(x_n_t_one_hot_d, Q_t_d, Q_bar_s_d, Q_bar_t_d, x_n_0_probs_d)
                     
                     x_n_s_d = x_n_s_probs_d.multinomial(1).squeeze(-1); x_n_s.append(x_n_s_d)
                else:
                    print(f"Warning: Skipping sampling for dim {d} due to empty x_n_t or dim mismatch.")
                    num_classes_d = num_classes_list[d] if d < len(num_classes_list) else 1
                    x_n_s.append(torch.zeros(num_new_nodes_total, dtype=torch.long, device=device))
            if len(x_n_s) == D: x_n_t = torch.stack(x_n_s, dim=1)
        return list(torch.split(x_n_t, num_new_nodes_list_cpu))

    @torch.no_grad()
    def sample_edge_layer(self,
# ... (代码保持不变) ...
                          num_nodes_cumsum, edge_index_list,
                          batch_x_n, batch_abs_level, batch_rel_level,
                          num_new_nodes_list, batch_query_src, batch_query_dst,
                          query_src_list, query_dst_list,
                          h_y=None, # Expect encoded y here
                          curr_level=None,
                          min_num_steps_e=None,
                          max_num_steps_e=None,
                          x_n_l_list_active=None): # Accept new node types
        device = batch_x_n.device
        e_t_mask_list = []; batch_size = len(num_new_nodes_list); marginal_list = []; num_query_list = []
        
        # --- FIX: Ensure e_t_mask_list has correct number of elements ---
        for i in range(batch_size):
            num_query_i = len(query_src_list[i]) if i < len(query_src_list) else 0; num_query_list.append(num_query_i)
            if num_query_i == 0:
                 marginal_list.append(0.5)
                 e_t_mask_list.append(torch.BoolTensor([])) # Append empty tensor
                 continue 
            
            num_new_nodes_i = num_new_nodes_list[i] if i < len(num_new_nodes_list) else 0; num_src_cand_i = num_query_i // num_new_nodes_i if num_new_nodes_i > 0 else 0
            mean_in_deg_i = min(self.edge_diffusion.avg_in_deg, num_src_cand_i) if num_src_cand_i > 0 else 0; marginal_i = mean_in_deg_i / num_src_cand_i if num_src_cand_i > 0 else 0.0; marginal_list.append(marginal_i)
            prior_i = torch.full((num_query_i,), marginal_i); e_t_mask_i_flat = torch.bernoulli(prior_i)
            if num_new_nodes_i > 0 and num_src_cand_i > 0: # Ensure reshape is valid
                e_t_mask_i = e_t_mask_i_flat.reshape(num_new_nodes_i, num_src_cand_i); isolated_mask = (e_t_mask_i.sum(dim=1) == 0)
                if isolated_mask.any(): 
                  e_t_mask_i[isolated_mask, 0] = 1 
                e_t_mask_list.append(e_t_mask_i.reshape(-1))
            else: # Append flat mask if reshape is not possible
                 e_t_mask_list.append(e_t_mask_i_flat)
        # --- End FIX ---
        
        e_t_mask = torch.cat(e_t_mask_list).bool().to(device) if e_t_mask_list else torch.BoolTensor([]).to(device)
        num_nodes = len(batch_x_n); num_queries = len(batch_query_src)
        
        if num_queries == 0: return edge_index_list
        if e_t_mask.shape[0] != num_queries:
             raise ValueError(f"Mismatch e_t_mask size ({e_t_mask.shape[0]}) vs num_queries ({num_queries}) right after init. "
                             f"num_query_list: {num_query_list}, query_src_list lengths: {[len(q) for q in query_src_list]}")

        batch_edge_index = self.get_batch_A(num_nodes_cumsum, edge_index_list, device, return_edge_index=True)
        T_x_e = self.edge_diffusion.T
        if max_num_steps_e is not None: T_x_e = min(T_x_e, max_num_steps_e)
        time_x_e_list = list(reversed(range(0, T_x_e)))
        if min_num_steps_e is not None and self.max_level is not None and self.max_level > 0: num_steps_e = min_num_steps_e + int((T_x_e - min_num_steps_e) * (curr_level / self.max_level)); time_x_e_list = time_x_e_list[-num_steps_e:] if num_steps_e > 0 else []
        
        for s_x_e in time_x_e_list:
            t_x_e = s_x_e + 1; alpha_t = self.edge_diffusion.alphas[t_x_e]; alpha_bar_s = self.edge_diffusion.alpha_bars[s_x_e]; alpha_bar_t = self.edge_diffusion.alpha_bars[t_x_e]
            edge_index_t = torch.empty((2,0), dtype=torch.long, device=device)
            
            valid_mask = e_t_mask[:num_queries] # Ensure mask is not longer than queries
            if valid_mask.numel() > 0 and valid_mask.any():
                 valid_query_src = batch_query_src[:valid_mask.shape[0]]
                 valid_query_dst = batch_query_dst[:valid_mask.shape[0]]
                 edge_index_t = torch.stack([
                    valid_query_dst[valid_mask],
                    valid_query_src[valid_mask]
                 ]).to(device)

            A = dglsp.spmatrix(torch.cat([batch_edge_index, edge_index_t], dim=1), shape=(num_nodes, num_nodes)).to(device)
            t_x_e_tensor = torch.full((num_queries, 1), t_x_e, dtype=torch.long, device=device)
            e_0_logits = self.edge_pred_model( A, batch_x_n, batch_abs_level, batch_rel_level, t_x_e_tensor, batch_query_src, batch_query_dst, h_y)
            if e_0_logits.numel() == 0: continue

            # --- [START] 关键修复：在 Softmax 之前处理 NAN/INF (同样适用于 EdgePred) ---
            if torch.isnan(e_0_logits).any() or torch.isinf(e_0_logits).any():
                # print(f"Warning: NaN/Inf detected in edge_pred_model logits at step {s_x_e}. Clamping.")
                e_0_logits = torch.nan_to_num(e_0_logits, nan=0.0)
                e_0_logits = torch.clamp(e_0_logits, min=-1e4, max=1e4)
            # --- [END] 关键修复 ---

            # --- START: Syntax Masking ---
            if x_n_l_list_active is not None:
                syntax_mask = torch.zeros_like(e_0_logits[:, 0]) # Shape [num_queries]
                query_offset = 0
                for i_graph in range(batch_size):
                    num_new_nodes_i = num_new_nodes_list[i_graph]
                    if num_new_nodes_i == 0: continue
                    num_queries_i = num_query_list[i_graph]
                    if num_queries_i == 0: continue
                    
                    num_src_cand_i = num_queries_i // num_new_nodes_i
                    node_types_i = x_n_l_list_active[i_graph]
                    if node_types_i.ndim > 1: # Handle [N, 1] shape
                         node_types_i = node_types_i.squeeze(-1)

                    for j in range(num_new_nodes_i):
                        node_type = node_types_i[j].item()
                        q_start = query_offset + j * num_src_cand_i
                        q_end = q_start + num_src_cand_i
                        node_logits = e_0_logits[q_start:q_end, 1]

                        # Rule 1: NOT (2) and BUF (8) must have exactly 1 input
                        if node_type == 2 or node_type == 8: # Assuming 2 is NOT, 8 is BUF
                            if node_logits.numel() > 0:
                                best_edge_idx = torch.argmax(node_logits)
                                syntax_mask[q_start:q_end] = -torch.inf
                                syntax_mask[q_start + best_edge_idx] = 0.0
                        
                        # Rule 2: AND (1) must have exactly 2 inputs
                        elif node_type == 1: # Assuming 1 is AND
                            if node_logits.numel() >= 2:
                                best_indices = torch.topk(node_logits, 2).indices
                                syntax_mask[q_start:q_end] = -torch.inf
                                syntax_mask[q_start + best_indices] = 0.0
                            # else: allow <2 inputs, converter script will handle
                        
                    query_offset += num_queries_i
                
                e_0_logits = e_0_logits + syntax_mask.unsqueeze(-1)
            # --- END: Syntax Masking ---

            e_0_probs = e_0_logits.softmax(dim=-1)
            if e_t_mask.shape[0] != num_queries: print(f"Warning: Mismatch e_t_mask size ({e_t_mask.shape[0]}) vs num_queries ({num_queries}). Skipping posterior."); continue
            e_t_one_hot = F.one_hot(e_t_mask.long(), num_classes=2).float()
            e_s_mask = self.posterior_edge(e_t_one_hot, alpha_t, alpha_bar_s, alpha_bar_t, e_0_probs, marginal_list, num_new_nodes_list, num_query_list)
            e_t_mask = e_s_mask

        if e_t_mask.numel() != num_queries: print(f"Warning: Final edge mask size ({e_t_mask.numel()}) mismatch with total queries ({num_queries}). Cannot reconstruct."); return edge_index_list
        num_query_cumsum = torch.cumsum(torch.tensor([0] + num_query_list), dim=0)
        edge_index_list_ = []
        for i in range(batch_size):
            original_edges_i = edge_index_list[i] if i < len(edge_index_list) else torch.empty((2,0), dtype=torch.long, device=device)
            start_q, end_q = num_query_cumsum[i], num_query_cumsum[i+1]; mask_i = e_t_mask[start_q:end_q]
            query_src_i = query_src_list[i] if i < len(query_src_list) else torch.LongTensor([]); query_dst_i = query_dst_list[i] if i < len(query_dst_list) else torch.LongTensor([])
            if mask_i.numel() != query_src_i.numel():
                 print(f"Warning: Mask/Query mismatch in final step graph {i}. Mask: {mask_i.numel()}, Query: {query_src_i.numel()}. Skipping edges.")
                 sampled_src_i = torch.LongTensor([])
                 sampled_dst_i = torch.LongTensor([])
            else:
                 sampled_src_i = query_src_i[mask_i]; sampled_dst_i = query_dst_i[mask_i]
            new_edges_i = torch.stack([sampled_dst_i, sampled_src_i]) if sampled_src_i.numel() > 0 else torch.empty((2,0), dtype=torch.long, device=device)
            combined_edges_i = torch.cat([original_edges_i, new_edges_i], dim=1); edge_index_list_.append(combined_edges_i)
        return edge_index_list_

    def get_batch_A(self, num_nodes_cumsum, edge_index_list, device, return_edge_index=False):
# ... (代码保持不变) ...
        batch_size = len(edge_index_list); edge_index_list_ = []
        for i in range(batch_size):
            if i < len(num_nodes_cumsum) and isinstance(edge_index_list[i], torch.Tensor): edge_index_list_.append(edge_index_list[i] + num_nodes_cumsum[i])
        if not edge_index_list_: batch_edge_index = torch.empty((2,0), dtype=torch.long, device=device)
        else: batch_edge_index = torch.cat(edge_index_list_, dim=1)
        if return_edge_index: return batch_edge_index
        N = num_nodes_cumsum[-1].item() if len(num_nodes_cumsum) > 0 else 0; shape = (N, N) if N > 0 else (0, 0)
        batch_edge_index_dev = batch_edge_index.to(device); batch_A = dglsp.spmatrix(batch_edge_index_dev, shape=shape)
        return batch_A
    def get_batch_A_n2g(self, num_nodes_cumsum, device):
# ... (代码保持不变) ...
        batch_size = len(num_nodes_cumsum) - 1
        if batch_size < 0: return dglsp.spmatrix(torch.empty((2,0), dtype=torch.long, device=device), shape=(0, 0))
        nids, gids = [], []
        for i in range(batch_size):
            start_node = num_nodes_cumsum[i]; end_node = num_nodes_cumsum[i+1]
            if end_node > start_node: nids.append(torch.arange(start_node, end_node).long()); gids.append(torch.ones(end_node - start_node).fill_(i).long())
        N = num_nodes_cumsum[-1].item() if len(num_nodes_cumsum) > 0 else 0; shape = (batch_size, N) if N > 0 and batch_size >= 0 else (max(0, batch_size), 0)
        if not nids: n2g_index = torch.empty((2,0), dtype=torch.long)
        else: nids = torch.cat(nids, dim=0); gids = torch.cat(gids, dim=0); n2g_index = torch.stack([gids, nids])
        n2g_index_dev = n2g_index.to(device); batch_A_n2g = dglsp.spmatrix(n2g_index_dev, shape=shape)
        return batch_A_n2g
    def get_batch_y(self, y_list, x_n_list, device):
# ... (代码保持不变) ...
        if self.y_encoder is None or y_list is None: return None
        processed_x_n_list = []
        if x_n_list and isinstance(x_n_list, list) and len(x_n_list) > 0 and not isinstance(x_n_list[0], torch.Tensor): # Check if list of non-tensors
             try: processed_x_n_list = [torch.tensor(x, dtype=torch.long, device=device) for x in x_n_list]
             except Exception as e: print(f"Warning: Could not convert x_n_list in get_batch_y: {e}"); processed_x_n_list = x_n_list
        else: processed_x_n_list = x_n_list # Assume it's a list of tensors or empty
        h_y_graphs = self.y_encoder(y_list, device)
        h_y_per_node_list = []
        list_len = min(len(y_list), len(processed_x_n_list))
        if h_y_graphs.shape[0] < list_len: list_len = h_y_graphs.shape[0]
        for i in range(list_len):
            num_nodes = len(processed_x_n_list[i])
            if num_nodes > 0: h_y_per_node_list.append(h_y_graphs[i].expand(num_nodes, -1))
        if not h_y_per_node_list:
             encoder_output_size = self.y_encoder.output_size if self.y_encoder else 0
             return torch.empty((0, encoder_output_size), device=device, dtype=torch.float)
        else: batch_h_y = torch.cat(h_y_per_node_list, dim=0).to(device); return batch_h_y

    @torch.no_grad()
    def sample(self,
               device,
               batch_size=1,
               y=None, # Expects list of y tensors
               min_num_steps_n=None,
               max_num_steps_n=None,
               min_num_steps_e=None,
               max_num_steps_e=None):
        if y is not None:
            if not isinstance(y, list): raise TypeError("Conditional input 'y' must be a list of tensors.")
            batch_size = len(y) # Override batch_size if y is provided
        y_list = y

        edge_index_list = [torch.empty((2,0), dtype=torch.long, device=device) for _ in range(batch_size)]
        x_n_list = [] # List of node feature tensors for each graph
        abs_level_list = [] # List of absolute level tensors

        # --- FIX: Ensure init_x_n is always 2D ---
        init_feature_dim = 1 # Default
        if isinstance(self.dummy_x_n, int):
             init_feature_dim = 1
        elif isinstance(self.dummy_x_n, torch.Tensor):
             if self.dummy_x_n.ndim == 1: init_feature_dim = self.dummy_x_n.shape[0] # [F]
             elif self.dummy_x_n.ndim > 1: init_feature_dim = self.dummy_x_n.shape[-1] # [?, F]
        
        for i in range(batch_size):
             if isinstance(self.dummy_x_n, int):
                 init_x_n = torch.tensor([[self.dummy_x_n]], dtype=torch.long, device=device) # Shape [1, 1]
             elif isinstance(self.dummy_x_n, torch.Tensor):
                 init_x_n = self.dummy_x_n.to(device)
                 if init_x_n.ndim == 0: init_x_n = init_x_n.view(1, 1) # [1, 1]
                 elif init_x_n.ndim == 1: init_x_n = init_x_n.unsqueeze(0) # Shape [1, F]
                 if init_x_n.shape[0] != 1: init_x_n = init_x_n[0].unsqueeze(0) # Take first row
             else:
                  init_x_n = torch.tensor([[0]], dtype=torch.long, device=device) # Default dummy [1,1]
             x_n_list.append(init_x_n); abs_level_list.append(torch.tensor([[0.]], device=device))
        # --- End FIX ---
        
        init_feature_dim = x_n_list[0].shape[1] # Get feature dim from the created dummy node


        level = 0.; edge_index_finished, x_n_finished, y_finished = [], [], []
        
        # --- FIX: Revert to original list replacement logic ---
        while True: # Loop until all graphs are finished
             current_batch_size = len(x_n_list)
             if current_batch_size == 0:
                 break # All graphs finished

             # --- Prepare batch data ---
             num_nodes_per_active_graph = [len(x) for x in x_n_list]
             num_nodes_cumsum = torch.cumsum(torch.tensor([0] + num_nodes_per_active_graph), dim=0).to(device)
             total_active_nodes = num_nodes_cumsum[-1].item()

             batch_x_n = torch.cat(x_n_list, dim=0).to(device)
             batch_abs_level = torch.cat(abs_level_list, dim=0).to(device)
             batch_rel_level = batch_abs_level.max() - batch_abs_level if batch_abs_level.numel() > 0 else batch_abs_level
             
             batch_A = self.get_batch_A(num_nodes_cumsum, edge_index_list, device)
             batch_A_n2g = self.get_batch_A_n2g(num_nodes_cumsum, device)
             batch_h_y = self.get_batch_y(y_list, x_n_list, device) if y_list is not None else None

             # --- Sample Node Layer ---
             x_n_l_list_active = self.sample_node_layer(
                 batch_A, batch_x_n, batch_abs_level, batch_rel_level,
                 batch_A_n2g, curr_level=level, h_y=batch_h_y,
                 min_num_steps_n=min_num_steps_n, max_num_steps_n=max_num_steps_n)

             # --- Prepare lists for next iteration ---
             next_edge_index_list, next_x_n_list, next_abs_level_list = [], [], []
             query_src_list_step, query_dst_list_step, num_new_nodes_list_step = [], [], []
             batch_query_src_step, batch_query_dst_step = [], []
             node_count_step = 0
             next_y_list = [] if y_list is not None else None
             
             # --- MODIFICATION: Track new node types for edge sampling ---
             x_n_l_list_for_edge_sampling = [] # List of new node type tensors

             # --- Iterate over *currently active* graphs ---
             for i_active, x_n_l_i in enumerate(x_n_l_list_active):
                  num_new_nodes_i = len(x_n_l_i)
                  
                  if num_new_nodes_i == 0: # Graph finished
                      if len(x_n_list[i_active]) > 1: # Don't save empty/dummy graph
                           edge_index_finished.append(edge_index_list[i_active] - 1)
                           x_n_finished.append(x_n_list[i_active][1:])
                           if y_list is not None: y_finished.append(y_list[i_active])
                  
                  else: # Graph continues
                      N_old_i = len(x_n_list[i_active])
                      
                      x_n_l_i_dev = x_n_l_i.to(device)
                      # Ensure 2D
                      if x_n_l_i_dev.ndim == 1: x_n_l_i_dev = x_n_l_i_dev.unsqueeze(-1)
                      
                      existing_x_n = x_n_list[i_active]
                      if existing_x_n.ndim == 1: existing_x_n = existing_x_n.unsqueeze(-1)
                          
                      if existing_x_n.shape[1] != x_n_l_i_dev.shape[1]:
                           raise ValueError(f"Feature dimension mismatch in sample loop: "
                                            f"Existing nodes {existing_x_n.shape} vs New nodes {x_n_l_i_dev.shape}")
                      
                      next_x_n = torch.cat([existing_x_n, x_n_l_i_dev], dim=0)
                      next_level = torch.full((num_new_nodes_i, 1), level + 1.0, device=device); next_abs_level = torch.cat([abs_level_list[i_active], next_level])
                      
                      next_x_n_list.append(next_x_n)
                      next_abs_level_list.append(next_abs_level)
                      next_edge_index_list.append(edge_index_list[i_active]) # Carry over old edges
                      if y_list is not None: next_y_list.append(y_list[i_active]) # Carry over label
                      
                      # --- MODIFICATION: Store new node types ---
                      x_n_l_list_for_edge_sampling.append(x_n_l_i_dev) # Store the new types
                      
                      if level >= 0: # Start sampling edges
                          src_candidates_i = torch.arange(1, N_old_i, device=device)
                          dst_nodes_i = torch.arange(N_old_i, N_old_i + num_new_nodes_i, device=device)
                          
                          if src_candidates_i.numel() > 0:
                              query_src_i = src_candidates_i.repeat_interleave(num_new_nodes_i)
                              query_dst_i = dst_nodes_i.repeat(len(src_candidates_i))
                              
                              query_src_list_step.append(query_src_i) # Local indices
                              query_dst_list_step.append(query_dst_i) # Local indices
                              num_new_nodes_list_step.append(num_new_nodes_i)
                              
                              batch_query_src_step.append(query_src_i + node_count_step) # Batch-global indices
                              batch_query_dst_step.append(query_dst_i + node_count_step) # Batch-global indices
                          else:
                              # Graph has no previous nodes to connect from
                              query_src_list_step.append(torch.LongTensor([]).to(device))
                              query_dst_list_step.append(torch.LongTensor([]).to(device))
                              num_new_nodes_list_step.append(num_new_nodes_i)
                          
                      node_count_step += N_old_i + num_new_nodes_i

             # --- Replace the old active lists with the new active lists ---
             edge_index_list = next_edge_index_list
             x_n_list = next_x_n_list
             abs_level_list = next_abs_level_list
             y_list = next_y_list # Update y_list as well
             
             level += 1.0
             
             if len(edge_index_list) == 0: # If new list is empty, all graphs are done
                 break
             
             # --- Sample Edge Layer ---
             if level > 0 and batch_query_src_step: # Only sample if queries were generated
                 # --- FIX: Filter *all* inputs to sample_edge_layer ---
                 # We need to find which graphs *in the new lists* correspond to the queries
                 # This logic is complex. Revert to simpler logic:
                 # We assume batch_query_src_step only contains queries for graphs that
                 # *are* in the next_..._lists.
                 
                 current_batch_size_edge = len(x_n_list) # Use the new batch size
                 
                 active_edge_index_edge = edge_index_list
                 active_x_n_edge = x_n_list
                 active_abs_level_edge = abs_level_list
                 active_y_edge = y_list if y_list is not None else None

                 num_nodes_per_active_graph_edge = [len(x) for x in active_x_n_edge]
                 num_nodes_cumsum_edge = torch.cumsum(torch.tensor([0] + num_nodes_per_active_graph_edge), dim=0).to(device)
                 
                 if not active_x_n_edge: continue
                 
                 feature_dim_edge = active_x_n_edge[0].shape[1] if active_x_n_edge[0].ndim > 1 else 1

                 batch_x_n_edge = torch.cat(active_x_n_edge).to(device)
                 if batch_x_n_edge.ndim == 1: batch_x_n_edge = batch_x_n_edge.unsqueeze(-1)
                 
                 batch_abs_level_edge = torch.cat(active_abs_level_edge).to(device)
                 batch_rel_level_edge = batch_abs_level_edge.max() - batch_abs_level_edge if batch_abs_level_edge.numel() > 0 else batch_abs_level_edge
                 batch_h_y_edge = self.get_batch_y(active_y_edge, active_x_n_edge, device) if active_y_edge is not None else None
                 batch_query_src = torch.cat(batch_query_src_step).to(device)
                 batch_query_dst = torch.cat(batch_query_dst_step).to(device)

                 if batch_query_src.numel() == 0: continue

                 updated_active_edge_indices = self.sample_edge_layer(
                     num_nodes_cumsum_edge, active_edge_index_edge, batch_x_n_edge, batch_abs_level_edge,
                     batch_rel_level_edge, num_new_nodes_list_step, batch_query_src,
                     batch_query_dst, query_src_list_step, query_dst_list_step, batch_h_y_edge,
                     curr_level=level,
                     min_num_steps_e=min_num_steps_e,
                     max_num_steps_e=max_num_steps_e,
                     # --- MODIFICATION: Pass new node types ---
                     x_n_l_list_active=x_n_l_list_for_edge_sampling
                 )
                 
                 # --- Direct Replacement ---
                 edge_index_list = updated_active_edge_indices
                 # --- End Replacement ---


             if self.max_level is not None and level >= self.max_level:
                 break
        
        # Add any graphs remaining in the active lists (if max_level reached)
        for i_active in range(len(x_n_list)): # Use index into the final active lists
             if len(x_n_list[i_active]) > 1: # Check based on final list state
                  edge_index_finished.append(edge_index_list[i_active] - 1); x_n_finished.append(x_n_list[i_active][1:])
                  if y_list is not None: y_finished.append(y_list[i_active])

        if y is None: return edge_index_finished, x_n_finished
        else: final_y = y_finished; return edge_index_finished, x_n_finished, final_y
    # --- End FIX in sample ---
