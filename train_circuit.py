import dgl.sparse as dglsp
import pandas as pd
import time
import torch
import torch.nn as nn
import wandb

from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm

from setup_utils import set_seed, load_yaml
# ... (imports remain the same) ...
from src.dataset import (load_dataset, LayerDAGNodeCountDataset,
                         LayerDAGNodePredDataset, LayerDAGEdgePredDataset, collate_node_count,
                         collate_node_pred, collate_edge_pred)
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG

# ... (eval_node_count 保持不变) ...
@torch.no_grad()
def eval_node_count(device, val_loader, model, main_model):
# ... (代码保持不变) ...
    model.eval()
    total_nll, total_count, true_count = 0, 0, 0
    for batch_data in tqdm(val_loader, desc="Evaluating Node Count"):
        if batch_data is None: continue
        if len(batch_data) == 9:
            batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_y, batch_n2g_index, batch_label, batch_x_n_list = batch_data
        elif len(batch_data) == 7:
            batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_n2g_index, batch_label = batch_data
            batch_y, batch_x_n_list = None, None
        else: continue
        num_nodes = len(batch_x_n_collated);
        if num_nodes == 0: continue
        batch_A = dglsp.spmatrix(batch_edge_index, shape=(num_nodes, num_nodes)).to(device); batch_x_n = batch_x_n_collated.to(device); batch_abs_level = batch_abs_level.to(device); batch_rel_level = batch_rel_level.to(device); batch_A_n2g = dglsp.spmatrix(batch_n2g_index, shape=(batch_size, num_nodes)).to(device); batch_label = batch_label.to(device)
        batch_h_y = None
        if batch_y is not None and batch_x_n_list is not None: batch_h_y = main_model.get_batch_y(batch_y, batch_x_n_list, device)
        batch_logits = model(batch_A, batch_x_n, batch_abs_level, batch_rel_level, batch_A_n2g, h_y=batch_h_y)
        batch_nll = -batch_logits.log_softmax(dim=-1)
        batch_label = batch_label.clamp(max=batch_logits.shape[-1] - 1); current_batch_size = batch_logits.shape[0]
        if current_batch_size != batch_label.shape[0]: continue
        batch_nll = batch_nll[torch.arange(current_batch_size, device=device), batch_label]; total_nll += batch_nll.sum().item()
        batch_probs = batch_logits.softmax(dim=-1); batch_preds = batch_probs.multinomial(1).squeeze(-1); true_count += (batch_preds == batch_label).sum().item(); total_count += current_batch_size
    return (total_nll / total_count if total_count > 0 else 0), (true_count / total_count if total_count > 0 else 0)

def main_node_count(device, train_set, val_set, model, main_model, config, patience):
    loader_cfg = config['loader']; num_workers = loader_cfg.get('num_workers', 0); context = 'spawn' if device.type == 'cuda' and num_workers > 0 else None
    train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_node_count, **loader_cfg, drop_last=True, multiprocessing_context=context)
    val_loader = DataLoader(val_set, shuffle=False, collate_fn=collate_node_count, **loader_cfg, multiprocessing_context=context)
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    best_val_nll, best_val_acc, best_state_dict, num_patient_epochs = float('inf'), 0, deepcopy(model.state_dict()), 0
    for epoch in range(config['num_epochs']):
        model.train()
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1} Node Count"):
            if batch_data is None: continue
            if len(batch_data) == 9:
                batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_y, batch_n2g_index, batch_label, batch_x_n_list = batch_data
            elif len(batch_data) == 7:
                batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_n2g_index, batch_label = batch_data
                batch_y, batch_x_n_list = None, None
            else: continue
            num_nodes = len(batch_x_n_collated);
            if num_nodes == 0: continue
            batch_A = dglsp.spmatrix(batch_edge_index, shape=(num_nodes, num_nodes)).to(device); batch_x_n = batch_x_n_collated.to(device); batch_abs_level = batch_abs_level.to(device); batch_rel_level = batch_rel_level.to(device); batch_A_n2g = dglsp.spmatrix(batch_n2g_index, shape=(batch_size, num_nodes)).to(device); batch_label = batch_label.to(device)
            batch_h_y = None
            if batch_y is not None and batch_x_n_list is not None: batch_h_y = main_model.get_batch_y(batch_y, batch_x_n_list, device)
            batch_pred = model(batch_A, batch_x_n, batch_abs_level, batch_rel_level, batch_A_n2g, h_y=batch_h_y)
            if batch_pred.shape[0] != batch_label.shape[0]: continue
            batch_label = batch_label.clamp(max=batch_pred.shape[-1] - 1); loss = criterion(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            # --- FIX: Add Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 1.0 是一个常用的阈值
            # --- End FIX ---
            optimizer.step()
            wandb.log({'node_count/loss': loss.item()})
        val_nll, val_acc = eval_node_count(device, val_loader, model, main_model)
        if val_nll < best_val_nll: best_val_nll = val_nll
        if val_acc > best_val_acc: best_val_acc = val_acc; best_state_dict = deepcopy(model.state_dict()); num_patient_epochs = 0
        else: num_patient_epochs += 1
        wandb.log({'node_count/epoch': epoch, 'node_count/val_nll': val_nll, 'node_count/best_val_nll': best_val_nll, 'node_count/val_acc': val_acc, 'node_count/best_val_acc': best_val_acc, 'node_count/num_patient_epochs': num_patient_epochs})
        if (patience is not None) and (num_patient_epochs == patience): break
    return best_state_dict

# ... (eval_node_pred 保持不变) ...
@torch.no_grad()
def eval_node_pred(device, val_loader, model, main_model):
# ... (代码保持不变) ...
    model.eval()
    total_nll, total_count = 0, 0
    for batch_data in tqdm(val_loader, desc="Evaluating Node Pred"):
        if batch_data is None: continue
        if len(batch_data) == 13: # Conditional
            batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_n2g_index, batch_z_t, batch_t, batch_y, query2g, num_query_cumsum, batch_z, batch_x_n_list = batch_data
        elif len(batch_data) == 11: # Unconditional
            batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_n2g_index, batch_z_t, batch_t, query2g, num_query_cumsum, batch_z = batch_data
            batch_y, batch_x_n_list = None, None
        else: continue
        num_nodes = len(batch_x_n_collated);
        if num_nodes == 0 or batch_z_t.numel() == 0: continue
        batch_A = dglsp.spmatrix(batch_edge_index, shape=(num_nodes, num_nodes)).to(device); batch_x_n = batch_x_n_collated.to(device); batch_abs_level = batch_abs_level.to(device); batch_rel_level = batch_rel_level.to(device); batch_A_n2g = dglsp.spmatrix(batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
        batch_z_t = batch_z_t.to(device); batch_t = batch_t.to(device); query2g = query2g.to(device); num_query_cumsum = num_query_cumsum.to(device); batch_z = batch_z.to(device)
        batch_h_y = None
        if batch_y is not None and batch_x_n_list is not None: batch_h_y = main_model.get_batch_y(batch_y, batch_x_n_list, device)
        batch_logits = model(batch_A, batch_x_n, batch_abs_level, batch_rel_level, batch_A_n2g, batch_z_t, batch_t, query2g, num_query_cumsum, h_y=batch_h_y)
        if not batch_logits or not batch_logits[0].numel(): continue
        D = len(batch_logits); batch_num_queries = batch_logits[0].shape[0]
        if batch_z.shape[0] != batch_num_queries or batch_z.shape[1] < D: continue
        for d in range(D):
            batch_logits_d = batch_logits[d]; batch_nll_d = -batch_logits_d.log_softmax(dim=-1)
            target_indices = batch_z[:, d].clamp(0, batch_nll_d.shape[-1] - 1)
            batch_nll_d = batch_nll_d[torch.arange(batch_num_queries, device=device), target_indices]
            total_nll += batch_nll_d.sum().item()
        total_count += batch_num_queries * D
    return total_nll / total_count if total_count > 0 else 0

def main_node_pred(device, train_set, val_set, model, main_model, config, patience):
    loader_cfg = config['loader']; num_workers = loader_cfg.get('num_workers', 0); context = 'spawn' if device.type == 'cuda' and num_workers > 0 else None
    train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_node_pred, **loader_cfg, multiprocessing_context=context)
    val_loader = DataLoader(val_set, collate_fn=collate_node_pred, **loader_cfg, multiprocessing_context=context)
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    best_val_nll, best_state_dict, num_patient_epochs = float('inf'), deepcopy(model.state_dict()), 0
    for epoch in range(config['num_epochs']):
        val_nll = eval_node_pred(device, val_loader, model, main_model)
        if val_nll < best_val_nll: best_val_nll = val_nll; best_state_dict = deepcopy(model.state_dict()); num_patient_epochs = 0
        else: num_patient_epochs += 1
        wandb.log({'node_pred/epoch': epoch, 'node_pred/val_nll': val_nll, 'node_pred/best_val_nll': best_val_nll, 'node_pred/num_patient_epochs': num_patient_epochs})
        if (patience is not None) and (num_patient_epochs == patience): break
        model.train()
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1} Node Pred"):
            if batch_data is None: continue
            if len(batch_data) == 13: # Conditional
                batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_n2g_index, batch_z_t, batch_t, batch_y, query2g, num_query_cumsum, batch_z, batch_x_n_list = batch_data
            elif len(batch_data) == 11: # Unconditional
                batch_size, batch_edge_index, batch_x_n_collated, batch_abs_level, batch_rel_level, batch_n2g_index, batch_z_t, batch_t, query2g, num_query_cumsum, batch_z = batch_data
                batch_y, batch_x_n_list = None, None
            else: continue
            num_nodes = len(batch_x_n_collated);
            if num_nodes == 0 or batch_z_t.numel() == 0: continue
            batch_A = dglsp.spmatrix(batch_edge_index, shape=(num_nodes, num_nodes)).to(device); batch_x_n = batch_x_n_collated.to(device); batch_abs_level = batch_abs_level.to(device); batch_rel_level = batch_rel_level.to(device); batch_A_n2g = dglsp.spmatrix(batch_n2g_index, shape=(batch_size, num_nodes)).to(device)
            batch_z_t = batch_z_t.to(device); batch_t = batch_t.to(device); query2g = query2g.to(device); num_query_cumsum = num_query_cumsum.to(device); batch_z = batch_z.to(device)
            batch_h_y = None
            if batch_y is not None and batch_x_n_list is not None: batch_h_y = main_model.get_batch_y(batch_y, batch_x_n_list, device)
            batch_pred = model(batch_A, batch_x_n, batch_abs_level, batch_rel_level, batch_A_n2g, batch_z_t, batch_t, query2g, num_query_cumsum, h_y=batch_h_y)
            if not batch_pred or not batch_pred[0].numel(): continue
            loss = 0; D = len(batch_pred)
            if batch_z.shape[0] != batch_pred[0].shape[0] or batch_z.shape[1] < D: continue
            for d in range(D): target_indices = batch_z[:, d].clamp(0, batch_pred[d].shape[-1] - 1); loss = loss + criterion(batch_pred[d], target_indices)
            if D > 0: loss /= D
            optimizer.zero_grad()
            loss.backward()
            # --- FIX: Add Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # --- End FIX ---
            optimizer.step()
            wandb.log({'node_pred/loss': loss.item()})
    return best_state_dict

# ... (eval_edge_pred 评估函数) ...
@torch.no_grad()
def eval_edge_pred(device, val_loader, model, main_model):
# ... (代码保持不变) ...
    model.eval()
    total_nll, total_count = 0, 0
    for batch_data in tqdm(val_loader, desc="Evaluating Edge Pred"):
        if batch_data is None: continue
        
        # --- [MODIFIED] 更新解包逻辑以匹配 collate_edge_pred ---
        if len(batch_data) == 11: # Conditional (10 + 1)
            batch_edge_index, batch_noisy_edge_index, batch_x_n_list, batch_abs_level, batch_rel_level, batch_t, batch_y, batch_query_src, batch_query_dst, batch_label, batch_query_dst_x_n = batch_data
        elif len(batch_data) == 10: # Unconditional (9 + 1)
             batch_edge_index, batch_noisy_edge_index, batch_x_n_list, batch_abs_level, batch_rel_level, batch_t, batch_query_src, batch_query_dst, batch_label, batch_query_dst_x_n = batch_data
             batch_y = None
        else: continue
        # --- [END MODIFIED] ---

        if not batch_x_n_list: continue
        batch_x_n_collated = torch.cat(batch_x_n_list, dim=0).long(); num_nodes = len(batch_x_n_collated)
        if num_nodes == 0 or batch_query_src.numel() == 0: continue
        batch_size = len(batch_x_n_list)
        batch_A = dglsp.spmatrix(torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1), shape=(num_nodes, num_nodes)).to(device); batch_x_n = batch_x_n_collated.to(device); batch_abs_level = batch_abs_level.to(device); batch_rel_level = batch_rel_level.to(device); batch_t = batch_t.to(device)
        batch_query_src = batch_query_src.to(device); batch_query_dst = batch_query_dst.to(device); batch_label = batch_label.to(device)
        batch_h_y = None
        if batch_y is not None: batch_h_y = main_model.get_batch_y(batch_y, batch_x_n_list, device)
        batch_logits = model(batch_A, batch_x_n, batch_abs_level, batch_rel_level, batch_t, batch_query_src, batch_query_dst, h_y=batch_h_y)
        if batch_logits.numel() == 0: continue
        batch_nll = -batch_logits.log_softmax(dim=-1); batch_num_queries = batch_logits.shape[0]
        batch_label = batch_label.clamp(0, batch_nll.shape[-1] - 1)
        batch_nll = batch_nll[torch.arange(batch_num_queries, device=device), batch_label]
        total_nll += batch_nll.sum().item(); total_count += batch_num_queries
    return total_nll / total_count if total_count > 0 else 0

def main_edge_pred(device, train_set, val_set, model, main_model, config, patience):
    loader_cfg = config['loader']; num_workers = loader_cfg.get('num_workers', 0); context = 'spawn' if device.type == 'cuda' and num_workers > 0 else None
    train_loader = DataLoader(train_set, shuffle=True, collate_fn=collate_edge_pred, **loader_cfg, multiprocessing_context=context)
    val_loader = DataLoader(val_set, collate_fn=collate_edge_pred, **loader_cfg, multiprocessing_context=context)
    
    # --- [MODIFIED] 将 reduction 设为 'none' 以便进行加权 ---
    criterion = nn.CrossEntropyLoss(reduction='none') 
    
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    best_val_nll, best_state_dict, num_patient_epochs = float('inf'), deepcopy(model.state_dict()), 0
    for epoch in range(config['num_epochs']):
# ... (rest of main_edge_pred loop, add gradient clipping) ...
        val_nll = eval_edge_pred(device, val_loader, model, main_model)
        if val_nll < best_val_nll: best_val_nll = val_nll; best_state_dict = deepcopy(model.state_dict()); num_patient_epochs = 0
        else: num_patient_epochs += 1
        wandb.log({'edge_pred/epoch': epoch, 'edge_pred/val_nll': val_nll, 'edge_pred/best_val_nll': best_val_nll, 'edge_pred/num_patient_epochs': num_patient_epochs})
        if (patience is not None) and (num_patient_epochs == patience): break
        model.train()
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1} Edge Pred"):
            if batch_data is None: continue
            
            # --- [MODIFIED] 更新解包逻辑以接收 batch_query_dst_x_n ---
            if len(batch_data) == 11: # Conditional (10 + 1)
                batch_edge_index, batch_noisy_edge_index, batch_x_n_list, batch_abs_level, batch_rel_level, batch_t, batch_y, batch_query_src, batch_query_dst, batch_label, batch_query_dst_x_n = batch_data
            elif len(batch_data) == 10: # Unconditional (9 + 1)
                 batch_edge_index, batch_noisy_edge_index, batch_x_n_list, batch_abs_level, batch_rel_level, batch_t, batch_query_src, batch_query_dst, batch_label, batch_query_dst_x_n = batch_data
                 batch_y = None
            else: continue
            # --- [END MODIFIED] ---

            if not batch_x_n_list: continue
            batch_x_n_collated = torch.cat(batch_x_n_list, dim=0).long(); num_nodes = len(batch_x_n_collated)
            if num_nodes == 0 or batch_query_src.numel() == 0: continue
            batch_size = len(batch_x_n_list)
            
            batch_A = dglsp.spmatrix(torch.cat([batch_edge_index, batch_noisy_edge_index], dim=1), shape=(num_nodes, num_nodes)).to(device); batch_x_n = batch_x_n_collated.to(device); batch_abs_level = batch_abs_level.to(device); batch_rel_level = batch_rel_level.to(device); batch_t = batch_t.to(device)
            batch_query_src = batch_query_src.to(device); batch_query_dst = batch_query_dst.to(device); batch_label = batch_label.to(device)
            
            # --- [ADDED] 将 batch_query_dst_x_n 移至 device ---
            batch_query_dst_x_n = batch_query_dst_x_n.to(device)
            
            batch_h_y = None
            if batch_y is not None: batch_h_y = main_model.get_batch_y(batch_y, batch_x_n_list, device)
            batch_pred = model(batch_A, batch_x_n, batch_abs_level, batch_rel_level, batch_t, batch_query_src, batch_query_dst, h_y=batch_h_y)
            if batch_pred.numel() == 0: continue
            
            batch_label = batch_label.clamp(0, batch_pred.shape[-1] - 1)
            
            # --- [START] 软约束 - 基于门类型的加权损失 ---
            
            # 1. 计算每个查询的 NLL 损失 (reduction='none')
            per_query_loss = criterion(batch_pred, batch_label)
            
            # 2. 计算权重
            if batch_query_dst_x_n.numel() > 0 and batch_query_dst_x_n.shape[0] == per_query_loss.shape[0]:
                # 假设门类型是第一个特征 (索引 0)
                if batch_query_dst_x_n.ndim > 1:
                    gate_types = batch_query_dst_x_n[:, 0]
                else:
                    gate_types = batch_query_dst_x_n
                
                weights = torch.ones_like(gate_types, dtype=torch.float, device=device)
                
                # 假设类型来自 bench 文件：1=AND, 2=NOT, 8=BUF
                AND_TYPE = 1
                NOT_TYPE = 2
                BUF_TYPE = 8
                WEIGHT_FACTOR = 2.0 # 软约束权重因子

                # 找到所有 AND, NOT, BUF 门
                is_and = (gate_types == AND_TYPE)
                is_not = (gate_types == NOT_TYPE)
                is_buf = (gate_types == BUF_TYPE)
                
                weights[is_and | is_not | is_buf] = WEIGHT_FACTOR
                
                # 3. 计算加权平均损失
                # 确保权重总和不为零，避免除零
                if weights.sum() > 0:
                    loss = (per_query_loss * weights).sum() / weights.sum()
                else:
                    loss = per_query_loss.mean() # 回退
            else:
                # 回退到标准损失 (如果门类型信息丢失或维度不匹配)
                loss = per_query_loss.mean()
            # --- [END] 软约束 ---

            optimizer.zero_grad()
            loss.backward()
            # --- FIX: Add Gradient Clipping ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # --- End FIX ---
            optimizer.step()
            wandb.log({'edge_pred/loss': loss.item()})
    return best_state_dict

# ... (main function 保持不变) ...
def main(args):
    # torch.autograd.set_detect_anomaly(True) # Keep commented out unless debugging
    torch.set_num_threads(args.num_threads)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"; device = torch.device(device_str)
    set_seed(args.seed); config = load_yaml(args.config_file); dataset_name = config['general']['dataset']
    config_df = pd.json_normalize(config, sep='/'); ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    wandb.init(project=f'LayerDAG_{dataset_name}', name=f'{ts}', config=config_df.to_dict(orient='records')[0])
    try: train_set, val_set, _ = load_dataset(dataset_name)
    except ValueError as e: print(f"Error loading dataset: {e}"); return
    if train_set is None or val_set is None or len(train_set) == 0: print(f"Error: Dataset '{dataset_name}' is empty or failed to load."); return
    try:
        train_node_count_dataset = LayerDAGNodeCountDataset(train_set, config['general']['conditional'])
        val_node_count_dataset = LayerDAGNodeCountDataset(val_set, config['general']['conditional'])
        if len(train_node_count_dataset) == 0: print("Error: train_node_count_dataset empty after init."); return
        train_node_pred_dataset = LayerDAGNodePredDataset(train_set, config['general']['conditional'])
        val_node_pred_dataset = LayerDAGNodePredDataset(val_set, config['general']['conditional'], get_marginal=False)
        if not hasattr(train_node_pred_dataset, 'x_n_marginal') or not train_node_pred_dataset.x_n_marginal:
             print("Error: x_n_marginal not calculated."); num_cats = train_set.num_categories - 1
             default_marginal = [(torch.ones(nc)/nc) for nc in (num_cats.tolist() if isinstance(num_cats, torch.Tensor) else [num_cats])] if isinstance(num_cats, (torch.Tensor, list)) else [torch.ones(num_cats) / num_cats]
             print(f"Warning: Using default marginal: {default_marginal}"); train_node_pred_dataset.x_n_marginal = default_marginal
        node_diffusion_config = {'marginal_list': train_node_pred_dataset.x_n_marginal, 'T': config['node_pred']['T']}
        node_diffusion_config['device'] = device
        node_diffusion = DiscreteDiffusion(**node_diffusion_config); train_node_pred_dataset.node_diffusion = node_diffusion; val_node_pred_dataset.node_diffusion = node_diffusion
        train_edge_pred_dataset = LayerDAGEdgePredDataset(train_set, config['general']['conditional'])
        val_edge_pred_dataset = LayerDAGEdgePredDataset(val_set, config['general']['conditional'])
        edge_diffusion_config = {'avg_in_deg': train_edge_pred_dataset.avg_in_deg, 'T': config['edge_pred']['T']}
        edge_diffusion_config['device'] = device
        edge_diffusion = EdgeDiscreteDiffusion(**edge_diffusion_config); train_edge_pred_dataset.edge_diffusion = edge_diffusion; val_edge_pred_dataset.edge_diffusion = edge_diffusion
    except Exception as e: print(f"Error during Dataset initialization: {e}"); return
    max_level_train = train_node_pred_dataset.input_level.max().item() if hasattr(train_node_pred_dataset, 'input_level') and train_node_pred_dataset.input_level.numel() > 0 else 0
    max_level_val = val_node_pred_dataset.input_level.max().item() if hasattr(val_node_pred_dataset, 'input_level') and val_node_pred_dataset.input_level.numel() > 0 else 0
    model_config = { 'num_x_n_cat': train_set.num_categories, 'node_count_encoder_config': config['node_count']['model'], 'max_layer_size': train_node_count_dataset.max_layer_size, 'node_pred_graph_encoder_config': config['node_pred']['graph_encoder'], 'node_predictor_config': config['node_pred']['predictor'], 'edge_pred_graph_encoder_config': config['edge_pred']['graph_encoder'], 'edge_predictor_config': config['edge_pred']['predictor'], 'max_level': max(max_level_train, max_level_val)}
    if not isinstance(model_config['num_x_n_cat'], (int, torch.Tensor, list)) or (isinstance(model_config['num_x_n_cat'], int) and model_config['num_x_n_cat'] <= 0): print(f"Error: Invalid num_x_n_cat: {model_config['num_x_n_cat']}"); return
    model = LayerDAG(device=device, node_diffusion=node_diffusion, edge_diffusion=edge_diffusion, **model_config)
    print("Starting Node Count Training..."); node_count_state_dict = main_node_count(device, train_node_count_dataset, val_node_count_dataset, model.node_count_model, model, config['node_count'], config['general']['patience']); model.node_count_model.load_state_dict(node_count_state_dict); print("Node Count Training Complete.")
    print("Starting Node Prediction Training..."); node_pred_state_dict = main_node_pred(device, train_node_pred_dataset, val_node_pred_dataset, model.node_pred_model, model, config['node_pred'], config['general']['patience']); model.node_pred_model.load_state_dict(node_pred_state_dict); print("Node Prediction Training Complete.")
    print("Starting Edge Prediction Training..."); edge_pred_state_dict = main_edge_pred(device, train_edge_pred_dataset, val_edge_pred_dataset, model.edge_pred_model, model, config['edge_pred'], config['general']['patience']); model.edge_pred_model.load_state_dict(edge_pred_state_dict); print("Edge Prediction Training Complete.")
    save_path = f'model_{dataset_name}_{ts}.pth'; print(f"Saving final model to {save_path}")
    torch.save({'dataset': dataset_name, 'node_diffusion_config': node_diffusion_config, 'edge_diffusion_config': edge_diffusion_config, 'model_config': model_config, 'model_state_dict': model.state_dict()}, save_path); print("Model saved.")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(); parser.add_argument("--config_file", type=str, required=True); parser.add_argument("--num_threads", type=int, default=16); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); main(args)
