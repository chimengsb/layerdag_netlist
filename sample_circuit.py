import os
import torch

from pprint import pprint
from tqdm import tqdm

# MODIFICATION: Make sure these imports point to your updated files
from setup_utils import set_seed
from src.dataset import load_dataset, DAGDataset # Assuming DAGDataset is in the same place
# MODIFICATION: Import your updated model
from src.model import DiscreteDiffusion, EdgeDiscreteDiffusion, LayerDAG
# MODIFICATION: Removed TPUTileEvaluator as it's specific

def sample_circuit_subset(args, device, dummy_category, model, subset):
    """Generates synthetic graphs based on conditions from the subset."""

    syn_set = DAGDataset(num_categories=dummy_category, label=True)

    raw_y_batch = [] # This will now be a list of truth table tensors

    # Iterate through the labels (truth tables) of the real dataset subset
    # subset.y should already be a list of tensors from circuit_bench_dataset.py
    for i, y_tensor in enumerate(tqdm(subset.y, desc="Sampling Batches")):
        raw_y_batch.append(y_tensor)

        # Process a batch when it's full or when it's the last item
        if (len(raw_y_batch) == args.batch_size) or (i == len(subset.y) - 1):

            # Call the model's sample function.
            # model.sample should internally use y_encoder for the list of tensors.
            # It returns lists of edge_index tensors, node feature tensors, and the original y tensors.
            batch_edge_index, batch_x_n, batch_y_out = model.sample(
                device, len(raw_y_batch), raw_y_batch, # Pass the list of tensors directly
                min_num_steps_n=args.min_num_steps_n,
                max_num_steps_n=args.max_num_steps_n,
                min_num_steps_e=args.min_num_steps_e,
                max_num_steps_e=args.max_num_steps_e)

            # Add each generated graph to the synthetic dataset
            for j in range(len(batch_edge_index)):
                edge_index_j = batch_edge_index[j] # Shape [2, num_edges]
                x_n_j = batch_x_n[j] # Shape [num_nodes] or [num_nodes, features]
                y_j = batch_y_out[j] # Original truth table tensor used as condition

                # Ensure tensors are moved to CPU before saving if they aren't already
                # Convert edge_index from [dst, src] to separate src, dst lists for add_data
                dst_j, src_j = edge_index_j.cpu()
                syn_set.add_data(src_j, dst_j, x_n_j.cpu(), y_j.cpu())

            # Clear the batch
            raw_y_batch = []

    return syn_set

def dump_to_file(syn_set, file_name, sample_dir):
    """Saves the generated synthetic dataset to a .pth file."""
    file_path = os.path.join(sample_dir, file_name)
    data_dict = {
        'src_list': [],
        'dst_list': [],
        'x_n_list': [],
        'y_list': [] # Will store truth table tensors
    }
    print(f"Saving {len(syn_set)} generated graphs to {file_path}...")
    for i in range(len(syn_set)):
        # __getitem__ in DAGDataset returns src, dst, x_n, y
        src_i, dst_i, x_n_i, y_i = syn_set[i]

        data_dict['src_list'].append(src_i)
        data_dict['dst_list'].append(dst_i)
        data_dict['x_n_list'].append(x_n_i)
        data_dict['y_list'].append(y_i)

    torch.save(data_dict, file_path)
    print("Save complete.")

def generate_circuit_samples(args, device, model):
    """Loads dataset, generates samples for train/val splits, and saves them."""
    # MODIFICATION: Define output directory for circuit samples
    sample_dir = 'circuit_bench_samples'
    os.makedirs(sample_dir, exist_ok=True)
    print(f"Generated samples will be saved in '{sample_dir}'")

    # MODIFICATION: Load the circuit_bench dataset
    # This assumes load_dataset is correctly configured in src/dataset/__init__.py
    print("Loading circuit_bench dataset to get conditions (y)...")
    try:
        train_set, val_set, _ = load_dataset('circuit_bench')
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return
    if train_set is None or len(train_set) == 0:
         print("Error: Loaded training set is empty or None.")
         return
    # Also check validation set
    if val_set is None or len(val_set) == 0:
         print("Warning: Loaded validation set is empty or None.")
         # Decide if this is acceptable or should be an error
         # return

    # Extract dummy category info from the loaded dataset
    # Ensure train_set has the dummy_category attribute
    if not hasattr(train_set, 'dummy_category'):
         print("Error: Loaded train_set does not have 'dummy_category' attribute.")
         # Fallback or error handling needed. Using num_categories-1 as a guess.
         if hasattr(train_set, 'num_categories'):
              # Ensure num_categories is valid before using
              num_cats = train_set.num_categories
              if isinstance(num_cats, torch.Tensor): num_cats = num_cats.item()
              if isinstance(num_cats, int) and num_cats > 0:
                   dummy_category = num_cats - 1
                   print(f"Warning: Using fallback dummy_category: {dummy_category}")
              else:
                   print("Error: Cannot determine dummy_category from invalid num_categories. Exiting.")
                   return
         else:
              print("Error: Cannot determine dummy_category. Exiting.")
              return
    else:
        dummy_category = train_set.dummy_category


    print("Generating synthetic samples for the training set conditions...")
    train_syn_set = sample_circuit_subset(args, device, dummy_category, model, train_set)

    # Generate validation samples only if val_set is not empty
    if val_set is not None and len(val_set) > 0:
        print("Generating synthetic samples for the validation set conditions...")
        val_syn_set = sample_circuit_subset(args, device, dummy_category, model, val_set)
    else:
        val_syn_set = None # No validation samples generated

    # MODIFICATION: Removed the specific evaluator.eval call
    # You might add a different evaluation function here later.
    print("Sample generation complete.")

    # Save the generated datasets
    dump_to_file(train_syn_set, 'train_generated.pth', sample_dir)
    if val_syn_set:
        dump_to_file(val_syn_set, 'val_generated.pth', sample_dir)

def main(args):
    torch.set_num_threads(args.num_threads)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"Using device: {device}")

    print(f"Loading model checkpoint from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    ckpt = torch.load(args.model_path, map_location=device) # Load directly to target device

    # MODIFICATION: Check for the correct dataset name
    dataset_name = ckpt.get('dataset', 'unknown') # Use .get for safety
    print(f"Checkpoint trained on dataset: {dataset_name}")
    if dataset_name != 'circuit_bench':
        print(f"Warning: Checkpoint was trained on '{dataset_name}', but we are sampling for 'circuit_bench'. Ensure compatibility.")
        # Decide whether to assert or just warn:
        # assert dataset_name == 'circuit_bench', "Model checkpoint dataset mismatch!"


    # Reconstruct diffusion models
    try:
        node_diffusion_config = ckpt['node_diffusion_config']
        edge_diffusion_config = ckpt['edge_diffusion_config']

        node_diffusion_config['device'] = device
        edge_diffusion_config['device'] = device # Assuming
        node_diffusion = DiscreteDiffusion(**node_diffusion_config)
        edge_diffusion = EdgeDiscreteDiffusion(**edge_diffusion_config)
    except KeyError as e:
        print(f"Error: Missing configuration in checkpoint: {e}")
        return
    except Exception as e:
         print(f"Error initializing diffusion models: {e}")
         # Print traceback for more details on initialization error
         import traceback
         traceback.print_exc()
         return

    try:
        model = LayerDAG(device=device,
                           node_diffusion=node_diffusion,
                           edge_diffusion=edge_diffusion,
                           **ckpt['model_config'])
        print("Model configuration:")
        pprint(ckpt['model_config'])
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval() # Set model to evaluation mode
        print("Model loaded successfully.")
    except KeyError as e:
         print(f"Error: Missing configuration or state_dict in checkpoint: {e}")
         return
    except Exception as e:
        print(f"Error loading model state: {e}")
        import traceback
        traceback.print_exc()
        return

    set_seed(args.seed)

    # MODIFICATION: Call the renamed generation function
    generate_circuit_samples(args, device, model)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate synthetic circuit graphs using a trained LayerDAG model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for conditional generation.") # Adjusted default
    parser.add_argument("--num_threads", type=int, default=8, help="Number of CPU threads.") # Adjusted default
    parser.add_argument("--min_num_steps_n", type=int, default=None, help="Min diffusion steps for nodes.")
    parser.add_argument("--min_num_steps_e", type=int, default=None, help="Min diffusion steps for edges.")
    parser.add_argument("--max_num_steps_n", type=int, default=None, help="Max diffusion steps for nodes.")
    parser.add_argument("--max_num_steps_e", type=int, default=None, help="Max diffusion steps for edges.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    main(args)

