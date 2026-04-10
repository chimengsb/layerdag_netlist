import torch
import os
import argparse
import collections
from collections import defaultdict
from tqdm import tqdm

def get_reverse_node_type_map():
    """
    定义节点类型整数回字符串的映射。
    !!! 警告: 这个字典必须与您的 `generate_...py` 脚本中的
    node_type_map *完全一致* 才能正确工作。
    """
    # 这是基于 hist_86 的 node_type_map
    node_type_map = {
        'INPUT': 0, 'AND': 1, 'NOT': 2, 'NAND': 3, 'OR': 4,
        'NOR': 5, 'XOR': 6, 'XNOR': 7, 'BUF': 8, 'DFF': 9, 'UNKNOWN': 10
    }
    
    # 反转字典
    int_to_gate = {v: k for k, v in node_type_map.items()}
    return int_to_gate

def convert_graph_to_bench(src, dst, x_n, int_to_gate, file_path):
    """
    将单个图 (src, dst, x_n) 转换为 .bench 文件格式。
    
    Args:
        src (torch.Tensor): 1D 张量，边的源节点索引。
        dst (torch.Tensor): 1D 张量，边的目标节点索引。
        x_n (torch.Tensor): 1D 张量，节点特征 (类型)。
        int_to_gate (dict): 从整数到门名称的映射。
        file_path (str): 输出的 .bench 文件路径。
    """
    num_nodes = len(x_n)
    if num_nodes == 0:
        print(f"Skipping empty graph for {file_path}")
        return

    # --- 1. 构建图结构以进行拓扑排序 ---
    
    # 节点 j 的输入节点列表: inputs_for_node[j] = [in1, in2, ...]
    inputs_for_node = defaultdict(list)
    # 节点 j 的入度: in_degree[j] = count
    in_degree = [0] * num_nodes
    # 节点 j 的出度: out_degree[j] = count
    out_degree = [0] * num_nodes
    
    src_list = src.tolist()
    dst_list = dst.tolist()

    for src_node, dst_node in zip(src_list, dst_list):
        inputs_for_node[dst_node].append(src_node)
        in_degree[dst_node] += 1
        out_degree[src_node] += 1

    # --- 2. 准备拓扑排序 (Kahn's Algorithm) ---
    queue = collections.deque()
    node_name_map = {} # 映射: 节点索引 int -> 导线名称 str
    
    input_lines = []
    gate_lines = []

    # --- 3. 处理所有节点，创建名称，并找到起始节点 (INPUTs) ---
    for j in range(num_nodes):
        # 为每个节点索引分配一个唯一的导线名称
        wire_name = f"n{j}"
        node_name_map[j] = wire_name
        
        # 如果一个节点入度为 0，它是图的起始节点
        if in_degree[j] == 0:
            gate_type = int_to_gate.get(x_n[j].item(), 'UNKNOWN')
            
            if gate_type == 'INPUT':
                input_lines.append(f"INPUT({wire_name})")
                queue.append(j)
            else:
                # 这是一个没有输入的门 (例如常数生成器，如果支持的话)
                # print(f"Warning: Node {j} is a non-INPUT with 0 in-degree. Treating as constant.")
                gate_lines.append(f"{wire_name} = {gate_type}()")
                queue.append(j)

    # --- 4. 执行拓扑排序并生成门逻辑 ---
    
    # 构建出度邻接表，用于遍历
    out_adj = defaultdict(list)
    for src_node, dst_node in zip(src_list, dst_list):
        out_adj[src_node].append(dst_node)

    processed_nodes = 0
    while queue:
        src_node_idx = queue.popleft()
        processed_nodes += 1
        
        for dst_node_idx in out_adj[src_node_idx]:
            in_degree[dst_node_idx] -= 1
            
            # 如果一个节点的所有输入都已被处理，则将该节点加入队列
            if in_degree[dst_node_idx] == 0:
                queue.append(dst_node_idx)
                
                # 为这个节点生成 .bench 门语句
                gate_type = int_to_gate.get(x_n[dst_node_idx].item(), 'UNKNOWN')
                output_wire_name = node_name_map[dst_node_idx]
                
                # 获取其输入节点的名称
                input_node_indices = inputs_for_node[dst_node_idx]
                input_wire_names = [node_name_map[idx] for idx in input_node_indices]
                inputs_str = ", ".join(input_wire_names)
                
                gate_lines.append(f"{output_wire_name} = {gate_type}({inputs_str})")
    
    if processed_nodes != num_nodes:
        print(f"Warning: Graph for {file_path} may contain a cycle! "
              f"Processed {processed_nodes}/{num_nodes} nodes.")

    # --- 5. 推断并写入 OUTPUT 定义 (启发式) ---
    output_lines = []
    for j in range(num_nodes):
        gate_type = int_to_gate.get(x_n[j].item(), 'UNKNOWN')
        # 启发式：如果一个节点不是 INPUT 且没有扇出，它是一个主输出
        if out_degree[j] == 0 and gate_type != 'INPUT':
            output_lines.append(f"OUTPUT({node_name_map[j]})")

    # --- 6. 写入文件 ---
    with open(file_path, 'w') as f:
        f.write(f"# Generated circuit from .pth file (node count: {num_nodes})\n")
        f.write("\n")
        
        # 写入 INPUTs
        for line in input_lines:
            f.write(line + "\n")
        f.write("\n")
        
        # 写入 OUTPUTs
        for line in output_lines:
            f.write(line + "\n")
        f.write("\n")
        
        # 写入逻辑门
        for line in gate_lines:
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Convert generated .pth graph data back to .bench files.")
    parser.add_argument("--pth_file", type=str, required=True, 
                        help="Path to the generated .pth file (e.g., 'circuit_bench_samples/train_generated.pth').")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the generated .bench files.")
    args = parser.parse_args()

    # 1. 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 加载 .pth 文件
    print(f"Loading data from {args.pth_file}...")
    try:
        data_dict = torch.load(args.pth_file)
    except Exception as e:
        print(f"Error loading file {args.pth_file}: {e}")
        return

    # 3. 获取反向映射表
    int_to_gate = get_reverse_node_type_map()
    print("Using node type map:")
    print(int_to_gate)

    # 4. 提取数据列表
    src_list = data_dict.get('src_list')
    dst_list = data_dict.get('dst_list')
    x_n_list = data_dict.get('x_n_list')
    y_list = data_dict.get('y_list') # y (真值表) 在此脚本中不使用，但我们确认它存在

    if not all([src_list, dst_list, x_n_list, y_list]):
        print("Error: The .pth file is missing one or more required keys ('src_list', 'dst_list', 'x_n_list', 'y_list').")
        return

    num_graphs = len(x_n_list)
    print(f"Found {num_graphs} graphs to convert.")

    # 5. 遍历并转换每个图
    for i in tqdm(range(num_graphs), desc="Converting graphs to .bench"):
        src = src_list[i]
        dst = dst_list[i]
        x_n = x_n_list[i]
        
        # 构建输出文件路径
        output_filename = f"generated_circuit_{i}.bench"
        output_filepath = os.path.join(args.output_dir, output_filename)
        
        try:
            convert_graph_to_bench(src, dst, x_n, int_to_gate, output_filepath)
        except Exception as e:
            print(f"Error converting graph {i}: {e}")
            # 继续处理下一个图

    print(f"\nConversion complete. {num_graphs} .bench files saved in '{args.output_dir}'.")

if __name__ == '__main__':
    main()
# ```

# ### 如何使用

# 1.  **保存脚本**: 将上面的代码保存为一个新文件，例如 `convert_pth_to_bench.py`。
# 2.  **检查映射表 (重要!)**: 打开此文件，查看 `get_reverse_node_type_map` 函数。**请务必确认**其内部的 `node_type_map` 字典与您在 `generate_dummy_dag_data.py` 中使用的**完全一致**。
# 3.  **运行转换**:
#     在您的终端中，运行以下命令：
#     ```bash
#     python convert_pth_to_bench.py \
#         --pth_file circuit_bench_samples/train_generated.pth \
#         --output_dir generated_bench_files/
    
