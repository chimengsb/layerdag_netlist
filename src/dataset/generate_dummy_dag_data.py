import os
import torch
import re
import random
from collections import defaultdict

def count_inputs_in_bench(file_path):
# ... (This function remains the same) ...
    """
    高效地计算一个 .bench 文件中的 INPUT 数量。
    """
    input_count = 0
    # 编译后的正则表达式效率更高
    input_pattern = re.compile(r"^\s*INPUT\(.+\)\s*$")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if input_pattern.match(line):
                    input_count += 1
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return float('inf') # 返回一个极大值以在筛选中排除此文件
    return input_count
def simulate_circuit(input_names, output_names, gate_logic,file_path_for_error = 'Unknown'):
    """
    (已更新 v2) 通过迭代模拟电路所有输入组合来计算其真值表，处理非拓扑排序，确保输出为 0/1。
    """
    num_inputs = len(input_names)
    num_outputs = len(output_names)

    if num_inputs == 0 or num_inputs > 16:
        return torch.empty((0, max(num_outputs, 0)), dtype=torch.int8)

    num_combinations = 2 ** num_inputs
    truth_table = torch.zeros((num_combinations, num_outputs), dtype=torch.int8) # Use int8 for 0/1

    gate_ops = {
        # FIX: Use bitwise XOR for a safer NOT operation
        'NOT': lambda x: 1 ^ x[0],
        'BUF': lambda x: x[0],
        # Ensure AND handles single input gracefully (acts like BUF)
        'AND': lambda x: x[0] & x[1] if len(x) > 1 else x[0],
        'NAND': lambda x: 1 ^ (x[0] & x[1]) if len(x) > 1 else 1 ^ x[0],
        'OR': lambda x: x[0] | x[1] if len(x) > 1 else x[0],
        'NOR': lambda x: 1 ^ (x[0] | x[1]) if len(x) > 1 else 1 ^ x[0],
        'XOR': lambda x: x[0] ^ x[1] if len(x) > 1 else x[0],
        'XNOR': lambda x: 1 ^ (x[0] ^ x[1]) if len(x) > 1 else 1 ^ x[0],
    }

    all_possible_nodes = set(input_names) | {g[0] for g in gate_logic}

    for i in range(num_combinations):
        node_values = {}
        # Set current input combination values (guaranteed 0 or 1)
        for j in range(num_inputs):
            input_val = (i >> j) & 1
            node_values[input_names[j]] = input_val

        # --- Iterative Calculation Logic ---
        gates_to_process = list(gate_logic)
        changed_in_iteration = True
        max_iterations = len(gate_logic) + 1
        iteration_count = 0

        while changed_in_iteration and iteration_count < max_iterations:
            changed_in_iteration = False
            remaining_gates = []
            for out_name, gate_type, in_names in gates_to_process:
                # Check if all inputs are known (present in node_values)
                if all(n in node_values for n in in_names):
                    # Fetch input values (should be 0 or 1 at this point)
                    input_vals = [node_values[n] for n in in_names]
                    
                    calculated_value = 0 # Default to 0
                    if gate_type in gate_ops:
                        try:
                            # Perform the gate operation
                            if gate_type in ['NOT', 'BUF']:
                                if len(input_vals) == 1:
                                    calculated_value = gate_ops[gate_type](input_vals)
                                else:
                                     # print(f"Warning: Gate {gate_type}@{out_name} expects 1 input, got {len(input_vals)}. Using first or default 0.")
                                     calculated_value = gate_ops[gate_type](input_vals[:1]) if input_vals else 0
                            elif len(input_vals) > 0 : # For multi-input gates
                                calculated_value = gate_ops[gate_type](input_vals)
                            # else: input_vals is empty, keep default 0

                            # Ensure the result is strictly 0 or 1
                            node_values[out_name] = int(calculated_value) & 1

                        except Exception as e:
                            print(f"Error calculating gate {gate_type}@{out_name}: {e}. Defaulting output to 0.")
                            node_values[out_name] = 0
                    else:
                        # print(f"Warning: Unknown gate type '{gate_type}'@{out_name}. Defaulting output to 0.")
                        node_values[out_name] = 0
                        
                    changed_in_iteration = True
                else:
                    # Inputs not ready, keep for next iteration
                    remaining_gates.append((out_name, gate_type, in_names))
            
            gates_to_process = remaining_gates
            iteration_count += 1
            
        # Handle gates that could not be processed after iterations
        if gates_to_process:
             unprocessed_gates = [g[0] for g in gates_to_process]
             # print(f"Warning: Could not simulate gates {unprocessed_gates} for input combination {i}. Assigning default 0.")
             for out_name, _, _ in unprocessed_gates:
                 node_values[out_name] = 0 # Default unprocessed gates to 0

        # --- Record Output Values ---
        for j, out_name in enumerate(output_names):
            # Ensure the value assigned is 0 or 1
            final_value = node_values.get(out_name, 0) # Default to 0 if output is somehow undefined
            truth_table[i, j] = int(final_value) & 1
            
            
        if truth_table.numel() > 0: # Only check if the table is not empty
            is_binary = torch.all((truth_table == 0) | (truth_table == 1))
            if not is_binary:
                raise ValueError(f"Error in file '{os.path.basename(file_path_for_error)}': "
                                f"Truth table simulation resulted in non-binary values. "
                                f"Please check the simulation logic or the .bench file structure.")
    return truth_table

def parse_bench_file(file_path, node_type_map):
    """
    (已重构) 使用更健壮的两遍扫描法解析.bench文件。
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    input_names, output_names, gate_logic = set(), set(), []
    all_node_names = set()

    # 正则表达式
    input_pattern = re.compile(r"INPUT\((.+)\)")
    output_pattern = re.compile(r"OUTPUT\((.+)\)")
    gate_pattern = re.compile(r"(.+?)\s*=\s*([A-Z]+)\((.+)\)")

    # --- 第一遍扫描：收集所有唯一的节点名称 ---
    for line in lines:
        match_in = input_pattern.match(line)
        if match_in:
            name = match_in.group(1).strip()
            all_node_names.add(name)
            continue
        
        match_out = output_pattern.match(line)
        if match_out:
            name = match_out.group(1).strip()
            all_node_names.add(name) # 输出本身也是一个节点
            continue

        match_gate = gate_pattern.match(line)
        if match_gate:
            out_name, _, in_str = match_gate.groups()
            out_name = out_name.strip()
            in_names = [n.strip() for n in in_str.split(',')]
            all_node_names.add(out_name)
            for name in in_names:
                all_node_names.add(name)

    # --- 建立从名称到索引的确定性映射 ---
    sorted_node_names = sorted(list(all_node_names))
    name_to_idx = {name: i for i, name in enumerate(sorted_node_names)}
    num_nodes = len(sorted_node_names)
    node_features = [node_type_map['UNKNOWN']] * num_nodes # 先用未知类型填充

    src_nodes, dst_nodes = [], []

    # --- 第二遍扫描：填充特征和边 ---
    gate_logic_temp = []
    for line in lines:
        match_in = input_pattern.match(line)
        if match_in:
            name = match_in.group(1).strip()
            input_names.add(name)
            node_features[name_to_idx[name]] = node_type_map['INPUT']
            continue
        
        match_out = output_pattern.match(line)
        if match_out:
            output_names.add(match_out.group(1).strip())
            continue

        match_gate = gate_pattern.match(line)
        if match_gate:
            out_name, g_type, in_str = match_gate.groups()
            out_name, g_type = out_name.strip(), g_type.strip()
            in_names = [n.strip() for n in in_str.split(',')]
            
            node_features[name_to_idx[out_name]] = node_type_map.get(g_type, node_type_map['UNKNOWN'])
            
            dst_idx = name_to_idx[out_name]
            for in_name in in_names:
                src_nodes.append(name_to_idx[in_name])
                dst_nodes.append(dst_idx)
            gate_logic_temp.append((out_name, g_type, in_names))
    
    # 模拟电路以获取真值表 (y)
    # 确保输入和输出名称的顺序是确定的
    y = simulate_circuit(sorted(list(input_names)), sorted(list(output_names)), gate_logic_temp,file_path)
    # breakpoint()
    return (torch.LongTensor(src_nodes),
            torch.LongTensor(dst_nodes),
            torch.LongTensor(node_features),
            y)

def create_dataset_from_bench_files(bench_root_dir, output_base_path='data_files/circuit_dag_processed', max_inputs=14, train_split=0.8, val_split=0.1):
# ... (This function remains the same) ...
    """
    扫描目录，筛选文件，划分数据集，然后解析并创建数据集文件。
    """
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"创建输出目录: {output_base_path}")

    print(f"正在从 '{bench_root_dir}' 扫描 .bench 文件...")
    all_bench_files = [os.path.join(bench_root_dir, f) for f in os.listdir(bench_root_dir) if f.endswith('.bench')]
    
    print(f"筛选输入数量 <= {max_inputs} 的文件...")
    eligible_files = [path for path in all_bench_files if count_inputs_in_bench(path) <= max_inputs]
    
    print(f"找到 {len(all_bench_files)} 个文件，其中 {len(eligible_files)} 个符合条件。")

    # 随机打乱文件列表以确保随机划分
    random.seed(42) # 添加种子以保证划分可复现
    random.shuffle(eligible_files)

    # 计算划分点
    num_train = int(len(eligible_files) * train_split)
    num_val = int(len(eligible_files) * val_split)

    # 划分文件列表
    train_files = eligible_files[:num_train]
    val_files = eligible_files[num_train : num_train + num_val]
    test_files = eligible_files[num_train + num_val:]
    
    print(f"数据集划分为: {len(train_files)} (训练), {len(val_files)} (验证), {len(test_files)} (测试)")

    file_splits = {'train': train_files, 'val': val_files, 'test': test_files}
    node_type_map = {'INPUT': 0, 'AND': 1, 'NOT': 2, 'UNKNOWN': 9}
    
    for split_name, files in file_splits.items():
        print(f"--- 正在处理 {split_name} 数据集 ---")
        data_dict = defaultdict(list)
        
        for file_path in files:
            try:
                src, dst, x_n, y = parse_bench_file(file_path, node_type_map)
                
                # 添加一个基本检查，确保图不是空的
                if len(x_n) > 0:
                    data_dict['src_list'].append(src)
                    data_dict['dst_list'].append(dst)
                    data_dict['x_n_list'].append(x_n)
                    data_dict['y_list'].append(y)
                else:
                    print(f"警告：解析文件 {os.path.basename(file_path)} 得到一个空图，已跳过。")
            except Exception as e:
                print(f"解析文件 {os.path.basename(file_path)} 时出错: {e}")

        output_file_path = os.path.join(output_base_path, f'{split_name}.pth')
        torch.save(dict(data_dict), output_file_path)
        print(f"已将 {len(data_dict['src_list'])} 个有效电路图保存到 {output_file_path}")

if __name__ == '__main__':
    # --- 用户需要修改的部分 ---
    # 1. 将 'path/to/your/bench/files' 替换为包含所有 .bench 文件的真实目录路径。
    #    这个目录下应该直接存放 .bench 文件，而不是 train/val/test 子目录。
    # bench_files_directory = '/mnt/local_data2/liumiao/code/LSG/LayerDAG/src/dataset/test_data'
    bench_files_directory = '/mnt/local_data2/liumiao/trans2/process_data/rawaig'
    # --------------------------

    # 检查路径是否存在
    if not os.path.isdir(bench_files_directory):
        print("="*50)
        print(f"错误：目录 '{bench_files_directory}' 不存在。")
        print("请在脚本中修改 'bench_files_directory' 变量，")
        print("使其指向您存放 .bench 文件的文件夹。")
        print("为了演示，将创建一个虚拟文件夹 'dummy_bench_files_flat' 并使用它。")
        print("="*50)
        
        # # 为了让脚本可以独立运行，创建一个虚拟的 .bench 文件结构。
        # dummy_data_dir = 'dummy_bench_files_flat'
        # if not os.path.exists(dummy_data_dir):
        #     print(f"创建虚拟 .bench 文件到目录 '{dummy_data_dir}'...")
        #     os.makedirs(dummy_data_dir, exist_ok=True)
        #     base_content = "C = NOT(IN_0)\nD = AND(IN_0, IN_1)\nOUTPUT(C)\nOUTPUT(D)\n"
        #     for i in range(100): # 创建100个虚拟文件
        #         num_inputs = random.randint(2, 14)
        #         inputs_str = "\n".join([f"INPUT(IN_{j})" for j in range(num_inputs)])
        #         with open(os.path.join(dummy_data_dir, f'valid_circuit_{i}.bench'), 'w') as f:
        #             f.write(f"# Circuit with {num_inputs} inputs\n")
        #             f.write(inputs_str + "\n")
        #             f.write(base_content)
        #     print("虚拟文件创建完成。")
        # bench_files_directory = dummy_data_dir

    # 2. 运行主函数，它会自动完成筛选、划分和转换
    create_dataset_from_bench_files(
        bench_root_dir=bench_files_directory,
        output_base_path='/mnt/local_data2/liumiao/code/LSG/LayerDAG/src/dataset/data_files/circuit_dag_processed'
    )

