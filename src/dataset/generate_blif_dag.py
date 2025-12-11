import os
import torch
import re
import random
from collections import defaultdict
import functools

# --- 配置部分 ---
# 将 BLIF 文件中的门名称映射到标准逻辑类型
# 根据 mece_m.lib 和常见多输入门进行了扩充
GATE_MAPPING = {
    # Inverters (反相器) - 覆盖 mece_m.lib 中的 inv1-inv4
    'inv': 'NOT', 'inv1': 'NOT', 'inv1x': 'NOT', 'not': 'NOT',
    'inv2': 'NOT', 'inv3': 'NOT', 'inv4': 'NOT',
    
    # Buffers (缓冲器)
    'buf': 'BUF', 'buff': 'BUF', 'buffer': 'BUF',
    
    # AND Gates (与门) - 支持多输入命名
    'and2': 'AND', 'and3': 'AND', 'and4': 'AND', 'and': 'AND',
    
    # NAND Gates (与非门)
    'nand2': 'NAND', 'nand3': 'NAND', 'nand4': 'NAND', 'nand': 'NAND',
    
    # OR Gates (或门)
    'or2': 'OR', 'or3': 'OR', 'or4': 'OR', 'or': 'OR',
    
    # NOR Gates (或非门)
    'nor2': 'NOR', 'nor3': 'NOR', 'nor4': 'NOR', 'nor': 'NOR',
    
    # XOR Gates (异或门) - 覆盖 mece_m.lib 中的 xor2a, xor2b
    'xor2': 'XOR', 'xor': 'XOR',
    'xor2a': 'XOR', 'xor2b': 'XOR',
    
    # XNOR Gates (同或门) - 覆盖 mece_m.lib 中的 xnor2a, xnor2b
    'xnor2': 'XNOR', 'xnor': 'XNOR',
    'xnor2a': 'XNOR', 'xnor2b': 'XNOR',
    
    # Constants (常量)
    'one': 'ONE', 'zero': 'ZERO',
    'const0': 'ZERO', 'const1': 'ONE'
}

# 扩展节点特征映射，包含 BLIF 中可能出现的所有逻辑
NODE_TYPE_MAP = {
    'INPUT': 0, 
    'AND': 1, 
    'NOT': 2, 
    'NAND': 3, 
    'OR': 4, 
    'NOR': 5, 
    'XOR': 6, 
    'XNOR': 7, 
    'BUF': 8,
    'UNKNOWN': 9,
    'CONST': 10 # 用于常量 0/1
}

def read_blif_lines(file_path):
    """
    读取 BLIF 文件并处理反斜杠 (\) 续行符，将多行命令合并为一行。
    """
    processed_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_line = ""
        for line in lines:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 处理续行符
            if line.endswith('\\'):
                current_line += line[:-1] + " "
            else:
                current_line += line
                processed_lines.append(current_line)
                current_line = ""
    except Exception as e:
        print(f"读取文件错误 {file_path}: {e}")
        return []
    return processed_lines

def count_inputs_in_blif(file_path):
    """
    计算 BLIF 文件中的输入数量。
    """
    lines = read_blif_lines(file_path)
    input_count = 0
    for line in lines:
        if line.startswith(".inputs"):
            # .inputs in1 in2 ...
            parts = line.split()
            input_count += len(parts) - 1 # 减去 '.inputs' 本身
    return input_count

def simulate_circuit(input_names, output_names, gate_logic, file_path_for_error='Unknown'):
    """
    模拟电路生成真值表。
    使用 functools.reduce 支持任意数量的输入（多输入门）。
    """
    num_inputs = len(input_names)
    num_outputs = len(output_names)

    if num_inputs == 0 or num_inputs > 16:
        return torch.empty((0, max(num_outputs, 0)), dtype=torch.int8)

    num_combinations = 2 ** num_inputs
    truth_table = torch.zeros((num_combinations, num_outputs), dtype=torch.int8)

    # --- 逻辑运算定义 (支持多输入) ---
    def bitwise_and(x): return functools.reduce(lambda a, b: a & b, x) if x else 0
    def bitwise_or(x): return functools.reduce(lambda a, b: a | b, x) if x else 0
    def bitwise_xor(x): return functools.reduce(lambda a, b: a ^ b, x) if x else 0
    
    # 辅助函数：确保非门和缓冲器只接受单输入，多输入则取第一个
    def safe_unary(x, op):
        if not x: return 0
        return op(x[0])

    gate_ops = {
        'NOT': lambda x: safe_unary(x, lambda v: 1 ^ v),
        'BUF': lambda x: safe_unary(x, lambda v: v),
        'AND': lambda x: bitwise_and(x),
        'NAND': lambda x: 1 ^ bitwise_and(x),
        'OR': lambda x: bitwise_or(x),
        'NOR': lambda x: 1 ^ bitwise_or(x),
        'XOR': lambda x: bitwise_xor(x),
        'XNOR': lambda x: 1 ^ bitwise_xor(x),
        'ONE': lambda x: 1,
        'ZERO': lambda x: 0
    }

    for i in range(num_combinations):
        node_values = {}
        # 设置输入值
        for j in range(num_inputs):
            input_val = (i >> j) & 1
            node_values[input_names[j]] = input_val

        # 迭代计算
        gates_to_process = list(gate_logic)
        changed_in_iteration = True
        max_iterations = len(gate_logic) + 2
        iteration_count = 0

        while changed_in_iteration and iteration_count < max_iterations:
            changed_in_iteration = False
            remaining_gates = []
            
            for out_name, gate_type, in_names in gates_to_process:
                # 检查是否所有输入都在 node_values 中
                if all(n in node_values for n in in_names):
                    input_vals = [node_values[n] for n in in_names]
                    
                    calculated_value = 0
                    if gate_type in gate_ops:
                        try:
                            calculated_value = gate_ops[gate_type](input_vals)
                            node_values[out_name] = int(calculated_value) & 1
                        except Exception as e:
                            # 仅在调试时打开，防止刷屏
                            # print(f"Sim Error {gate_type}@{out_name}: {e}")
                            node_values[out_name] = 0
                    else:
                        node_values[out_name] = 0
                        
                    changed_in_iteration = True
                else:
                    remaining_gates.append((out_name, gate_type, in_names))
            
            gates_to_process = remaining_gates
            iteration_count += 1
        
        # 对于死循环或无法计算的门，默认给 0，防止崩溃
        for out_name, _, _ in gates_to_process:
            node_values[out_name] = 0

        # 记录输出
        for j, out_name in enumerate(output_names):
            final_value = node_values.get(out_name, 0)
            truth_table[i, j] = int(final_value) & 1

    return truth_table

def parse_blif_file(file_path, node_type_map):
    """
    解析 .blif 文件，构建 DAG 和节点特征。
    """
    lines = read_blif_lines(file_path)

    input_names = [] 
    output_names = []
    gate_logic = [] # (out_name, gate_type, [in_names])
    all_node_names = set()

    for line in lines:
        parts = line.split()
        if not parts: continue
        
        cmd = parts[0]

        if cmd == '.model':
            continue
        elif cmd == '.inputs':
            inputs = parts[1:]
            input_names.extend(inputs)
            for n in inputs: all_node_names.add(n)
        elif cmd == '.outputs':
            outputs = parts[1:]
            output_names.extend(outputs)
            for n in outputs: all_node_names.add(n)
        elif cmd == '.gate':
            # 格式: .gate type pin1=net1 pin2=net2 ...
            if len(parts) < 2: continue
            
            raw_gate_type = parts[1]
            params = parts[2:]
            
            # 映射门类型，如果找不到则为 UNKNOWN
            standard_type = GATE_MAPPING.get(raw_gate_type, 'UNKNOWN')
            
            # 解析引脚连接
            in_nets = []
            out_net = None
            
            for param in params:
                if '=' not in param: 
                    continue
                pin, net = param.split('=')
                
                # 识别输出引脚
                # 在 mece_m.lib 中，输出定义为 O=...
                # 一般 blif 中输出引脚可能是 O, Y, out, output
                if pin in ['O', 'Y', 'out', 'output']:
                    out_net = net
                else:
                    in_nets.append(net)
            
            if out_net:
                gate_logic.append((out_net, standard_type, in_nets))
                all_node_names.add(out_net)
                for n in in_nets: all_node_names.add(n)
            else:
                pass 
                # print(f"Warning: No output pin found in line: {line}")
        
        elif cmd == '.end':
            break

    # 去重并排序，确保顺序确定性
    input_names = sorted(list(set(input_names)))
    output_names = sorted(list(set(output_names)))
    
    sorted_node_names = sorted(list(all_node_names))
    name_to_idx = {name: i for i, name in enumerate(sorted_node_names)}
    num_nodes = len(sorted_node_names)
    
    # 初始化节点特征
    node_features = [node_type_map['UNKNOWN']] * num_nodes
    src_nodes, dst_nodes = [], []

    # 1. 标记输入节点
    for name in input_names:
        if name in name_to_idx:
            node_features[name_to_idx[name]] = node_type_map['INPUT']

    # 2. 构建图连接和标记门节点
    valid_gate_logic = [] 
    
    for out_name, g_type, in_names in gate_logic:
        if out_name not in name_to_idx: continue
        
        # 设置节点类型
        node_features[name_to_idx[out_name]] = node_type_map.get(g_type, node_type_map['UNKNOWN'])
        
        dst_idx = name_to_idx[out_name]
        for in_name in in_names:
            if in_name in name_to_idx:
                src_nodes.append(name_to_idx[in_name])
                dst_nodes.append(dst_idx)
        
        valid_gate_logic.append((out_name, g_type, in_names))

    # 3. 模拟电路获取真值表
    y = simulate_circuit(input_names, output_names, valid_gate_logic, file_path)

    return (torch.LongTensor(src_nodes),
            torch.LongTensor(dst_nodes),
            torch.LongTensor(node_features),
            y)

def create_dataset_from_blif_files(bench_root_dir, output_base_path='data_files/circuit_dag_processed', max_inputs=14, train_split=0.8, val_split=0.1):
    """
    主处理函数：扫描、筛选、解析 BLIF 并保存。
    """
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"创建输出目录: {output_base_path}")

    print(f"正在从 '{bench_root_dir}' 扫描 .blif 文件...")
    all_files = [os.path.join(bench_root_dir, f) for f in os.listdir(bench_root_dir) if f.endswith('.blif')]
    
    print(f"筛选输入数量 <= {max_inputs} 的文件...")
    eligible_files = [path for path in all_files if count_inputs_in_blif(path) <= max_inputs]
    
    print(f"找到 {len(all_files)} 个文件，其中 {len(eligible_files)} 个符合条件。")

    random.seed(42)
    random.shuffle(eligible_files)

    num_train = int(len(eligible_files) * train_split)
    num_val = int(len(eligible_files) * val_split)

    train_files = eligible_files[:num_train]
    val_files = eligible_files[num_train : num_train + num_val]
    test_files = eligible_files[num_train + num_val:]
    
    print(f"数据集划分为: {len(train_files)} (训练), {len(val_files)} (验证), {len(test_files)} (测试)")

    file_splits = {'train': train_files, 'val': val_files, 'test': test_files}
    node_type_map = NODE_TYPE_MAP
    
    for split_name, files in file_splits.items():
        print(f"--- 正在处理 {split_name} 数据集 ---")
        data_dict = defaultdict(list)
        
        for file_path in files:
            try:
                src, dst, x_n, y = parse_blif_file(file_path, node_type_map)
                
                if len(x_n) > 0:
                    data_dict['src_list'].append(src)
                    data_dict['dst_list'].append(dst)
                    data_dict['x_n_list'].append(x_n)
                    data_dict['y_list'].append(y)
                else:
                    print(f"警告：解析文件 {os.path.basename(file_path)} 得到一个空图，已跳过。")
            except Exception as e:
                # import traceback
                # traceback.print_exc()
                print(f"解析文件 {os.path.basename(file_path)} 时出错: {e}")

        output_file_path = os.path.join(output_base_path, f'{split_name}.pth')
        torch.save(dict(data_dict), output_file_path)
        print(f"已将 {len(data_dict['src_list'])} 个有效电路图保存到 {output_file_path}")

if __name__ == '__main__':
    # --- 用户设置 ---
    # 修改此处为你存放 .blif 文件的文件夹路径
    blif_files_directory = '/mnt/local_data2/liumiao/data/LSG/blifs' 
    output_directory = './data_files/netlist_dag_processed'
    # ---------------

    if not os.path.exists(blif_files_directory):
        print(f"提示：目录 '{blif_files_directory}' 不存在。请修改脚本中的路径。")
    else:
        create_dataset_from_blif_files(
            bench_root_dir=blif_files_directory,
            output_base_path=output_directory
        )