import os
import torch
import re

def parse_bench_file(file_path, node_type_map):
    """
    解析单个 .bench 文件，并将其转换为图结构。

    Args:
        file_path (str): .bench 文件的路径。
        node_type_map (dict): 将节点类型（如 'INPUT', 'AND'）映射到整数的字典。

    Returns:
        tuple: 包含 src, dst, x_n, y 张量的元组。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    name_to_idx = {}
    idx_counter = 0
    node_features = []
    src_nodes = []
    dst_nodes = []

    # 正则表达式来解析不同类型的行
    input_pattern = re.compile(r"INPUT\((.+)\)")
    gate_pattern = re.compile(r"(.+?)\s*=\s*([A-Z]+)\((.+)\)")

    # 第一次遍历：识别所有输入节点并分配索引
    for line in lines:
        match = input_pattern.match(line.strip())
        if match:
            node_name = match.group(1)
            if node_name not in name_to_idx:
                name_to_idx[node_name] = idx_counter
                node_features.append(node_type_map['INPUT'])
                idx_counter += 1

    # 第二次遍历：处理逻辑门和连接
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('OUTPUT'):
            continue
        
        match = gate_pattern.match(line)
        if match:
            output_name, gate_type, inputs_str = match.groups()
            output_name = output_name.strip()
            inputs = [name.strip() for name in inputs_str.split(',')]

            # 为输出门节点分配索引
            if output_name not in name_to_idx:
                name_to_idx[output_name] = idx_counter
                
                # 如果遇到未知的门类型，可以给一个默认值或抛出错误
                node_type_id = node_type_map.get(gate_type, node_type_map.get('UNKNOWN', -1))
                node_features.append(node_type_id)
                idx_counter += 1

            # 创建从输入到输出的边
            output_idx = name_to_idx[output_name]
            for input_name in inputs:
                if input_name in name_to_idx:
                    input_idx = name_to_idx[input_name]
                    src_nodes.append(input_idx)
                    dst_nodes.append(output_idx)
                else:
                    # 这个情况理论上不应该发生，因为 bench 文件是按拓扑顺序排列的
                    print(f"警告：在文件 {file_path} 中找不到输入节点 '{input_name}'")

    # 生成一个虚拟的图标签 y，你可以根据需要替换为真实数据
    # 例如，从文件名或另一个文件中读取电路的延迟、面积等
    y = torch.tensor(float(len(node_features))) # 例如，使用节点数量作为标签

    return (torch.LongTensor(src_nodes),
            torch.LongTensor(dst_nodes),
            torch.LongTensor(node_features),
            y)

def create_dataset_from_bench_files(bench_root_dir, output_base_path='data_files/circuit_dag_processed'):
    """
    遍历包含 .bench 文件的目录，解析它们，并创建数据集文件。
    """
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"创建输出目录: {output_base_path}")

    # 定义节点类型的词汇表
    # 你可以根据你的 .bench 文件扩充这个词汇表
    node_type_map = {'INPUT': 0, 'AND': 1, 'NOT': 2, 'UNKNOWN': 3}
    
    splits = ['train', 'val', 'test']
    for split_name in splits:
        print(f"--- 正在处理 {split_name} 数据集 ---")
        split_dir = os.path.join(bench_root_dir, split_name)
        
        if not os.path.isdir(split_dir):
            print(f"警告：找不到目录 {split_dir}，跳过该部分。")
            continue

        data_dict = {
            'src_list': [], 'dst_list': [], 'x_n_list': [], 'y_list': []
        }
        
        bench_files = [f for f in os.listdir(split_dir) if f.endswith('.bench')]
        for file_name in bench_files:
            file_path = os.path.join(split_dir, file_name)
            try:
                src, dst, x_n, y = parse_bench_file(file_path, node_type_map)
                data_dict['src_list'].append(src)
                data_dict['dst_list'].append(dst)
                data_dict['x_n_list'].append(x_n)
                data_dict['y_list'].append(y)
            except Exception as e:
                print(f"解析文件 {file_name} 时出错: {e}")

        output_file_path = os.path.join(output_base_path, f'{split_name}.pth')
        torch.save(data_dict, output_file_path)
        print(f"已将 {len(bench_files)} 个电路图保存到 {output_file_path}")

def create_dummy_bench_files(root_dir='dummy_bench_files'):
    """为了让脚本可以独立运行，创建一个虚拟的 .bench 文件结构。"""
    if os.path.exists(root_dir):
        return # 如果已存在则不重复创建
    print(f"创建虚拟 .bench 文件到目录 '{root_dir}'...")
    sample_content = """
# sample.bench
INPUT(A)
INPUT(B)
C = NOT(A)
D = NOT(B)
E = AND(C, D)
F = AND(A, B)
G = AND(E, F)
"""
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(root_dir, split)
        os.makedirs(split_path, exist_ok=True)
        for i in range(5): # 每个集合创建5个文件
            with open(os.path.join(split_path, f'circuit_{i}.bench'), 'w') as f:
                f.write(sample_content.replace('A', f'n{i*10}').replace('B', f'n{i*10+1}'))
    print("虚拟文件创建完成。")

if __name__ == '__main__':
    # 1. 创建一些虚拟的 .bench 文件用于演示
    dummy_data_dir = 'dummy_bench_files'
    create_dummy_bench_files(dummy_data_dir)
    
    # 2. 运行主函数，将虚拟的 .bench 文件转换为 .pth 数据集
    create_dataset_from_bench_files(
        bench_root_dir=dummy_data_dir,
        output_base_path='data_files/circuit_dag_processed'
    )
# ```

### 主要改动和如何使用

# 1.  **核心解析逻辑 (`parse_bench_file`)**：
#     * 此函数是新增的核心，负责读取单个 `.bench` 文件。
#     * 它使用正则表达式来识别 `INPUT` 和逻辑门（如 `AND`, `NOT`）的行。
#     * 它会构建一个从节点名称（如 `'0'`, `'new_n45'`）到整数索引的映射，并根据这个映射创建图的边列表 (`src`, `dst`) 和节点特征列表 (`x_n`)。
#     * 节点特征现在是节点的类型，例如 `INPUT` 对应 `0`，`AND` 对应 `1`。

# 2.  **主处理流程 (`create_dataset_from_bench_files`)**：
#     * 这个函数取代了原来生成随机数据的 `create_and_save_data`。
#     * 你需要提供一个根目录，该目录下应包含 `train`、`val` 和 `test` 三个子目录，每个子目录里存放着对应的 `.bench` 文件。
#     * 它会遍历所有文件，调用 `parse_bench_file` 进行解析，并将结果整理成与之前格式完全相同的字典，最后保存为 `.pth` 文件。

# 3.  **如何运行**：
#     * **准备你的数据**：按照下面的结构组织你的 `.bench` 文件：
#         ```
#         my_circuits/
#         ├── train/
#         │   ├── circuit1.bench
#         │   └── circuit2.bench
#         ├── val/
#         │   └── circuit3.bench
#         └── test/
#             └── circuit4.bench
        

