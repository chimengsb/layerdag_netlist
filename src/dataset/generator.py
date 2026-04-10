import os
import torch
import random
import numpy as np

def generate_single_dag(num_node_types=10):
    """
    生成一个随机的有向无环图 (DAG)。
    
    为了保证图是无环的，我们只允许从索引较小的节点连接到索引较大的节点。
    """
    # 随机决定节点数量，例如 5 到 20 个节点
    num_nodes = random.randint(5, 20)
    
    # 1. 生成节点特征 (x_n)
    # 每个节点的特征是一个整数，代表它的类别
    x_n = torch.randint(0, num_node_types, (num_nodes,))
    
    # 2. 生成边 (src, dst)
    src_nodes = []
    dst_nodes = []
    # 遍历所有可能的边（从 i 到 j，其中 i < j）
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # 以 30% 的概率创建一条边
            if random.random() < 0.3:
                src_nodes.append(i)
                dst_nodes.append(j)
    
    src = torch.LongTensor(src_nodes)
    dst = torch.LongTensor(dst_nodes)
    
    # 3. 生成图的标签 (y)，例如一个随机的浮点数
    y = torch.tensor(random.uniform(0.1, 10.0))
    
    return src, dst, x_n, y

def create_and_save_data(base_path='data_files/my_dag_processed', num_node_types=10):
    """
    创建训练、验证和测试数据集，并将其保存到文件。
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"创建目录: {base_path}")

    splits = {
        'train': 100,  # 100 个训练图
        'val': 20,     # 20 个验证图
        'test': 20     # 20 个测试图
    }

    for split_name, num_graphs in splits.items():
        print(f"正在生成 {split_name} 数据集...")
        data_dict = {
            'src_list': [],
            'dst_list': [],
            'x_n_list': [],
            'y_list': []
        }
        
        for _ in range(num_graphs):
            src, dst, x_n, y = generate_single_dag(num_node_types)
            data_dict['src_list'].append(src)
            data_dict['dst_list'].append(dst)
            data_dict['x_n_list'].append(x_n)
            data_dict['y_list'].append(y)
            
        file_path = os.path.join(base_path, f'{split_name}.pth')
        torch.save(data_dict, file_path)
        print(f"已将 {num_graphs} 个图保存到 {file_path}")

if __name__ == '__main__':
    # 运行此脚本以生成数据文件
    create_and_save_data()
