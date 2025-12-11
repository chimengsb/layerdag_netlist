import os
import torch

# from .general import DAGDataset

from torch.utils.data import Dataset

class DAGDataset(Dataset):
    """
    Parameters
    ----------
    label : bool
        Whether each DAG has a label like runtime/latency.
    """
    def __init__(self, num_categories, label=False):
        self.src = []
        self.dst = []
        self.x_n = []

        self.label = label
        if self.label:
            self.y = []

        self.dummy_category = num_categories
        if isinstance(self.dummy_category, torch.Tensor):
            self.dummy_category = self.dummy_category.tolist()

        self.num_categories = num_categories + 1

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        if self.label:
            return self.src[index], self.dst[index], self.x_n[index], self.y[index]
        else:
            return self.src[index], self.dst[index], self.x_n[index]

    def add_data(self, src, dst, x_n, y=None):
        self.src.append(src)
        self.dst.append(dst)
        self.x_n.append(x_n)
        if (y is not None) and (self.label):
            self.y.append(y)




def to_dag_dataset(data_dict, num_categories):
    """
    将包含图数据列表的字典转换为 DAGDataset 对象。
    此函数与 tpu_tile 数据加载器中的版本相同。
    """
    dataset = DAGDataset(num_categories=num_categories, label=True)

    src_list = data_dict['src_list']
    dst_list = data_dict['dst_list']
    x_n_list = data_dict['x_n_list']
    y_list = data_dict['y_list']

    num_g = len(src_list)
    
    
    
    for i in range(num_g):
        x_n_i = x_n_list[i]
        src_i = src_list[i]
        dst_i = dst_list[i]
        num_nodes_in_graph = len(x_n_i)

        # 检查1: 节点特征ID是否越界 (之前的修复)
        if x_n_i.numel() > 0:
            max_val_in_graph = x_n_i.max().item()
            assert max_val_in_graph < num_categories, \
                f"Error in graph data {i}: Node feature ID {max_val_in_graph} is out of bounds. " \
                f"Calculated num_categories is {num_categories} (valid real IDs are 0 to {num_categories - 1})."

        # --- START OF NEW FIX ---
        # 检查2: 边列表中的节点实例ID是否越界。
        if src_i.numel() > 0:
            max_src_index = src_i.max().item()
            assert max_src_index < num_nodes_in_graph, \
                f"Error in graph data {i}: A source node index ({max_src_index}) is out of bounds. " \
                f"The graph has only {num_nodes_in_graph} nodes (indices 0 to {num_nodes_in_graph - 1})."
        
        if dst_i.numel() > 0:
            max_dst_index = dst_i.max().item()
            assert max_dst_index < num_nodes_in_graph, \
                f"Error in graph data {i}: A destination node index ({max_dst_index}) is out of bounds. " \
                f"The graph has only {num_nodes_in_graph} nodes (indices 0 to {num_nodes_in_graph - 1})."
        # --- END OF NEW FIX ---

        dataset.add_data(src_list[i],
                         dst_list[i],
                         x_n_list[i],
                         y_list[i])

    return dataset

def get_circuit_bench():
    """
    加载预处理好的 circuit_bench 数据集。
    """
    # 1. 定位到包含 .pth 文件的数据目录
    # root_path = os.path.dirname(os.path.abspath(__file__))
    # 假设此文件的目录结构与 tpu_tile.py 相同
    root_path = '/mnt/local_data2/liumiao/code/LSG/LayerDAG/src/dataset/data_files/circuit_dag_processed'

    train_path = os.path.join(root_path, 'train.pth')
    val_path = os.path.join(root_path, 'val.pth')
    test_path = os.path.join(root_path, 'test.pth')

    print('Loading Circuit Bench dataset...')
    # 2. 使用 torch.load 加载数据字典
    train_set = torch.load(train_path)
    val_set = torch.load(val_path)
    test_set = torch.load(test_path)

    # 3. FIX: 从所有数据分割（train, val, test）中计算总的节点类别数，
    #    以防止验证集或测试集中出现训练集中没有的节点类型。
    all_x_n = train_set['x_n_list'] + val_set['x_n_list'] + test_set['x_n_list']
    
    if all_x_n:
        # 过滤掉空的张量，然后拼接
        non_empty_x_n = [t for t in all_x_n if t.numel() > 0]
        if non_empty_x_n:
            num_categories = torch.cat(non_empty_x_n).max().item() + 1
        else:
            num_categories = 10 # 备用值
            print("警告: 数据集中所有图都没有节点，使用默认类别数 10。")
    else:
        num_categories = 10 # 备用值
        print("警告: 无法从数据中推断节点类别数，使用默认值 10。")
    
    print(f"INFO: Determined number of node categories: {num_categories}")
    
    # 4. 将字典转换为 DAGDataset 对象
    train_set = to_dag_dataset(train_set, num_categories)
    val_set = to_dag_dataset(val_set, num_categories)
    test_set = to_dag_dataset(test_set, num_categories)

    return train_set, val_set, test_set
# ```

# import sys
# sys.setrecursionlimit(10000)

import numpy as np
if __name__ == '__main__':
    train_set, val_set, test_set = get_circuit_bench()
    torch.set_printoptions(threshold=np.inf)
    for x in train_set:
        xxx  = x[3]
        xxx = xxx.to(torch.int64)
        mask = (xxx != 1) & (xxx != 0)
        indices = torch.nonzero(mask)
        xx = sum(xxx)
        if (xx[0]<0):
          print(indices)
          # print('xxx',xxx)
          # breakpoint()
        # print('xxx',sum(xxx))



# ### 如何使用

# 现在，您可以在您的主训练脚本中，像导入 `get_tpu_tile` 一样导入和使用 `get_circuit_bench` 函数来加载您的电路数据集。

# 例如，在一个名为 `dataset.py` 的 `__init__.py` 文件中，您可以这样组织：

# ```python
# # In datasets/__init__.py

# from .tpu_tile import get_tpu_tile
# from .circuit_bench_dataset import get_circuit_bench

# def load_dataset(name):
#     if name == 'tpu_tile':
#         return get_tpu_tile()
#     elif name == 'circuit_bench':
#         return get_circuit_bench()
#     else:
#         raise ValueError(f'Unknown dataset: {name}')
