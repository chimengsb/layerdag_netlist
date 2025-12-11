import os
import torch
import sys

# 尝试导入 DAGDataset
# 如果 general.py 在同一目录下，使用 .general
# 如果作为脚本运行，可能需要调整路径
try:
    from .general import DAGDataset
except ImportError:
    # 假设 general.py 在同一级目录，用于简单测试
    from general import DAGDataset

def to_dag_dataset(data_dict, num_categories):
    """
    将包含列表数据的字典转换为 DAGDataset 对象。
    
    Args:
        data_dict: 包含 'src_list', 'dst_list', 'x_n_list', 'y_list' 的字典
        num_categories: 节点类型的总数 (用于定义 dummy node 的 ID)
    """
    # 初始化 DAGDataset，label=True 表示包含 y (真值表)
    dataset = DAGDataset(num_categories=num_categories, label=True)

    src_list = data_dict['src_list']
    dst_list = data_dict['dst_list']
    x_n_list = data_dict['x_n_list']
    y_list = data_dict['y_list']

    num_g = len(src_list)
    for i in range(num_g):
        # 将每个图的数据添加到 dataset 中
        dataset.add_data(src_list[i],
                         dst_list[i],
                         x_n_list[i],
                         y_list[i])

    return dataset

def get_blif_dataset(processed_data_dir=None):
    """
    加载并返回 BLIF DAG 数据集 (Train, Val, Test)。
    
    Args:
        processed_data_dir: .pth 文件所在的目录路径。
                            如果为 None，默认寻找相对于当前文件的 ../../data_files/circuit_dag_processed
    """
    if processed_data_dir is None:
        # 获取当前脚本的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 默认路径假设: 项目根目录/data_files/circuit_dag_processed
        # 根据 process_blif_dag.py 的默认输出调整
        processed_data_dir = '/mnt/local_data2/liumiao/code/LSG/LayerDAG_netlist/src/dataset/data_files/netlist_dag_processed'

    train_path = os.path.join(processed_data_dir, 'train.pth')
    val_path = os.path.join(processed_data_dir, 'val.pth')
    test_path = os.path.join(processed_data_dir, 'test.pth')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"BLIF dataset not found at {train_path}. Please run process_blif_dag.py first.")

    print(f'Loading BLIF DAG dataset from {processed_data_dir}...')
    
    # 加载预处理后的 .pth 文件
    train_dict = torch.load(train_path)
    val_dict = torch.load(val_path)
    test_dict = torch.load(test_path)

    # 动态计算类别数量 (max category ID + 1)
    # x_n_list 中的值对应节点类型 ID (0: INPUT, 1: AND, 2: NOT, ...)
    # 我们需要找到所有数据集中最大的类型 ID
    all_x_n = train_dict['x_n_list'] + val_dict['x_n_list'] + test_dict['x_n_list']
    if len(all_x_n) > 0:
        # 注意：这里假设 x_n_list 里的元素是 tensor
        # 如果是 list，可能需要先转换或迭代查找
        max_cat = 0
        for x in all_x_n:
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                max_cat = max(max_cat, x.max().item())
            elif isinstance(x, list) and len(x) > 0:
                max_cat = max(max_cat, max(x))
        num_categories = max_cat + 1
    else:
        # 默认值，如果数据集为空（虽然不太可能）
        num_categories = 12 # 基于之前的 NODE_TYPE_MAP

    print(f"Detected number of node categories: {num_categories}")

    # 转换为 DAGDataset 对象
    train_set = to_dag_dataset(train_dict, num_categories)
    val_set = to_dag_dataset(val_dict, num_categories)
    test_set = to_dag_dataset(test_dict, num_categories)

    return train_set, val_set, test_set

if __name__ == "__main__":
    # 简单的测试代码
    try:
        # 你可以修改这里的路径进行测试
        test_dir = './data_files/netlist_dag_processed'
        if os.path.exists(test_dir):
            train, val, test = get_blif_dataset(test_dir)
            print(f"Successfully loaded datasets:")
            print(f"  Train size: {len(train)}")
            print(f"  Val size:   {len(val)}")
            print(f"  Test size:  {len(test)}")
            
            # 检查第一个样本
            src, dst, x_n, y = train[0]
            print("\nSample 0 info:")
            print(f"  Nodes: {x_n.shape}")
            print(f"  Edges: {src.shape}")
            print(f"  Truth Table shape: {y.shape}")
        else:
            print(f"Test directory {test_dir} does not exist. Skipping test.")
    except Exception as e:
        print(f"An error occurred: {e}")