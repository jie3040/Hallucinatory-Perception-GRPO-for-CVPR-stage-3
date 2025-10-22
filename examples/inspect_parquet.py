"""
检查 Parquet 文件的数据格式
"""

import pyarrow.parquet as pq
import pandas as pd

def inspect_parquet(file_path: str):
    """检查 Parquet 文件格式"""
    
    print("=" * 80)
    print(f"检查文件: {file_path}")
    print("=" * 80)
    
    # 读取文件
    table = pq.read_table(file_path)
    df = table.to_pandas()
    
    print(f"\n文件信息:")
    print(f"  - 总行数: {len(df)}")
    print(f"  - 列名: {list(df.columns)}")
    print(f"  - 内存占用: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    print(f"\n各列类型:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    print(f"\n第一行数据详情:")
    print("-" * 80)
    first_row = df.iloc[0]
    for col in df.columns:
        value = first_row[col]
        print(f"\n{col}:")
        print(f"  类型: {type(value)}")
        if isinstance(value, (list, tuple)):
            print(f"  长度: {len(value)}")
            if len(value) > 0:
                print(f"  第一个元素类型: {type(value[0])}")
                print(f"  第一个元素: {value[0] if not isinstance(value[0], bytes) else '<bytes>'}")
        elif isinstance(value, dict):
            print(f"  字典键: {list(value.keys())}")
            for k, v in value.items():
                if isinstance(v, bytes):
                    print(f"    {k}: <bytes, 长度 {len(v)}>")
                else:
                    print(f"    {k}: {type(v)} = {str(v)[:100] if v is not None else None}")
        elif isinstance(value, bytes):
            print(f"  长度: {len(value)} bytes")
        else:
            print(f"  值: {str(value)[:200]}")
    
    print(f"\n最后一行数据详情:")
    print("-" * 80)
    last_row = df.iloc[-1]
    for col in df.columns:
        value = last_row[col]
        print(f"\n{col}:")
        print(f"  类型: {type(value)}")
        if isinstance(value, (list, tuple)):
            print(f"  长度: {len(value)}")
            if len(value) > 0:
                print(f"  第一个元素: {value[0] if not isinstance(value[0], bytes) else '<bytes>'}")
        elif isinstance(value, dict):
            print(f"  字典键: {list(value.keys())}")
        elif isinstance(value, bytes):
            print(f"  长度: {len(value)} bytes")
        else:
            print(f"  值: {str(value)[:200]}")
    
    # 检查 caption 字段
    print(f"\n\nCaption 字段详细分析:")
    print("-" * 80)
    captions = df['caption']
    
    # 统计不同类型
    types = {}
    for idx, cap in enumerate(captions):
        t = type(cap).__name__
        if t not in types:
            types[t] = {'count': 0, 'example_idx': idx}
        types[t]['count'] += 1
    
    print(f"Caption 类型分布:")
    for t, info in types.items():
        print(f"  - {t}: {info['count']} 个")
        example_idx = info['example_idx']
        example = captions.iloc[example_idx]
        print(f"    示例 (行{example_idx}): {example}")
    
    # 检查是否有 None
    none_count = captions.isna().sum()
    print(f"\nNone/NaN 数量: {none_count}")
    
    if none_count > 0:
        none_indices = captions[captions.isna()].index.tolist()[:5]
        print(f"前5个 None 的索引: {none_indices}")
    
    # 检查空列表
    if isinstance(captions.iloc[0], list):
        empty_lists = sum(1 for c in captions if isinstance(c, list) and len(c) == 0)
        print(f"空列表数量: {empty_lists}")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python inspect_parquet.py <parquet文件路径>")
        print("例如: python inspect_parquet.py /root/autodl-tmp/flickr30k/0000.parquet")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = inspect_parquet(file_path)
    
    print("\n" + "=" * 80)
    print("检查完成！")
    print("=" * 80)
