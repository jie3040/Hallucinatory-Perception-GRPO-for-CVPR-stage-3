"""
处理本地 Flickr30k Parquet 文件（EasyR1 兼容版）
生成符合 EasyR1 格式的 data.parquet 文件
"""

import pandas as pd
import pyarrow.parquet as pq
import os
from datasets import Dataset
from PIL import Image
import io
from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil


def process_local_flickr30k(
    parquet_dir: str = "./flickr30k",
    output_dir: str = "./data/flickr30k_caption",
    train_ratio: float = 0.9,
    prompt_text: str = "Give a caption of the picture",
    seed: int = 42
):
    """
    处理本地 Flickr30k Parquet 文件，生成 EasyR1 兼容格式
    
    Args:
        parquet_dir: Parquet 文件所在目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        prompt_text: Prompt 文本
        seed: 随机种子
    """
    
    print("=" * 80)
    print("处理本地 Flickr30k Parquet 文件（EasyR1 兼容版）")
    print("=" * 80)
    
    # 1. 构建文件路径
    file_paths = [os.path.join(parquet_dir, f"{i:04d}.parquet") for i in range(9)]
    
    # 检查文件是否存在
    print("\n步骤 1/6: 检查文件")
    print("-" * 80)
    existing_files = []
    for path in file_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"✓ {path} ({file_size:.2f} MB)")
            existing_files.append(path)
        else:
            print(f"✗ 文件不存在: {path}")
    
    if not existing_files:
        raise FileNotFoundError(f"在 {parquet_dir} 中没有找到任何 Parquet 文件！")
    
    print(f"\n找到 {len(existing_files)} 个文件")
    
    # 2. 读取第一个文件查看原始表头
    print("\n步骤 2/6: 检查数据结构")
    print("-" * 80)
    first_table = pq.read_table(existing_files[0])
    print(f"原始表头: {first_table.column_names}")
    
    first_df = first_table.to_pandas()
    print(f"第一个文件行数: {len(first_df)}")
    
    # 显示第一个样本的 caption
    sample_caption = first_df['caption'].iloc[0]
    print(f"\nCaption 示例:")
    print(f"  类型: {type(sample_caption)}")
    if isinstance(sample_caption, np.ndarray):
        print(f"  长度: {len(sample_caption)}")
        print(f"  第一个: {sample_caption[0]}")
    
    # 3. 处理所有文件
    print("\n步骤 3/6: 处理所有 Parquet 文件")
    print("-" * 80)
    
    all_data = []  # 存储处理后的数据字典
    
    for file_idx, file_path in enumerate(existing_files):
        print(f"\n处理文件 {file_idx + 1}/{len(existing_files)}: {os.path.basename(file_path)}")
        
        # 读取 parquet 文件
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        print(f"  行数: {len(df)}")
        
        # 处理每一行
        success_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  处理行"):
            # 1. 处理图像
            image_data = row['image']
            
            try:
                # image 是一个字典，包含 'bytes' 和 'path'
                if isinstance(image_data, dict):
                    if 'bytes' in image_data and image_data['bytes'] is not None:
                        # 从字节数据创建图像
                        image = Image.open(io.BytesIO(image_data['bytes']))
                    elif 'path' in image_data and image_data['path'] is not None:
                        # 从路径加载图像
                        image = Image.open(image_data['path'])
                    else:
                        continue
                elif isinstance(image_data, Image.Image):
                    image = image_data
                elif isinstance(image_data, bytes):
                    image = Image.open(io.BytesIO(image_data))
                else:
                    continue
                
                # 确保是 RGB 格式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
            except Exception as e:
                continue
            
            # 2. 处理 caption - numpy.ndarray，取第一个
            caption_data = row['caption']
            
            try:
                if isinstance(caption_data, np.ndarray):
                    if len(caption_data) > 0:
                        caption = str(caption_data[0]).strip()
                    else:
                        continue
                elif isinstance(caption_data, list):
                    if len(caption_data) > 0:
                        caption = str(caption_data[0]).strip()
                    else:
                        continue
                elif isinstance(caption_data, str):
                    caption = caption_data.strip()
                else:
                    continue
                
                # 检查 caption 是否有效
                if not caption or len(caption) < 2:
                    continue
                
            except Exception as e:
                continue
            
            # 3. 构建 EasyR1 格式的数据项
            # 注意：images 是列表，videos 是空列表
            data_item = {
                'images': [image],              # 图像列表
                'caption': caption,             # caption 字符串
                'prompt': prompt_text,          # prompt 字符串
                'videos': []                    # 空的 videos 字段（EasyR1 需要）
            }
            
            all_data.append(data_item)
            success_count += 1
        
        print(f"  成功处理: {success_count}/{len(df)}")
    
    print(f"\n✓ 处理完成")
    print(f"  - 成功处理的样本数: {len(all_data)}")
    
    if len(all_data) == 0:
        raise ValueError("没有成功处理任何样本！请检查数据格式。")
    
    # 4. 划分训练集和验证集
    print("\n步骤 4/6: 划分训练集和验证集")
    print("-" * 80)
    
    total_samples = len(all_data)
    train_size = int(total_samples * train_ratio)
    
    # 设置随机种子并打乱
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 构建训练集和验证集
    train_data = [all_data[i] for i in train_indices]
    val_data = [all_data[i] for i in val_indices]
    
    print(f"训练集: {len(train_data)} 样本 ({train_ratio*100:.1f}%)")
    print(f"验证集: {len(val_data)} 样本 ({(1-train_ratio)*100:.1f}%)")
    
    # 5. 创建目录结构
    print("\n步骤 5/6: 创建目录结构")
    print("-" * 80)
    
    output_base = Path(output_dir)
    
    # 清理旧目录（如果存在）
    if output_base.exists():
        print(f"清理旧目录: {output_base}")
        shutil.rmtree(output_base)
    
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 创建目录:")
    print(f"  - {train_dir}")
    print(f"  - {val_dir}")
    
    # 6. 保存为 data.parquet（EasyR1 格式）
    print("\n步骤 6/6: 保存为 EasyR1 格式的 data.parquet")
    print("-" * 80)
    
    def save_split(data_list, output_path, name):
        """保存数据集为 data.parquet"""
        print(f"\n保存 {name}...")
        print(f"  样本数: {len(data_list)}")
        
        # 创建 Dataset
        ds = Dataset.from_list(data_list)
        
        # 保存为 data.parquet（重要：文件名必须是 data.parquet）
        parquet_file = output_path / "data.parquet"
        ds.to_parquet(str(parquet_file))
        
        file_size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"✓ 已保存: {parquet_file}")
        print(f"  文件大小: {file_size_mb:.2f} MB")
        
        return parquet_file
    
    train_parquet = save_split(train_data, train_dir, "训练集")
    val_parquet = save_split(val_data, val_dir, "验证集")
    
    # 7. 数据集统计
    print("\n" + "=" * 80)
    print("数据集统计")
    print("=" * 80)
    
    def get_caption_stats(data_list):
        captions = [item['caption'] for item in data_list]
        lengths = [len(c.split()) for c in captions]
        return {
            'count': len(captions),
            'avg_length': np.mean(lengths),
            'median_length': np.median(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'std_length': np.std(lengths),
        }
    
    train_stats = get_caption_stats(train_data)
    val_stats = get_caption_stats(val_data)
    
    print("\n训练集:")
    print(f"  - 样本数: {train_stats['count']}")
    print(f"  - Caption 平均长度: {train_stats['avg_length']:.2f} 词")
    print(f"  - Caption 中位数长度: {train_stats['median_length']:.0f} 词")
    print(f"  - Caption 长度范围: {train_stats['min_length']:.0f}-{train_stats['max_length']:.0f} 词")
    
    print("\n验证集:")
    print(f"  - 样本数: {val_stats['count']}")
    print(f"  - Caption 平均长度: {val_stats['avg_length']:.2f} 词")
    print(f"  - Caption 中位数长度: {val_stats['median_length']:.0f} 词")
    print(f"  - Caption 长度范围: {val_stats['min_length']:.0f}-{val_stats['max_length']:.0f} 词")
    
    # 8. 显示样本
    print("\n" + "=" * 80)
    print("样本示例")
    print("=" * 80)
    
    for i in range(min(3, len(train_data))):
        sample = train_data[i]
        image = sample['images'][0]
        print(f"\n样本 #{i+1}:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Caption: {sample['caption']}")
        print(f"  Image size: {image.size}")
        print(f"  Videos: {sample['videos']}")
    
    # 9. 生成使用说明
    print("\n" + "=" * 80)
    print("✓ 数据准备完成！")
    print("=" * 80)
    print(f"\n数据集已保存（EasyR1 格式）:")
    print(f"  - 训练集: {train_dir}/data.parquet")
    print(f"  - 验证集: {val_dir}/data.parquet")
    print(f"\n使用方式:")
    print(f"在 EasyR1 训练配置中设置:")
    print(f"  data.train_files={train_dir}")
    print(f"  data.val_files={val_dir}")
    print(f"\n或在命令行中:")
    print(f"  python3 -m verl.trainer.main \\")
    print(f"    config=examples/caption_config.yaml \\")
    print(f"    data.train_files={train_dir} \\")
    print(f"    data.val_files={val_dir}")
    
    return train_dir, val_dir


def verify_easyr1_dataset(dataset_dir: str):
    """
    验证 EasyR1 格式的数据集
    
    Args:
        dataset_dir: 数据集目录（包含 data.parquet）
    """
    from datasets import load_dataset
    
    print("\n" + "=" * 80)
    print(f"验证 EasyR1 数据集: {dataset_dir}")
    print("=" * 80)
    
    try:
        # 检查 data.parquet 是否存在
        parquet_file = Path(dataset_dir) / "data.parquet"
        if not parquet_file.exists():
            print(f"✗ 文件不存在: {parquet_file}")
            return False
        
        file_size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
        print(f"✓ 找到文件: {parquet_file}")
        print(f"  文件大小: {file_size_mb:.2f} MB")
        
        # 加载数据集
        print("\n加载数据集...")
        dataset = load_dataset("parquet", data_files=str(parquet_file))['train']
        
        print(f"✓ 数据集加载成功")
        print(f"  - 样本数: {len(dataset)}")
        print(f"  - 字段: {list(dataset.features.keys())}")
        
        # 检查第一个样本
        sample = dataset[0]
        print(f"\n第一个样本:")
        print(f"  - images: {type(sample['images'])}, 长度: {len(sample['images'])}")
        if len(sample['images']) > 0:
            img = sample['images'][0]
            print(f"    第一个图像: {type(img)}, 大小: {img.size}")
        print(f"  - caption: {sample['caption'][:100]}...")
        print(f"  - prompt: {sample['prompt']}")
        print(f"  - videos: {type(sample['videos'])}, 长度: {len(sample['videos'])}")
        
        print(f"\n✓ 数据集格式正确，可以用于 EasyR1 训练")
        return True
        
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="处理本地 Flickr30k Parquet 文件为 EasyR1 格式")
    parser.add_argument("--parquet_dir", type=str, 
                        default="./flickr30k",
                        help="Parquet 文件所在目录 (default: ./flickr30k)")
    parser.add_argument("--output_dir", type=str, 
                        default="./data/flickr30k_easyr1",
                        help="输出目录 (default: ./data/flickr30k_easyr1)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="训练集比例 (default: 0.9)")
    parser.add_argument("--prompt", type=str, 
                        default="Give a caption of the picture",
                        help="Prompt 文本")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--verify", type=str, default=None,
                        help="验证已存在的数据集目录")
    
    args = parser.parse_args()
    
    if args.verify:
        # 验证模式
        verify_easyr1_dataset(args.verify)
    else:
        # 处理数据集
        try:
            train_dir, val_dir = process_local_flickr30k(
                parquet_dir=args.parquet_dir,
                output_dir=args.output_dir,
                train_ratio=args.train_ratio,
                prompt_text=args.prompt,
                seed=args.seed
            )
            
            # 自动验证
            print("\n\n" + "=" * 80)
            print("自动验证训练集...")
            verify_easyr1_dataset(train_dir)
            
            print("\n" + "=" * 80)
            print("自动验证验证集...")
            verify_easyr1_dataset(val_dir)
            
        except Exception as e:
            print(f"\n✗ 处理失败: {e}")
            import traceback
            traceback.print_exc()
