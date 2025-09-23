# Ground Truth Terms 数据清洗工具

这个工具用于清洗 `term_preprocessed_samples_*.json` 文件中的 `ground_truth_term` 字段，移除在 `glossary_merged.json` 中标记为 `confused=true` 或不存在的术语。

## 文件说明

### 核心脚本
- `clean_ground_truth_terms.py` - 主要的Python清洗脚本
- `run_clean_ground_truth_terms.sh` - 单文件处理脚本
- `run_clean_all_files.sh` - 批量处理所有文件的脚本

### 输入文件
- `data/terms/glossary_merged.json` - 术语词典文件
- `data/samples/xl/term_preprocessed_samples_*.json` - 待清洗的样本文件

### 输出文件
- `data/samples/xl_cleaned/term_preprocessed_samples_*.json` - 清洗后的样本文件

## 使用方法

### 方法1：处理单个文件
```bash
python3 clean_ground_truth_terms.py \
    --glossary data/terms/glossary_merged.json \
    --input-dir data/samples/xl \
    --output-dir data/samples/xl_cleaned \
    --files term_preprocessed_samples_0_500000.json
```

### 方法2：使用便捷脚本处理单个文件
```bash
./run_clean_ground_truth_terms.sh
```

### 方法3：批量处理所有文件
```bash
./run_clean_all_files.sh
```

## 清洗规则

脚本会对每个样本的 `ground_truth_term` 字段进行以下处理：

1. **保留条件**：只保留在 `glossary_merged.json` 中存在且 `confused=false` 的术语
2. **移除条件**：移除以下术语：
   - 在glossary中不存在的术语
   - 在glossary中标记为 `confused=true` 的术语
3. **更新字段**：
   - `ground_truth_term`: 更新为清洗后的术语列表
   - `has_target`: 根据清洗后是否还有术语来设置true/false

## 输出统计信息

脚本会输出详细的统计信息：
- 总样本数
- 有目标术语的样本数
- 原始术语总数
- 清洗后术语总数
- 移除的术语总数
- 被移除的具体术语列表

## 示例输出

```
Processing data/samples/xl/term_preprocessed_samples_0_500000.json...
Loaded 88784 samples
Sample POD0000000001_S0000010: Removed terms {'Pasta'}
...
Saved cleaned data to data/samples/xl_cleaned/term_preprocessed_samples_0_500000.json
Statistics:
  - Total samples: 88784
  - Samples with targets: 33296
  - Original terms: 117889
  - Cleaned terms: 37992
  - Removed terms: 79897
```

## 支持的文件列表

脚本支持处理以下17个文件：
- term_preprocessed_samples_0_500000.json
- term_preprocessed_samples_500000_1000000.json
- term_preprocessed_samples_1000000_1500000.json
- term_preprocessed_samples_1500000_2000000.json
- term_preprocessed_samples_2000000_2500000.json
- term_preprocessed_samples_2500000_3000000.json
- term_preprocessed_samples_3000000_3500000.json
- term_preprocessed_samples_3500000_4000000.json
- term_preprocessed_samples_4000000_4500000.json
- term_preprocessed_samples_4500000_5000000.json
- term_preprocessed_samples_5000000_5500000.json
- term_preprocessed_samples_5500000_6000000.json
- term_preprocessed_samples_6000000_6500000.json
- term_preprocessed_samples_6500000_7000000.json
- term_preprocessed_samples_7000000_7500000.json
- term_preprocessed_samples_7500000_8000000.json
- term_preprocessed_samples_8000000_end.json

## 注意事项

1. 确保 `glossary_merged.json` 文件存在且可读
2. 输出目录会自动创建
3. 如果输入文件不存在，会跳过并给出警告
4. 处理大文件时可能需要较长时间和较多内存
5. 建议先用小文件测试确认结果正确

