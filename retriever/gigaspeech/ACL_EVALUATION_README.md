# ACL 60-60 数据集评估功能

本文档介绍如何使用修改后的SONAR评估脚本来评估ACL 60-60数据集的术语检索性能。

## 📁 数据集结构

ACL 60-60数据集包含以下结构：
```
data/acl-6060/2/
├── acl_6060/
│   ├── dev/
│   │   ├── segmented_wavs/
│   │   │   ├── gold/          # 人工标注分段
│   │   │   └── shas/          # 自动分段
│   │   └── text/
│   │       └── tagged_terminology/  # 术语标注文件
│   └── eval/                  # 评估集，结构同dev
└── intermediate_files/
    └── terminology_glossary.csv  # 术语词汇表
```

## 🚀 主要功能

### 1. ACL术语提取
- **从CSV词汇表提取**: 从`terminology_glossary.csv`中提取英文术语
- **从标注文本提取**: 从`tagged_terminology`文件中提取方括号标记的术语

### 2. 数据集加载
- **音频加载**: 支持加载gold和shas分段的音频文件
- **术语匹配**: 自动匹配音频文件与对应的术语标注
- **数据验证**: 验证音频文件有效性和术语标注完整性

### 3. 评估模式
- **索引构建**: 从dev或eval集提取的术语构建检索索引
- **性能评估**: 计算recall@1, recall@5, recall@10
- **结果保存**: 保存详细的评估结果到JSON文件

## 📋 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 标准ACL评估（使用single best model）
bash SONAR_ACL_test.sh 2 term true data/samples/xl/term_level_chunks_500000_1000000.json true

# 使用full best model
bash SONAR_ACL_test.sh 2 term false data/samples/xl/term_level_chunks_500000_1000000.json true
```

### 方法2: 直接调用Python脚本

```bash
python3 SONAR_ACL_test.py \
  --model_path data/clap_sonar_term_level_single_best.pt \
  --acl_mode \
  --acl_root_dir data/acl-6060/2/acl_6060 \
  --acl_glossary_path data/acl-6060/2/intermediate_files/terminology_glossary.csv \
  --acl_test_split eval \
  --acl_index_split dev \
  --acl_segmentation gold \
  --max_eval 1000
```

## ⚙️ 参数配置

### 核心参数
- `--acl_mode`: 启用ACL评估模式
- `--model_path`: 训练好的SONAR模型路径

### ACL特定参数
- `--acl_root_dir`: ACL数据集根目录 (默认: `data/acl-6060/2/acl_6060`)
- `--acl_glossary_path`: 术语词汇表CSV文件路径
- `--acl_test_split`: 测试数据分割 (`dev`/`eval`)
- `--acl_index_split`: 索引构建数据分割 (`dev`/`eval`)
- `--acl_segmentation`: 分段类型 (`gold`/`shas`)

### 其他参数
- `--max_eval`: 最大评估样本数 (默认: 1000)
- `--batch_size`: 文本编码批次大小 (默认: 512)
- `--audio_batch_size`: 音频编码批次大小 (默认: 1000)

## 📊 评估流程

1. **术语索引构建**
   - 从ACL术语词汇表或dev set提取术语
   - 使用SONAR文本编码器对术语进行编码
   - 构建FAISS检索索引

2. **测试数据加载**
   - 加载指定分割的分段音频文件
   - 解析对应的术语标注文件
   - 匹配音频文件与术语标注

3. **音频编码与检索**
   - 使用SONAR音频编码器对测试音频编码
   - 在术语索引中进行相似度检索
   - 计算top-k召回率

4. **结果保存**
   - 保存评估结果到JSON文件
   - 包含详细的配置信息和性能指标

## 📈 评估指标

- **Sample-level Recall**: 样本级平均召回率
- **Term-level Recall**: 术语级微平均召回率
- **Top-k Recall**: k=1,5,10的召回率
- **未命中术语分析**: 详细的未检索到术语统计

## 🔧 配置示例

### 使用不同分割组合

```bash
# 使用dev作为测试集和索引源
python3 SONAR_ACL_test.py --acl_mode --acl_test_split dev --acl_index_split dev

# 使用eval作为测试集，dev作为索引源
python3 SONAR_ACL_test.py --acl_mode --acl_test_split eval --acl_index_split dev

# 使用shas分段（自动分段）
python3 SONAR_ACL_test.py --acl_mode --acl_segmentation shas
```

### 不同模型评估

```bash
# 使用single best模型
python3 SONAR_ACL_test.py --acl_mode --model_path data/clap_sonar_term_level_single_best.pt

# 使用full best模型  
python3 SONAR_ACL_test.py --acl_mode --model_path data/clap_sonar_term_level_full_best.pt
```

## 📁 输出文件

评估完成后会生成以下文件：
- `{model_name}_acl_{test_split}_eval_results.json`: 详细评估结果
- 控制台输出: 实时评估进度和性能指标

## 🔍 数据集统计

基于初步测试：
- **术语词汇表**: 约309个唯一英文术语
- **dev标注文本**: 约254个唯一术语
- **音频文件**: dev集468个，eval集类似数量
- **有效样本**: 根据音频质量和术语标注过滤后的样本数

## 🚨 注意事项

1. **数据路径**: 确保ACL数据集路径正确
2. **模型兼容**: 使用与训练数据格式兼容的模型
3. **内存使用**: 大批次音频编码可能消耗大量GPU内存
4. **术语匹配**: 术语匹配基于小写字符串完全匹配

## 📝 示例输出

```
[INFO] ACL Evaluation Setup:
[INFO] - Index split: dev (309 terms)
[INFO] - Test split: eval (XXX samples)
[INFO] - Segmentation: gold

[EVAL] Sample-level Average Recall@1: XX.XX%
[EVAL] Sample-level Average Recall@5: XX.XX%
[EVAL] Sample-level Average Recall@10: XX.XX%

[EVAL] Term-level Micro-Average Recall@1: XX.XX% (XX/XXX terms)
[EVAL] Term-level Micro-Average Recall@5: XX.XX% (XX/XXX terms)
[EVAL] Term-level Micro-Average Recall@10: XX.XX% (XX/XXX terms)
```

这样的修改使得SONAR评估系统能够无缝支持ACL 60-60数据集，为多语言会议术语检索提供了标准化的评估框架。 