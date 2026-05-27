import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import librosa
import re
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, fbeta_score, auc
import torch
from datetime import datetime

# ================= 配置区域 =================
CONFIG = {
    "wav_dir": "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/segmented_wavs/gold",
    "en_txt_path": "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology/ACL.6060.dev.tagged.en-xx.en.txt",
    "glossary_path": "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_acl6060.json",
    
    "chunk_duration": 2.0,  # 2秒 chunk
    "chunk_overlap": 1.0,   # 1秒 overlap
    "target_beta": 1.0,     # F-Beta 的 Beta 值 (0.5=重准率, 1=平衡, 2=重召回)
    
    # Grid Search 配置（已废弃，使用 2D Grid Search）
    "grid_search_enabled": False,  # 是否启用旧版网格搜索
    "grid_threshold_min": 0.50,   # 搜索起始阈值
    "grid_threshold_max": 0.99,   # 搜索结束阈值
    "grid_step": 0.01,            # 搜索步长
    
    # 2D Grid Search 配置（Overlap × Threshold）
    "grid_2d_enabled": True,      # 是否启用二维网格搜索
    "grid_overlap_min": 1.0,      # 最小 overlap（秒）
    "grid_overlap_max": 1.9,      # 最大 overlap（秒）
    "grid_overlap_step": 0.1,     # overlap 步长（秒）
    "grid_threshold_values": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],  # threshold 取值
    "grid_target_beta": 2.0,      # F-Beta 的 Beta 值（2.0 = F2-Score，更重视 Recall）
    "arrival_step": 1.0,          # SST Agent 每次处理的新音频长度（秒），用于模拟流式延迟
    
    # RAG 配置
    "rag_enabled": True,  # 是否启用 RAG
    "rag_index_path": "/mnt/gemini/data2/jiaxuanluo/indices/qwen2_audio_term_index_acl6060.pkl",
    "rag_model_path": "/mnt/gemini/data2/jiaxuanluo/models/qwen2_audio_term_level_modal_v2_best.pt",  # TODO: 填写你的 RAG 模型路径
    "rag_base_model": "Qwen/Qwen2-Audio-7B-Instruct",
    "rag_device": "cuda:0",  # RAG 模型使用的设备
    "rag_top_k": 5,  # 检索 top-k 术语
    "rag_score_threshold": 0.0,  # 初始阈值设为 0，让所有结果都能被分析
    "rag_batch_size": 32,  # 批量推理的 batch size
}

# ================= 1. 数据加载与预处理 =================

def load_glossary_terms(json_path):
    """加载所有可能的术语列表（统一转换为小写）"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 提取所有 term 的 key (假设 key 就是英文术语)
    # 根据你的json结构, key是 "parsing", value里的 "term" 也是 "parsing"
    # 统一转换为小写
    terms = set(term.lower() for term in data.keys())
    return terms

def load_ground_truth(txt_path, glossary_terms):
    """
    解析文本文件，生成每句话的 Ground Truth Terms。
    假设 txt 文件每一行对应一个 sent_id，按顺序排列。
    文本中用 [] 标记的词就是 GT terms。
    例如: So we're going to be covering what [lexical] borrowing is, the [task] that we proposed...
    """
    ground_truths = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        # 使用正则提取所有 [term] 标记的词
        # 匹配模式: [xxx] 其中 xxx 是术语
        gt_in_sent = set()
        matches = re.findall(r'\[([^\]]+)\]', line)
        for term in matches:
            # 统一转换为小写
            term_lower = term.strip().lower()
            if term_lower:
                gt_in_sent.add(term_lower)
        ground_truths.append(gt_in_sent)
    return ground_truths

# ================= 2. 模拟 RAG 推理接口 =================

def get_terms_from_audio_chunk(audio_chunk, sr=16000, rag_retriever=None, rag_top_k=5):
    """
    使用 RAG 模型进行术语检索。
    输入：一个 audio chunk (numpy array)
    输出：一个 list of (term, score)
    """
    if rag_retriever is None or not rag_retriever.enabled:
        return []
    
    try:
        # 将 numpy array 转换为 torch tensor
        audio_tensor = torch.tensor(audio_chunk, dtype=torch.float32)
        
        if audio_tensor.numel() == 0:
            return []
        
        # 调用 RAG retriever 进行检索
        # retrieve 返回 List[Dict[str, str]]，包含 "term" 和 "translation"
        references = rag_retriever.retrieve(
            audio_tensor,
            top_k=rag_top_k,
            target_lang="zh",  # 可以根据需要调整
        )
        
        # 将结果转换为 (term, score) 格式
        # 注意: TermRAGRetriever.retrieve 只返回满足阈值的术语，没有直接返回 score
        # 我们需要访问内部的 confidence 数据
        # 这里我们使用一个 workaround: 重新调用底层逻辑获取分数
        
        # 为了获取 score，我们需要直接调用模型的 encode_audio 和 index.search
        with torch.no_grad():
            audio_inputs = [audio_tensor.numpy()]
            embedding = rag_retriever.model.encode_audio(audio_inputs)
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().float().numpy()
        
        # 搜索最相似的 top_k 个术语
        D, I = rag_retriever.index.search(embedding, rag_top_k)
        
        # 转换为 (term, score) 列表
        results = []
        seen_terms = set()
        for distance, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(rag_retriever.term_list):
                continue
            term_entry = rag_retriever.term_list[idx]
            if not isinstance(term_entry, dict):
                continue
            term = term_entry.get("term", "")
            if not term:
                continue
            # 统一转换为小写
            term_lower = term.lower()
            if term_lower in seen_terms:
                continue
            seen_terms.add(term_lower)
            
            # 将距离转换为 confidence score
            confidence = rag_retriever._distance_to_confidence(float(distance))
            results.append((term_lower, confidence))
        
        return results
        
    except Exception as e:
        print(f"Error in RAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_terms_from_batch_audio(audio_chunks_list, rag_retriever, rag_top_k=5):
    """
    批处理版本的术语检索。
    输入：List of audio chunks [numpy array, numpy array, ...]
    输出：List of List of (term, score) —— 对应每个 chunk 的结果
    """
    if rag_retriever is None or not rag_retriever.enabled:
        return [[] for _ in audio_chunks_list]
    
    batch_size = len(audio_chunks_list)
    if batch_size == 0:
        return []
    
    try:
        # 1. GPU 推理 (Batch Encoding)
        # RAG 模型内部通常支持 list input。
        # 如果 audio_chunks_list 是 list of numpy，直接传进去
        with torch.no_grad():
            # 注意：确保 encode_audio 能处理 list，或者手动 stack 成 tensor
            # 假设 rag_retriever.model.encode_audio 接受 list[np.ndarray] 并处理 padding
            embeddings = rag_retriever.model.encode_audio(audio_chunks_list)
        
        # 转换 embeddings 为 numpy (Batch_Size, Dim)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().float().numpy()
        
        # 2. FAISS 搜索 (Batch Search)
        # embedding shape: (B, D) -> search 结果 D, I shape: (B, top_k)
        D_batch, I_batch = rag_retriever.index.search(embeddings, rag_top_k)
        
        # 3. 解析结果
        batch_results = []
        
        # 遍历 Batch 中的每一个样本
        for i in range(batch_size):
            results = []
            seen_terms = set()
            
            # 遍历该样本的 Top-K 结果
            for distance, idx in zip(D_batch[i], I_batch[i]):
                if idx < 0 or idx >= len(rag_retriever.term_list):
                    continue
                
                term_entry = rag_retriever.term_list[idx]
                if not isinstance(term_entry, dict):
                    continue
                    
                term = term_entry.get("term", "")
                if not term:
                    continue
                    
                term_lower = term.lower()
                if term_lower in seen_terms:
                    continue
                seen_terms.add(term_lower)
                
                # 转换分数
                confidence = rag_retriever._distance_to_confidence(float(distance))
                results.append((term_lower, confidence))
            
            batch_results.append(results)
            
        return batch_results
        
    except Exception as e:
        print(f"Error in Batch RAG retrieval: {e}")
        import traceback
        traceback.print_exc()
        return [[] for _ in audio_chunks_list]

# ================= 3. 滑动窗口与推理主循环 =================

def run_inference_on_dataset(wav_dir, ground_truths, rag_retriever=None, rag_top_k=5, batch_size=32):
    """
    对文件夹下的 wav 进行滑动窗口推理，并聚合分数。
    使用批量处理提高效率。
    
    Args:
        wav_dir: 音频文件目录
        ground_truths: Ground Truth 列表
        rag_retriever: RAG 检索器
        rag_top_k: 检索 top-k
        batch_size: 批处理大小（每次处理多少个 chunk）
    """
    # 获取排序后的 wav 文件列表，确保和 txt 行号对应
    # 按照文件名中的数字排序，而不是字符串排序
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    def extract_number(filename):
        """从文件名中提取数字，例如 sent_123.wav -> 123"""
        basename = os.path.basename(filename)
        # 使用正则提取数字部分
        match = re.search(r'sent_(\d+)\.wav', basename)
        if match:
            return int(match.group(1))
        return 0  # 如果没有匹配到，返回 0
    
    wav_files = sorted(wav_files, key=extract_number)
    
    # 安全检查
    if len(wav_files) != len(ground_truths):
        error_msg = f"Wav文件数量 ({len(wav_files)}) 与 文本行数 ({len(ground_truths)}) 不一致！"
        print(f"错误: {error_msg}")
        raise Exception(error_msg)

    all_predictions = []  # 存储每个 segment 的预测结果字典 {term: max_score}
    
    print(f"正在进行滑动窗口推理（批量模式，batch_size={batch_size}）...")
    
    # 第一步：收集所有 chunks 和对应的 segment 索引
    all_chunks = []
    chunk_to_segment = []  # 记录每个 chunk 属于哪个 segment
    
    print("第 1 步：加载音频并切分 chunks...")
    for i, wav_path in enumerate(tqdm(wav_files, desc="加载音频")):
        try:
            # 加载音频
            y, sr = librosa.load(wav_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # 如果加载失败，记录一个空的预测，但不添加 chunks
            continue

        # 滑动窗口逻辑
        step = int((CONFIG["chunk_duration"] - CONFIG["chunk_overlap"]) * sr)
        window_size = int(CONFIG["chunk_duration"] * sr)
        
        # 如果音频短于窗口，直接处理整个音频
        if len(y) <= window_size:
            chunks = [y]
        else:
            # 生成 chunks
            chunks = [y[j:j+window_size] for j in range(0, len(y) - window_size + 1, step)]
        
        # 收集 chunks 和它们的归属
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_segment.append(i)
    
    print(f"总共收集了 {len(all_chunks)} 个 chunks，来自 {len(wav_files)} 个音频文件")
    
    # 第二步：批量推理
    print(f"第 2 步：批量 RAG 推理...")
    all_chunk_results = []
    
    for batch_start in tqdm(range(0, len(all_chunks), batch_size), desc="批量推理"):
        batch_end = min(batch_start + batch_size, len(all_chunks))
        batch_chunks = all_chunks[batch_start:batch_end]
        
        # 批量调用 RAG
        batch_results = get_terms_from_batch_audio(batch_chunks, rag_retriever, rag_top_k)
        all_chunk_results.extend(batch_results)
    
    # 第三步：聚合结果
    print("第 3 步：聚合结果...")
    segment_scores = [dict() for _ in range(len(wav_files))]  # 每个 segment 一个字典
    
    for chunk_idx, chunk_result in enumerate(all_chunk_results):
        segment_idx = chunk_to_segment[chunk_idx]
        
        # 将该 chunk 的结果添加到对应的 segment
        for term, score in chunk_result:
            term_lower = term.lower()
            if term_lower not in segment_scores[segment_idx]:
                segment_scores[segment_idx][term_lower] = []
            segment_scores[segment_idx][term_lower].append(score)
    
    # 第四步：Max Pooling 聚合每个 segment 的分数
    print("第 4 步：Max Pooling 聚合...")
    for segment_idx in range(len(wav_files)):
        aggregated_preds = {}
        for term, scores in segment_scores[segment_idx].items():
            aggregated_preds[term] = max(scores)
        all_predictions.append(aggregated_preds)
    
    print(f"完成！处理了 {len(all_predictions)} 个音频片段")
    return all_predictions

# ================= 4. 分析与可视化 =================

def analyze_and_plot(predictions, ground_truths, target_beta=1.0, output_dir=None, timestamp=None):
    """
    分析预测结果并生成可视化图表
    
    Args:
        predictions: 预测结果
        ground_truths: Ground Truth
        target_beta: F-beta score 的 beta 值
        output_dir: 输出目录，如果为 None 则使用当前目录
        timestamp: 时间戳字符串，用于文件命名
    """
    positive_scores = [] # GT 中存在的词的得分
    negative_scores = [] # GT 中不存在的词的得分 (误报)

    # 准备用于计算 PR 曲线的数据
    y_true_flat = []
    y_scores_flat = []
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成时间戳
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("正在计算统计数据...")
    for pred_dict, gt_set in zip(predictions, ground_truths):
        # 1. 处理 Positive Samples (在 GT 里的词)
        # 注意：这里只统计模型检索出来的词。如果模型没检出 GT 里的词，那是 FN，没有分数，无法画在密度图里。
        for term, score in pred_dict.items():
            if term in gt_set:
                positive_scores.append(score)
                y_true_flat.append(1)
                y_scores_flat.append(score)
            else:
                negative_scores.append(score)
                y_true_flat.append(0)
                y_scores_flat.append(score)
        
        # 为了 PR 曲线的完整性，我们需要把那些漏检的 (FN) 也加进去，score 设为 0
        # (可选步骤，取决于你是否想让 PR 曲线反映 Recall 的全貌)
        # for gt_term in gt_set:
        #     if gt_term not in pred_dict:
        #         y_true_flat.append(1)
        #         y_scores_flat.append(0.0)

    # --- 图表 A: 概率密度分布 (Density Plot) ---
    plt.figure(figsize=(10, 5))
    plt.hist(positive_scores, bins=50, alpha=0.6, label='Positive (Matches)', density=True, color='green')
    plt.hist(negative_scores, bins=50, alpha=0.6, label='Negative (FP/Noise)', density=True, color='red')
    plt.title('Score Distribution: Positive vs Negative Samples')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    score_dist_path = os.path.join(output_dir, f'score_distribution_{timestamp}.png')
    plt.savefig(score_dist_path)
    plt.close()
    print(f"分布图已保存至 {score_dist_path}")

    # --- 图表 B: PR 曲线与 F-Beta 阈值搜索 ---
    if not y_true_flat:
        print("没有收集到有效数据，请检查模型输出或路径。")
        return

    precision, recall, thresholds = precision_recall_curve(y_true_flat, y_scores_flat)
    
    # 计算 F-beta
    numerator = (1 + target_beta**2) * (precision * recall)
    denominator = (target_beta**2 * precision) + recall
    # 处理除零
    f_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    # 找到最佳点
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx]
    best_f_score = f_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]

    print(f"\n=== 最佳阈值分析 (Beta={target_beta}) ===")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Best F{target_beta}-Score: {best_f_score:.4f}")
    print(f"Precision at Best: {best_precision:.4f}")
    print(f"Recall at Best:    {best_recall:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='PR Curve')
    plt.scatter(best_recall, best_precision, marker='o', color='red', label=f'Best Thresh={best_threshold:.2f}')
    plt.title(f'Precision-Recall Curve (Target F{target_beta})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    pr_curve_path = os.path.join(output_dir, f'pr_curve_{timestamp}.png')
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"PR曲线已保存至 {pr_curve_path}")

    return best_threshold

def grid_search_threshold(predictions, ground_truths, threshold_range=(0.50, 0.99), step=0.01, output_dir=None, timestamp=None):
    """
    显式网格搜索最佳阈值
    
    Args:
        predictions: 预测结果 List[Dict[term, score]]
        ground_truths: Ground Truth List[Set[term]]
        threshold_range: 阈值搜索范围 (min, max)
        step: 搜索步长
        output_dir: 输出目录，用于保存结果 CSV
        timestamp: 时间戳字符串，用于文件命名
    
    Returns:
        best_threshold: 最佳阈值
        results: 所有阈值的结果列表
    """
    # 生成时间戳
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\n" + "="*60)
    print("开始网格搜索（Grid Search）...")
    print(f"阈值范围: {threshold_range[0]:.2f} - {threshold_range[1]:.2f}")
    print(f"搜索步长: {step:.2f}")
    print("="*60)
    
    results = []
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    
    for threshold in tqdm(thresholds, desc="Grid Search"):
        # 对每个阈值，过滤预测结果并计算指标
        tp = 0  # True Positive
        fp = 0  # False Positive
        fn = 0  # False Negative
        
        for pred_dict, gt_set in zip(predictions, ground_truths):
            # 过滤：只保留 score >= threshold 的预测
            filtered_preds = set(term for term, score in pred_dict.items() if score >= threshold)
            
            # 计算 TP, FP, FN
            tp += len(filtered_preds & gt_set)  # 预测正确的
            fp += len(filtered_preds - gt_set)  # 误报
            fn += len(gt_set - filtered_preds)  # 漏检
        
        # 计算指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
        })
    
    # 找到最佳阈值（基于 F1）
    best_result = max(results, key=lambda x: x['f1'])
    
    print("\n" + "="*60)
    print("网格搜索完成！")
    print("="*60)
    print(f"最佳阈值: {best_result['threshold']:.4f}")
    print(f"最佳 F1-Score: {best_result['f1']:.4f}")
    print(f"对应 Precision: {best_result['precision']:.4f}")
    print(f"对应 Recall: {best_result['recall']:.4f}")
    print(f"TP: {best_result['tp']}, FP: {best_result['fp']}, FN: {best_result['fn']}")
    print("="*60)
    
    # 保存详细结果到 CSV
    if output_dir:
        import csv
        csv_path = os.path.join(output_dir, f'grid_search_results_{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['threshold', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n详细结果已保存至: {csv_path}")
        
        # 绘制 F1 vs Threshold 曲线
        plt.figure(figsize=(10, 6))
        thresholds_list = [r['threshold'] for r in results]
        f1_list = [r['f1'] for r in results]
        precision_list = [r['precision'] for r in results]
        recall_list = [r['recall'] for r in results]
        
        plt.plot(thresholds_list, f1_list, label='F1-Score', linewidth=2, color='blue')
        plt.plot(thresholds_list, precision_list, label='Precision', linewidth=1.5, color='green', alpha=0.7)
        plt.plot(thresholds_list, recall_list, label='Recall', linewidth=1.5, color='red', alpha=0.7)
        
        # 标记最佳点
        plt.scatter([best_result['threshold']], [best_result['f1']], 
                   color='blue', s=100, zorder=5, marker='*',
                   label=f"Best (T={best_result['threshold']:.3f}, F1={best_result['f1']:.3f})")
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Grid Search: Threshold vs Metrics', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim(threshold_range)
        plt.ylim(0, 1)
        
        curve_path = os.path.join(output_dir, f'grid_search_curve_{timestamp}.png')
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"曲线图已保存至: {curve_path}")
    
    return best_result['threshold'], results

def measure_streaming_latency(rag_retriever, overlap, chunk_duration=2.0, arrival_step=1.0, rag_top_k=5):
    """
    模拟流式场景下的单步推理延迟。
    
    Args:
        rag_retriever: RAG 模型实例
        overlap: 当前 overlap 设置
        chunk_duration: 窗口大小 (2.0s)
        arrival_step: SST Agent 每次接收的新音频长度 (通常是 0.3s ~ 1.0s)
        rag_top_k: 检索 top-k
    
    Returns:
        latency_ms: 处理这一批增量 chunks 所需的毫秒数
        micro_batch_size: 这一步产生的 chunks 数量
    """
    if rag_retriever is None or not rag_retriever.enabled:
        return 0.0, 0
    
    import time
    
    # 1. 计算步长（stride）
    stride = chunk_duration - overlap
    if stride <= 0.001:
        stride = 0.001  # 避免除零
    
    # 2. 计算 Micro-Batch Size
    # 逻辑：在 arrival_step 这么长的时间里，滑动窗口滑了多少次？
    # 例如：新来 1.0s 音频。
    # - Stride=1.0s (Overlap 1.0) -> num_chunks = 1
    # - Stride=0.1s (Overlap 1.9) -> num_chunks = 10
    num_chunks = int(np.ceil(arrival_step / stride))
    num_chunks = max(1, num_chunks)  # 至少处理 1 个
    
    # 3. 构造 Dummy Batch
    # Qwen-Audio 输入通常是 raw waveform (float array)
    # 假设采样率 16000，长度 2秒
    sample_len = int(chunk_duration * 16000)
    dummy_audio_batch = [
        np.random.uniform(-0.5, 0.5, sample_len).astype(np.float32) 
        for _ in range(num_chunks)
    ]
    
    # 4. 测量耗时 (GPU Warmup + Timing)
    try:
        # Warmup (跑一次不计时，激活 CUDA kernel)
        with torch.no_grad():
            _ = rag_retriever.model.encode_audio(dummy_audio_batch[:1])
        
        # 正式计时
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        with torch.no_grad():
            # A. Encode
            embeddings = rag_retriever.model.encode_audio(dummy_audio_batch)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().float().numpy()
            
            # B. Search (CPU FAISS)
            _, _ = rag_retriever.index.search(embeddings, rag_top_k)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待 GPU 完成
        
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        
    except Exception as e:
        print(f"Error measuring latency for overlap={overlap}: {e}")
        latency_ms = 0.0
    
    return latency_ms, num_chunks

def grid_search_2d_overlap_threshold(
    wav_dir, 
    ground_truths, 
    rag_retriever, 
    rag_top_k=5,
    batch_size=32,
    overlap_values=None,
    threshold_values=None,
    target_beta=2.0,
    output_dir=None,
    timestamp=None
):
    """
    二维网格搜索：Overlap × Threshold
    
    Args:
        wav_dir: 音频文件目录
        ground_truths: Ground Truth 列表
        rag_retriever: RAG 检索器
        rag_top_k: 检索 top-k
        batch_size: 批处理大小
        overlap_values: Overlap 取值列表（秒）
        threshold_values: Threshold 取值列表
        target_beta: F-Beta 的 Beta 值（2.0 = F2-Score）
        output_dir: 输出目录
        timestamp: 时间戳
    
    Returns:
        results_dict: 包含所有结果的字典
    """
    import time
    import seaborn as sns
    
    if overlap_values is None:
        overlap_values = np.arange(1.0, 2.0, 0.1)
    if threshold_values is None:
        threshold_values = np.arange(0.50, 1.00, 0.05)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*80)
    print("开始二维网格搜索：Overlap × Threshold")
    print("="*80)
    print(f"Overlap 范围: {overlap_values[0]:.1f}s - {overlap_values[-1]:.1f}s ({len(overlap_values)} 个点)")
    print(f"Threshold 范围: {threshold_values[0]:.2f} - {threshold_values[-1]:.2f} ({len(threshold_values)} 个点)")
    print(f"总搜索点数: {len(overlap_values)} × {len(threshold_values)} = {len(overlap_values) * len(threshold_values)}")
    print(f"目标指标: F{target_beta}-Score")
    print("="*80)
    
    # 存储结果
    # overlap_results[overlap] = {
    #     "predictions": [...],
    #     "latency_ms": float,
    # }
    overlap_results = {}
    
    # 原始配置
    original_chunk_duration = CONFIG["chunk_duration"]
    original_chunk_overlap = CONFIG["chunk_overlap"]
    
    # 第一步：对每个 overlap 运行推理并测量【流式模拟延迟】
    print("\n第 1 步：运行推理并测量【流式模拟延迟】...")
    print(f"假设 SST Agent 每次处理 {CONFIG.get('arrival_step', 1.0)}s 新音频")
    print("="*80)
    
    for overlap in tqdm(overlap_values, desc="Overlap 分析"):
        # A. 测量真实的流式延迟 (Micro-Batch Latency)
        # 假设你的 SST Agent 每次读入 arrival_step 秒新音频 (SimulEval 常见设置)
        arrival_step = CONFIG.get("arrival_step", 1.0)  # 默认 1.0 秒
        streaming_latency_ms, micro_batch_size = measure_streaming_latency(
            rag_retriever, 
            overlap, 
            chunk_duration=CONFIG["chunk_duration"], 
            arrival_step=arrival_step,
            rag_top_k=rag_top_k
        )
        
        # B. 运行全量离线推理以计算 Recall/Precision (这里可以用大 Batch 加速)
        CONFIG["chunk_overlap"] = float(overlap)
        predictions = run_inference_on_dataset(
            wav_dir,
            ground_truths,
            rag_retriever=rag_retriever,
            rag_top_k=rag_top_k,
            batch_size=64  # 这里的 batch_size 可以很大，只为了快速跑出 F2 分数
        )
        
        overlap_results[overlap] = {
            "predictions": predictions,
            "latency_ms": streaming_latency_ms,  # 记录的是流式延迟！
            "micro_batch_size": micro_batch_size,
        }
        
        # 计算 stride 用于打印日志
        stride = CONFIG["chunk_duration"] - overlap
        print(f"  Overlap {overlap:.1f}s (Stride={stride:.1f}s, Micro-Batch={micro_batch_size}): "
              f"单步延迟 {streaming_latency_ms:.2f} ms")
    
    # 恢复原始配置
    CONFIG["chunk_duration"] = original_chunk_duration
    CONFIG["chunk_overlap"] = original_chunk_overlap
    
    # 第二步：对每个 (overlap, threshold) 组合计算 F2-Score
    print("\n第 2 步：计算 F2-Score 矩阵...")
    f2_matrix = np.zeros((len(threshold_values), len(overlap_values)))
    latency_list = []
    micro_batch_sizes = []
    
    for j, overlap in enumerate(tqdm(overlap_values, desc="计算 F2")):
        predictions = overlap_results[overlap]["predictions"]
        latency_ms = overlap_results[overlap]["latency_ms"]
        micro_batch_size = overlap_results[overlap]["micro_batch_size"]
        latency_list.append(latency_ms)
        micro_batch_sizes.append(micro_batch_size)
        
        for i, threshold in enumerate(threshold_values):
            # 对每个阈值，过滤预测结果并计算指标
            tp = 0  # True Positive
            fp = 0  # False Positive
            fn = 0  # False Negative
            
            for pred_dict, gt_set in zip(predictions, ground_truths):
                # 过滤：只保留 score >= threshold 的预测
                filtered_preds = set(term for term, score in pred_dict.items() if score >= threshold)
                
                # 计算 TP, FP, FN
                tp += len(filtered_preds & gt_set)
                fp += len(filtered_preds - gt_set)
                fn += len(gt_set - filtered_preds)
            
            # 计算 F-Beta Score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall > 0:
                f_beta = (1 + target_beta**2) * (precision * recall) / (target_beta**2 * precision + recall)
            else:
                f_beta = 0.0
            
            f2_matrix[i, j] = f_beta
    
    # 找到最佳点
    best_i, best_j = np.unravel_index(np.argmax(f2_matrix), f2_matrix.shape)
    best_overlap = overlap_values[best_j]
    best_threshold = threshold_values[best_i]
    best_f2 = f2_matrix[best_i, best_j]
    best_latency = latency_list[best_j]
    best_micro_batch = micro_batch_sizes[best_j]
    best_stride = CONFIG["chunk_duration"] - best_overlap
    
    print("\n" + "="*80)
    print("二维网格搜索完成！")
    print("="*80)
    print(f"最佳配置:")
    print(f"  Overlap: {best_overlap:.1f}s (Stride: {best_stride:.1f}s)")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  F{target_beta}-Score: {best_f2:.4f}")
    print(f"  Micro-Batch Size: {best_micro_batch} chunks")
    print(f"  单步推理延迟: {best_latency:.2f} ms")
    print(f"  → 这是真实流式场景下每 {CONFIG.get('arrival_step', 1.0)}s 新音频的处理延迟")
    print("="*80)
    
    # 第三步：绘制热力图
    if output_dir:
        print("\n第 3 步：绘制热力图...")
        
        # 创建 X 轴标签：显示 overlap、micro-batch size 和对应的耗时
        x_labels = [f"{overlap:.1f}s\nBatch={mb}\n{latency:.0f}ms" 
                   for overlap, mb, latency in zip(overlap_values, micro_batch_sizes, latency_list)]
        
        # 创建热力图
        plt.figure(figsize=(14, 8))
        
        # 使用 seaborn 绘制热力图
        ax = sns.heatmap(
            f2_matrix,
            xticklabels=x_labels,
            yticklabels=[f"{t:.2f}" for t in threshold_values],
            annot=True,  # 显示数值
            fmt='.3f',   # 数值格式
            cmap='RdYlGn',  # 颜色映射：红-黄-绿
            vmin=0,
            vmax=1,
            cbar_kws={'label': f'F{target_beta}-Score'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        # 标记最佳点
        ax.add_patch(plt.Rectangle(
            (best_j, best_i), 1, 1,
            fill=False, edgecolor='blue', lw=3
        ))
        
        plt.title(
            f'2D Grid Search: Overlap × Threshold (F{target_beta}-Score)\n'
            f'Best: Overlap={best_overlap:.1f}s, Threshold={best_threshold:.2f}, '
            f'F{target_beta}={best_f2:.4f}, Batch={best_micro_batch}, Latency={best_latency:.0f}ms\n'
            f'(流式场景：每 {CONFIG.get("arrival_step", 1.0)}s 新音频的处理延迟)',
            fontsize=13, fontweight='bold', pad=20
        )
        plt.xlabel('Overlap (Micro-Batch Size, Latency)', fontsize=12, fontweight='bold')
        plt.ylabel('Threshold', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        heatmap_path = os.path.join(output_dir, f'grid_search_2d_heatmap_{timestamp}.png')
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"热力图已保存至: {heatmap_path}")
        
        # 保存详细数据到 CSV
        import csv
        csv_path = os.path.join(output_dir, f'grid_search_2d_results_{timestamp}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            header = ['threshold'] + [f'overlap_{o:.1f}s_batch{mb}_({l:.0f}ms)' 
                                     for o, mb, l in zip(overlap_values, micro_batch_sizes, latency_list)]
            writer.writerow(header)
            # 写入数据
            for i, threshold in enumerate(threshold_values):
                row = [f'{threshold:.2f}'] + [f'{f2_matrix[i, j]:.4f}' 
                                              for j in range(len(overlap_values))]
                writer.writerow(row)
        print(f"详细数据已保存至: {csv_path}")
        
        # 保存延迟统计信息
        stats_csv_path = os.path.join(output_dir, f'grid_search_2d_latency_stats_{timestamp}.csv')
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['overlap', 'stride', 'micro_batch_size', 'latency_ms', 'latency_per_chunk_ms'])
            for overlap, mb, latency in zip(overlap_values, micro_batch_sizes, latency_list):
                stride = CONFIG["chunk_duration"] - overlap
                latency_per_chunk = latency / mb if mb > 0 else 0
                writer.writerow([f'{overlap:.1f}', f'{stride:.1f}', mb, f'{latency:.2f}', f'{latency_per_chunk:.2f}'])
        print(f"延迟统计已保存至: {stats_csv_path}")
    
    return {
        'overlap_values': overlap_values,
        'threshold_values': threshold_values,
        'f2_matrix': f2_matrix,
        'latency_list': latency_list,
        'micro_batch_sizes': micro_batch_sizes,
        'best_overlap': best_overlap,
        'best_threshold': best_threshold,
        'best_f2': best_f2,
        'best_latency': best_latency,
        'best_micro_batch': best_micro_batch,
        'best_stride': best_stride,
        'arrival_step': CONFIG.get('arrival_step', 1.0),
    }

# ================= 主程序入口 =================

if __name__ == "__main__":
    # 0. 初始化 RAG Retriever (如果启用)
    rag_retriever = None
    if CONFIG.get("rag_enabled", False):
        print("正在初始化 RAG Retriever...")
        try:
            # 动态导入 TermRAGRetriever（假设它在同一个项目中）
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))
            from agents.infinisst_omni_vllm_rag import TermRAGRetriever
            
            rag_retriever = TermRAGRetriever(
                index_path=CONFIG.get("rag_index_path"),
                model_path=CONFIG.get("rag_model_path"),
                base_model_name=CONFIG.get("rag_base_model", "Qwen/Qwen2-Audio-7B-Instruct"),
                device=CONFIG.get("rag_device", "cuda:0"),
                top_k=CONFIG.get("rag_top_k", 10),
                target_lang="zh",
                score_threshold=CONFIG.get("rag_score_threshold", 0.0),
            )
            
            if rag_retriever and rag_retriever.enabled:
                print(f"✅ RAG Retriever 初始化成功")
                print(f"   - Index: {CONFIG.get('rag_index_path')}")
                print(f"   - Model: {CONFIG.get('rag_model_path')}")
                print(f"   - Device: {CONFIG.get('rag_device')}")
                print(f"   - Top-K: {CONFIG.get('rag_top_k')}")
                print(f"   - Batch Size: {CONFIG.get('rag_batch_size')}")
                print(f"   - Score Threshold: {CONFIG.get('rag_score_threshold')}")
            else:
                print("⚠️  RAG Retriever 初始化失败，将跳过 RAG 推理")
                rag_retriever = None
        except Exception as e:
            print(f"❌ RAG Retriever 初始化出错: {e}")
            import traceback
            traceback.print_exc()
            rag_retriever = None
    else:
        print("RAG Retriever 未启用")
    
    # 1. 加载 Glossary
    glossary_terms = load_glossary_terms(CONFIG["glossary_path"])
    print(f"加载了 {len(glossary_terms)} 个术语 (已转为小写)")
    
    # 2. 准备 Ground Truth
    ground_truths = load_ground_truth(CONFIG["en_txt_path"], glossary_terms)
    print(f"加载了 {len(ground_truths)} 个句子的 Ground Truth")
    
    # 3. 运行推理（批量模式）
    # 注意: 如果数据量大，建议先跑一次把 all_predictions 存成 json，下次直接加载分析
    predictions = run_inference_on_dataset(
        CONFIG["wav_dir"], 
        ground_truths,
        rag_retriever=rag_retriever,
        rag_top_k=CONFIG.get("rag_top_k", 10),
        batch_size=CONFIG.get("rag_batch_size", 32)
    )
    
    # 4. 分析与画图
    output_dir = "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成统一的时间戳，用于本次运行的所有文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n本次运行时间戳: {timestamp}")
    print(f"所有结果将保存到: {output_dir}")
    
    best_threshold = analyze_and_plot(
        predictions, 
        ground_truths, 
        target_beta=CONFIG["target_beta"], 
        output_dir=output_dir,
        timestamp=timestamp
    )
    
    # 4.5 网格搜索最佳阈值（旧版，可选）
    if CONFIG.get("grid_search_enabled", False):
        print("\n" + "="*80)
        print("开始网格搜索（旧版）...")
        print("="*80)
        grid_best_threshold, grid_results = grid_search_threshold(
            predictions, 
            ground_truths, 
            threshold_range=(CONFIG.get("grid_threshold_min", 0.50), 
                           CONFIG.get("grid_threshold_max", 0.99)), 
            step=CONFIG.get("grid_step", 0.01),
            output_dir=output_dir,
            timestamp=timestamp
        )
    
    # 4.6 二维网格搜索：Overlap × Threshold（新版）
    if CONFIG.get("grid_2d_enabled", False):
        print("\n" + "="*80)
        print("开始二维网格搜索：Overlap × Threshold")
        print("="*80)
        
        # 准备 overlap 和 threshold 取值
        overlap_min = CONFIG.get("grid_overlap_min", 1.0)
        overlap_max = CONFIG.get("grid_overlap_max", 1.9)
        overlap_step = CONFIG.get("grid_overlap_step", 0.1)
        overlap_values = np.arange(overlap_min, overlap_max + overlap_step/2, overlap_step)
        
        threshold_values = CONFIG.get("grid_threshold_values", 
                                     [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        
        target_beta = CONFIG.get("grid_target_beta", 2.0)
        
        grid_2d_results = grid_search_2d_overlap_threshold(
            CONFIG["wav_dir"],
            ground_truths,
            rag_retriever=rag_retriever,
            rag_top_k=CONFIG.get("rag_top_k", 5),
            batch_size=CONFIG.get("rag_batch_size", 32),
            overlap_values=overlap_values,
            threshold_values=threshold_values,
            target_beta=target_beta,
            output_dir=output_dir,
            timestamp=timestamp
        )
    
    # 5. (可选) 相对阈值建议
    # 计算所有 negative scores 的统计量
    all_neg_scores = []
    for pred_dict, gt_set in zip(predictions, ground_truths):
        for term, score in pred_dict.items():
            if term not in gt_set:
                all_neg_scores.append(score)
    
    if all_neg_scores:
        mu = np.mean(all_neg_scores)
        sigma = np.std(all_neg_scores)
        print("\n=== 相对阈值参考 ===")
        print(f"Noise Mean (mu): {mu:.4f}")
        print(f"Noise Std (sigma): {sigma:.4f}")
        print(f"Suggested Threshold (Mean + 2Std): {mu + 2*sigma:.4f}")
        print(f"Suggested Threshold (Mean + 3Std): {mu + 3*sigma:.4f}")