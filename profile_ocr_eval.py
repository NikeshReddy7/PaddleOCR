#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to compare PaddleOCR evaluation performance:
- CPU-only (pure Python Shapely)
- GPU-optimized (Paddle tensor operations)

Usage:
    python profile_ocr_eval.py --config configs/det/ch_PP-OCRv3_det.yml \
        --o Global.pretrained_model=path/to/model
"""

import os
import sys
import time
import argparse

# Add PaddleOCR to path
ocr_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ocr_path)

import paddle
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
import tools.program as program
from tools.program import preprocess

from ppocr.metrics.eval_det_iou import DetectionIoUEvaluator
from ppocr.metrics.eval_det_iou_gpu import DetectionIoUEvaluatorGPU


def benchmark_evaluators(evaluator_name, evaluator, num_samples=50):
    """
    Benchmark an evaluator with synthetic test data
    """
    import random
    import numpy as np
    
    print(f"\n{'='*70}")
    print(f"🔬 Benchmarking: {evaluator_name}")
    print(f"{'='*70}")
    
    # Generate synthetic ground truth and predictions
    results = []
    start_time = time.time()
    
    for sample_idx in range(num_samples):
        # Generate random polygons (0-30 boxes per image)
        num_gt = random.randint(5, 25)
        num_pred = random.randint(5, 25)
        
        gt_list = []
        for _ in range(num_gt):
            # Random 4-point polygon
            x1, y1 = random.randint(0, 500), random.randint(0, 500)
            x2, y2 = random.randint(x1+10, 800), random.randint(y1+10, 800)
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            gt_list.append({
                "points": points,
                "text": "",
                "ignore": random.random() < 0.1  # 10% ignored
            })
        
        pred_list = []
        for _ in range(num_pred):
            x1, y1 = random.randint(0, 500), random.randint(0, 500)
            x2, y2 = random.randint(x1+10, 800), random.randint(y1+10, 800)
            # Add some noise
            x1 += random.randint(-20, 20)
            y1 += random.randint(-20, 20)
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            pred_list.append({
                "points": points,
                "text": ""
            })
        
        # Evaluate
        result = evaluator.evaluate_image(gt_list, pred_list)
        results.append(result)
    
    # Combine results
    metrics = evaluator.combine_results(results)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\n📊 Results ({num_samples} samples):")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Hmean:     {metrics['hmean']:.4f}")
    print(f"\n⏱️  Performance:")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Per image:  {elapsed_time/num_samples*1000:.2f}ms")
    print(f"  Throughput: {num_samples/elapsed_time:.1f} images/sec")
    
    return {
        'name': evaluator_name,
        'time': elapsed_time,
        'per_image_ms': elapsed_time/num_samples*1000,
        'throughput': num_samples/elapsed_time,
        'metrics': metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to benchmark")
    parser.add_argument("--show_speedup", action="store_true", help="Show speedup comparison")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 PaddleOCR Evaluation Profiler")
    print("="*70)
    print(f"GPU Available: {paddle.device.is_compiled_with_cuda()}")
    print(f"GPU Device: {paddle.device.get_device()}")
    
    # Benchmark both evaluators
    cpu_result = benchmark_evaluators(
        "CPU (Python Shapely)",
        DetectionIoUEvaluator(),
        num_samples=args.num_samples
    )
    
    gpu_result = benchmark_evaluators(
        "GPU (Paddle Tensors)",
        DetectionIoUEvaluatorGPU(),
        num_samples=args.num_samples
    )
    
    # Comparison
    if args.show_speedup:
        print(f"\n{'='*70}")
        print("⚡ SPEEDUP COMPARISON")
        print(f"{'='*70}")
        speedup = cpu_result['time'] / gpu_result['time']
        print(f"\n📈 GPU is {speedup:.1f}x faster than CPU")
        print(f"  CPU time per image: {cpu_result['per_image_ms']:.2f}ms")
        print(f"  GPU time per image: {gpu_result['per_image_ms']:.2f}ms")
        print(f"  Time saved per image: {cpu_result['per_image_ms'] - gpu_result['per_image_ms']:.2f}ms")
        
        # For 200 images (typical eval set)
        num_images = 200
        cpu_total = cpu_result['per_image_ms'] * num_images / 1000
        gpu_total = gpu_result['per_image_ms'] * num_images / 1000
        print(f"\n🕐 For {num_images} image evaluation:")
        print(f"  CPU time: {cpu_total:.1f}s")
        print(f"  GPU time: {gpu_total:.1f}s")
        print(f"  Total time saved: {cpu_total - gpu_total:.1f}s ({100*(cpu_total-gpu_total)/cpu_total:.1f}%)")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
