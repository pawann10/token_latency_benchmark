#!/usr/bin/env python3
"""
benchmark.py

Simple script to load the benchmark CSV (produced by the Colab notebook),
print a short summary, and save latency / throughput plots to results/.
Usage:
    python benchmark.py --csv results/llama_benchmark_results.csv --out results/latency_plot.png
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

def print_summary(df: pd.DataFrame):
    print("\nBenchmark summary:")
    print(df.to_string(index=False))

def plot_and_save(df: pd.DataFrame, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Latency plot
    plt.figure(figsize=(7,4))
    plt.plot(df['batch_size'], df['mean_latency_s_per_token'], marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('Batch size (log2)')
    plt.ylabel('Mean latency (s per token)')
    plt.title('Token Generation Latency vs Batch Size')
    plt.grid(True)
    latency_path = out_prefix.with_name(out_prefix.stem + "_latency.png")
    plt.savefig(latency_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved latency plot to", latency_path)

    # Throughput plot
    plt.figure(figsize=(7,4))
    plt.plot(df['batch_size'], df['mean_throughput_tokens_per_s'], marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('Batch size (log2)')
    plt.ylabel('Throughput (tokens/s)')
    plt.title('Throughput vs Batch Size')
    plt.grid(True)
    throughput_path = out_prefix.with_name(out_prefix.stem + "_throughput.png")
    plt.savefig(throughput_path, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved throughput plot to", throughput_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("results/llama_benchmark_results.csv"),
                   help="Path to benchmark CSV (output from Colab).")
    p.add_argument("--out", type=Path, default=Path("results/latency_plot.png"),
                   help="Prefix for output plots (will produce _latency.png and _throughput.png).")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        print("Make sure you moved the downloaded CSV into the project folder under results/")
        return

    df = load_results(args.csv)
    print_summary(df)
    plot_and_save(df, args.out)

if __name__ == "__main__":
    main()
