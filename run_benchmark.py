import sys
import os

# Ensure we can import my_mteb
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from my_mteb.models import SentenceTransformerModel
from my_mteb.tasks import (
    STSBTask, Banking77Task, TwentyNewsgroupsTask, QQPTask,
    SciDocsRetrievalTask, AskUbuntuTask, SimulatedSummarizationTask, TatoebaTask
)

def main():
    print("=== MTEB (Complete) Benchmark Implementation ===")
    
    # 1. Load Model
    # Using a lightweight, high-performance model for demonstration
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading Model: {model_name} ...")
    model = SentenceTransformerModel(model_name)
    
    # 2. Define Tasks (Covering all 8 MTEB categories)
    tasks = [
        STSBTask(),                # STS
        Banking77Task(),           # Classification
        TwentyNewsgroupsTask(),    # Clustering
        QQPTask(),                 # Pair Classification
        SciDocsRetrievalTask(),    # Retrieval
        AskUbuntuTask(),           # Reranking
        SimulatedSummarizationTask(), # Summarization
        TatoebaTask()              # Bitext Mining
    ]
    
    # 3. Run Benchmark
    results = {}
    for task in tasks:
        print(f"\n--- Running Task: {task.name} ---")
        try:
            task_res = task.run(model)
            results[task.name] = task_res
            print(f"Result for {task.name}: {task_res}")
        except Exception as e:
            print(f"Error running {task.name}: {e}")
            results[task.name] = {"error": str(e)}

    # 4. Summary
    print("\n=== Benchmark Summary ===")
    for task_name, metrics in results.items():
        print(f"{task_name}:")
        if "error" in metrics:
             print(f"  Error: {metrics['error']}")
        else:
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
