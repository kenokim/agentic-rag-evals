import os
import sys
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Add server directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(current_dir)
sys.path.append(server_dir)

try:
    from service import LLM_MODEL, EMBEDDING_MODEL
except ImportError:
    # If running from root, adjust path
    sys.path.append(os.path.join(os.getcwd(), 'server'))
    from service import LLM_MODEL, EMBEDDING_MODEL

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def _get_arg(name: str, default: str | None = None) -> str | None:
    """Tiny argv parser: --name value"""
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return default
    return sys.argv[idx + 1]

def main():
    input_filename = _get_arg("--input")
    output_filename = _get_arg("--output")

    print("Starting evaluation of predictions jsonl...")
    
    # 1. Load predictions
    # Prefer user-provided input, otherwise prefer predictions_with_gt.jsonl if present.
    if input_filename:
        pred_path = os.path.join(current_dir, input_filename)
    else:
        default_with_gt = os.path.join(current_dir, "predictions_with_gt.jsonl")
        default_pred = os.path.join(current_dir, "predictions.jsonl")
        pred_path = default_with_gt if os.path.exists(default_with_gt) else default_pred
    if not os.path.exists(pred_path):
        print(f"Predictions file not found at {pred_path}")
        return
    
    predictions = load_jsonl(pred_path)
    print(f"Loaded {len(predictions)} predictions from {pred_path}.")

    # 2. Load Ground Truths (if needed)
    gt_path = os.path.join(current_dir, "questions_with_gt.jsonl")
    gt_map = {}
    if os.path.exists(gt_path):
        print(f"Loading ground truths from {gt_path}...")
        gt_data = load_jsonl(gt_path)
        for item in gt_data:
            if "question" in item and "ground_truth" in item:
                gt_map[item["question"]] = item["ground_truth"]
    else:
        print("Ground truth file not found. Evaluation might be limited.")

    # 3. Prepare data for Ragas
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in predictions:
        q = item.get("question")
        a = item.get("answer")
        c = item.get("contexts", [])
        
        # Fallback for ground_truth
        gt = item.get("ground_truth")
        if not gt and q in gt_map:
            gt = gt_map[q]
            
        # Ensure data integrity
        if q and a:
            questions.append(q)
            answers.append(a)
            # Ragas expects list of strings for contexts
            if isinstance(c, list):
                contexts.append([str(ctx) for ctx in c])
            else:
                contexts.append([])
            ground_truths.append(gt if gt else "")
            
    # Create HF Dataset
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    # 4. Setup Ragas LLM & Embeddings
    print("Initializing Ragas models...")
    evaluator_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # 5. Run Evaluation
    print("Calculating metrics...")
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        
        print("\nEvaluation Results:")
        print(results)
        
        # Save results
        if output_filename:
            output_csv = os.path.join(current_dir, output_filename)
        else:
            base = os.path.splitext(os.path.basename(pred_path))[0]
            output_csv = os.path.join(current_dir, f"{base}_evaluation_results.csv")
        results_df = results.to_pandas()
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

