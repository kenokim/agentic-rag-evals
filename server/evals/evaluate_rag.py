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
    from simple_rag import SimpleRAG
    from service import LLM_MODEL, EMBEDDING_MODEL
except ImportError:
    # If running from root, adjust path
    sys.path.append(os.path.join(os.getcwd(), 'server'))
    from simple_rag import SimpleRAG
    from service import LLM_MODEL, EMBEDDING_MODEL

def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def main():
    # 1. Check if predictions_with_gt.jsonl exists
    pred_file = os.path.join(current_dir, "predictions_with_gt.jsonl")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    if os.path.exists(pred_file):
        print(f"Found existing predictions at {pred_file}. Using them for evaluation.")
        eval_data = load_dataset(pred_file)
        
        for item in eval_data:
            questions.append(item.get("question", ""))
            answers.append(item.get("answer", ""))
            
            # Ensure contexts is a list of strings
            ctx = item.get("contexts", [])
            if isinstance(ctx, list):
                # Convert all elements to string just in case
                contexts.append([str(c) for c in ctx])
            else:
                contexts.append([])
                
            ground_truths.append(item.get("ground_truth", ""))
            
        print(f"Loaded {len(questions)} items from file.")
        
    else:
        # Initialize RAG
        print("Initializing SimpleRAG...")
        try:
            rag = SimpleRAG()
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            return
    
        # Load evaluation data
        dataset_path = os.path.join(server_dir, "dataset", "evals.jsonl")
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at {dataset_path}")
            return
    
        print(f"Loading dataset from {dataset_path}...")
        eval_data = load_dataset(dataset_path)
        
        # Run evaluation
        print(f"Running evaluation on {len(eval_data)} items...")
        for i, item in enumerate(eval_data):
            question = item["question"]
            ground_truth = item["ground_truth"]
            
            print(f"Processing {i+1}/{len(eval_data)}: {question}")
            
            try:
                # Get answer from RAG
                result = rag.get_answer(question)
                
                questions.append(question)
                answers.append(result["answer"])
                # Ensure contexts is a list of strings
                retrieved_contexts = result.get("contexts", [])
                # Fallback if contexts not present
                if not retrieved_contexts and "sources" in result:
                     retrieved_contexts = [s.get("content", "") for s in result["sources"]]
                
                contexts.append(retrieved_contexts)
                ground_truths.append(ground_truth)
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                continue
            
    # Prepare dataset for Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    # Setup LLM and Embeddings for Ragas
    # Using the same Google models as the application
    evaluator_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    evaluator_embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Evaluate
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
        output_csv = os.path.join(current_dir, "results.csv")
        results_df = results.to_pandas()
        results_df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

        # Save as JSON as well
        output_json = output_csv.replace('.csv', '.json')
        results_df.to_json(output_json, orient='records', force_ascii=False, indent=2)
        print(f"Results saved to {output_json}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()
