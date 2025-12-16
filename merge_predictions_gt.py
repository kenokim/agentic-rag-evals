import json

predictions_file = 'server/evals/predictions.jsonl'
gt_file = 'server/evals/questions_with_gt.jsonl'
output_file = 'server/evals/predictions_with_gt.jsonl'

# Load ground truths
gt_map = {}
with open(gt_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        gt_map[data['question']] = data['ground_truth']

# Process predictions and add ground truth
with open(predictions_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        question = data['question']
        
        if question in gt_map:
            data['ground_truth'] = gt_map[question]
        else:
            print(f"Warning: No ground truth found for question: {question}")
            
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Created {output_file} with ground truth.")
