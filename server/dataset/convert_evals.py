import json
import re

def parse_markdown_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 구분자로 섹션 나누기
    sections = content.split('⸻')
    
    jsonl_data = []
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        lines = section.split('\n')
        
        question = ""
        answer = ""
        context = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                question = line[2:].strip()
            elif line.startswith('A:'):
                answer = line[2:].strip()
            elif line.startswith('Context:'):
                context = line[8:].strip()
                # Remove quotes and trailing characters if present
                context = context.strip('"').strip('“').strip('”')
                # Remove trailing special characters like ￼
                context = re.sub(r'\s*￼.*$', '', context)

        if question and answer:
            entry = {
                "question": question,
                "ground_truth": answer,
                "contexts": [context] if context else []
            }
            jsonl_data.append(entry)

    # JSONL 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    input_path = "/Users/keno0403/Desktop/agent/rag-evals-ragas/server/dataset/evals.md"
    output_path = "/Users/keno0403/Desktop/agent/rag-evals-ragas/server/dataset/evals.jsonl"
    
    parse_markdown_to_jsonl(input_path, output_path)
    print(f"Converted {input_path} to {output_path}")







