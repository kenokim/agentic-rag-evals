# RAG API Evaluation Strategies

Chat API(RAG 파이프라인)의 성능을 객관적으로 측정하고 개선하기 위한 평가 방법론을 정리합니다.

## 1. RAG 평가 프레임워크

### RAGAS (RAG Assessment)
가장 널리 사용되는 오픈소스 RAG 평가 프레임워크입니다. "LLM-as-a-Judge" 방식을 사용하여, 또 다른 LLM(GPT-4 등)이 RAG 시스템의 답변 품질을 채점합니다.

*   특징: 참조(Ground Truth) 데이터 없이도 일부 지표(Faithfulness, Answer Relevance)를 평가할 수 있습니다.
*   주요 지표:
    *   Generation (생성 품질)
        *   Faithfulness (신뢰성): 답변이 주어진 문맥(Context)에 근거하고 있는가? (할루시네이션 탐지)
        *   Answer Relevance (답변 관련성): 답변이 질문의 의도에 부합하는가?
    *   Retrieval (검색 품질)
        *   Context Precision (문맥 정밀도): 검색된 내용 중 실제로 유용한 정보의 비율이 높은가?
        *   Context Recall (문맥 재현율): 답변에 필요한 핵심 정보가 빠짐없이 검색되었는가? (Ground Truth 필요)

### 기타 프레임워크
*   DeepEval: PyTest와 통합되어 CI/CD 파이프라인에서 테스트하기 좋은 프레임워크입니다.
*   TruLens: RAG 앱의 실험 추적 및 평가를 위한 도구입니다.

---

## 2. 평가를 위한 데이터셋 구성

평가를 수행하기 위해서는 다음과 같은 필드로 구성된 데이터셋이 필요합니다.

| 필드 | 설명 | 필수 여부 |
|------|------|----------|
| `question` | 사용자 질문 | Yes |
| `answer` | RAG 시스템이 생성한 답변 | Yes |
| `contexts` | RAG 시스템이 검색(Retrieve)한 문서 조각들의 리스트 | Yes |
| `ground_truth` | 사람이 작성한 이상적인 정답 | Yes (Recall 측정 시) |

### 데이터 예시 (JSON)
```json
{
  "question": "이 논문에서 제안하는 Transformer 모델의 주요 특징은?",
  "answer": "Transformer는 순환 신경망(RNN) 대신 어텐션 메커니즘만을 사용합니다.",
  "contexts": [
    "Transformer 모델은 RNN이나 CNN을 사용하지 않고 Attention만을...",
    "기존 Seq2Seq 모델의 한계를 극복하기 위해..."
  ],
  "ground_truth": "Transformer는 Recurrence 없이 Attention 메커니즘 전적으로 의존하는 모델 아키텍처입니다."
}
```

---

## 3. 테스트 데이터셋 구축 방법 (Synthetic Data)

평가 데이터(Golden Dataset)를 사람이 일일이 만드는 것은 매우 번거롭습니다. RAGAS는 LLM을 활용해 문서로부터 자동으로 테스트 데이터를 생성해주는 기능을 제공합니다.

### Synthetic Testset Generation (데이터 합성)
1.  Source Documents: RAG에 사용할 원본 문서(PDF 등)를 준비합니다.
2.  Generation: RAGAS의 `TestsetGenerator`를 사용하여 문서 내용을 바탕으로 `question`과 `ground_truth` 쌍을 자동으로 생성합니다.
    *   단순 질문 (Simple)
    *   추론이 필요한 질문 (Reasoning)
    *   여러 문맥을 조합해야 하는 질문 (Multi-context)
3.  Review: 생성된 데이터 중 품질이 낮은 것은 사람이 검수하여 제거하거나 수정합니다.

---

## 4. 평가 프로세스 (Workflow)

1.  준비 (Preparation): 평가할 RAG 파이프라인과 테스트 데이터셋(`question`, `ground_truth`) 준비.
2.  실행 (Inference): 테스트 데이터셋의 `question`을 RAG API에 전송하여 `answer`와 `contexts`를 수집.
3.  평가 (Evaluation): 수집된 데이터(`question`, `answer`, `contexts`, `ground_truth`)를 RAGAS 라이브러리에 입력하여 점수 산출.
4.  분석 (Analysis): 점수가 낮은 항목을 분석하여 검색(Retrieval) 문제인지 생성(Generation) 문제인지 파악 후 튜닝.

---

## 5. 일반 생성 평가 (General Generation Evaluation) - LLM-as-a-Judge

RAG의 검색(Retrieval) 성능과 별개로, **생성 모델의 답변 자체의 품질**이나 **특정 기준 준수 여부**를 평가해야 할 때가 있습니다. 이때 LLM(주로 GPT-4)을 심판(Judge)으로 활용하여 생성 결과를 채점하는 방식을 **LLM-as-a-Judge**라고 합니다.

### 평가 방식 유형

1.  **절대 평가 (Single-point Grading)**
    *   하나의 답변을 보고 점수(1~5점)를 매기거나, Pass/Fail 여부를 판정합니다.
    *   예: "이 답변이 도움이 되었는가? (1-5)", "사용자의 요청 사항을 모두 포함했는가? (Yes/No)"

2.  **상대 평가 (Pairwise Comparison)**
    *   두 개의 모델 답변(예: 기존 모델 A vs 신규 모델 B)을 동시에 보여주고 더 나은 답변을 선택하게 합니다.
    *   사람의 선호도 평가(RLHF)와 유사한 방식입니다.

### 주요 평가 기준 (Criteria)

일반적인 생성 작업에서 자주 사용되는 평가 기준입니다. 사용자가 커스텀 기준을 정의할 수도 있습니다.

*   **유용성 (Helpfulness)**: 답변이 사용자의 문제를 실제로 해결해 주는가?
*   **유창성 (Fluency & Coherence)**: 문법이 정확하고 문맥이 자연스럽게 연결되는가?
*   **안전성 (Safety/Harmlessness)**: 유해하거나 편향된 내용이 포함되지 않았는가?
*   **지시 이행 (Instruction Following)**: 프롬프트에 명시된 제약 조건(예: "3문장 이내로 요약하라", "JSON으로 출력하라")을 지켰는가?
*   **커스텀 기준 (Custom Criteria)**: 프로젝트 특화 기준 (예: "브랜드 톤앤매너를 지켰는가?", "코드 예제가 실행 가능한가?")

### LLM-as-a-Judge 프롬프트 예시

평가를 위해 LLM에게 전달하는 프롬프트는 보통 다음과 같은 구조를 가집니다.

> **System:** 당신은 공정한 AI 보조자 평가자입니다. 아래의 기준(Criteria)에 따라 사용자의 질문에 대한 모델의 답변을 1점에서 5점 사이로 평가하고, 그 이유를 설명하세요.
>
> **Criteria:**
> - 답변은 명확하고 간결해야 합니다.
> - 사용자의 질문에 직접적으로 대답해야 합니다.
>
> **Input:**
> - 사용자 질문: [Question]
> - 모델 답변: [Answer]
>
> **Output Format:**
> - Score: [1-5]
> - Reason: [평가 이유]
