# Ragas 평가 지표 가이드 (Ground Truth 유무에 따른 분류)

이 문서는 데이터셋에 Ground Truth(정답)가 포함되어 있는지 여부에 따라 측정 가능한 Ragas 평가 지표를 정리합니다.

## 1. Ground Truth가 없는 경우 (Unlabeled Data)
정답 데이터 없이 질문(Question)과 시스템이 생성한 답변(Answer), 그리고 검색된 맥락(Contexts)만으로 평가할 수 있는 지표입니다. 주로 생성 모델(Generator)의 품질을 평가합니다.

### 측정 가능한 지표

*   Faithfulness (충실도)
    *   설명: 생성된 답변이 검색된 맥락(Context)에 있는 정보에 기반하고 있는지 측정합니다.
    *   목적: 모델이 없는 사실을 지어내는 '환각(Hallucination)' 현상을 탐지하기 위함입니다.
    *   필요 데이터: Question, Answer, Contexts

*   Answer Relevancy (답변 관련성)
    *   설명: 생성된 답변이 사용자의 질문(Question)에 대해 얼마나 적절하고 관련성 있는지 측정합니다.
    *   목적: 질문의 의도에 맞는 답변을 하고 있는지 확인하기 위함입니다.
    *   필요 데이터: Question, Answer

## 2. Ground Truth가 있는 경우 (Labeled Data)
질문(Question)에 대한 모범 답안(Ground Truth)이 포함된 데이터셋입니다. 검색 모델(Retriever)의 성능과 전체적인 정확도를 평가할 수 있는 지표가 추가됩니다. (Ground Truth가 없는 경우의 지표들도 함께 측정 가능합니다.)

### 추가 측정 가능한 지표

*   Context Recall (맥락 재현율)
    *   설명: 정답(Ground Truth)을 도출하는 데 필요한 정보가 검색된 맥락(Contexts)에 포함되어 있는지 측정합니다.
    *   목적: 검색 단계에서 필요한 정보를 빠뜨리지 않고 가져왔는지 확인하기 위함입니다.
    *   필요 데이터: Question, Contexts, Ground Truth

*   Context Precision (맥락 정밀도)
    *   설명: 검색된 맥락(Contexts) 중에서 정답(Ground Truth)과 관련된 중요한 정보가 상위에 랭크되어 있는지 측정합니다.
    *   목적: 불필요한 정보(Noise) 대비 유용한 정보의 비율과 순서를 평가하기 위함입니다.
    *   필요 데이터: Question, Contexts, Ground Truth

## 3. Ragas 사용 시나리오 (데이터셋 준비 방식에 따른 분류)

Ragas를 사용하여 평가를 진행하는 방식은 데이터셋이 이미 준비되어 있는지, 아니면 실시간으로 생성되는지에 따라 크게 두 가지로 나뉩니다.

### A. 이미 저장된 데이터셋(JSON 등)을 사용하는 경우 (Offline Evaluation)
과거의 로그나 미리 만들어둔 테스트셋(Golden Dataset)을 사용하여 일괄적으로 평가하는 방식입니다.

*   상황:
    *   이미 `question`, `answer`, `contexts`, `ground_truth` 등이 포함된 JSON, CSV, HuggingFace Dataset 파일이 있는 경우.
    *   배포 전, 준비된 테스트셋으로 모델의 성능을 벤치마킹할 때.
*   흐름:
    1.  데이터 파일을 로드합니다 (예: `Dataset.from_json()`).
    2.  Ragas의 `evaluate()` 함수에 해당 데이터셋을 전달합니다.
    3.  평가 결과를 받아 분석합니다.

### B. API를 호출하여 실시간으로 평가하는 경우 (Online/Dynamic Evaluation)
RAG 파이프라인을 실제로 실행하면서 그때그때 생성되는 결과를 평가하는 방식입니다.

*   상황:
    *   테스트셋은 질문(`question`)만 있고, 답변(`answer`)과 검색 결과(`contexts`)를 현재 RAG 시스템에서 새로 생성해야 하는 경우.
    *   실제 운영 환경이나 테스트 코드 내에서 파이프라인의 실행 결과를 즉시 평가하고 싶을 때.
*   흐름:
    1.  평가할 질문 리스트를 준비합니다.
    2.  각 질문에 대해 RAG API(Retriever + Generator)를 호출하여 `answer`와 `contexts`를 수집합니다.
    3.  수집된 데이터를 Ragas가 요구하는 데이터셋 포맷(`Dataset` 객체 등)으로 변환하거나, 단건 평가 함수를 사용합니다.
    4.  `evaluate()` 함수를 실행하여 점수를 산출합니다.
