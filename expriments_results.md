# Dataset Infomation:
1. Data source 1 (Hugging Face):
    o Data set name: IELTS Writing Task 2 Evaluation Dataset (https://huggingface.co/datasets/chillies/IELTS-writing-task-2-evaluation).
    o Source: Hugging Face (author: chillies).
    o Quantity: About 10,324 lines.
    o Features: Each sample includes a prompt, an essay, a detailed assessment according to 4 criteria and a total score.
2. Data source 2 (Kaggle): 
    o Data set name: Raw IELTS Essays (https://www.kaggle.com/datasets/arsenycheplukov/raw-ielts-essays). 
    o Source: Kaggle (author: Arseny Cheplukov). 
    o Number: 6,944 lines. 
    o Characteristics: Data includes articles, total score, CEFR level and detailed comments for each criterion (Task Response, Coherence & Cohesion, Lexical Resource, Grammatical Range & Accuracy).
- Data Processing For valuation: IELTS Writing Task 2 Evaluation Dataset + Raw IELTS Essays

------------
# Approach 1: Classification Tuning
## Study One: BERT-based Classification Tuning (Roberta-base Encoder)
| Training Scheme   | Accuracy | RMSE  | MAE    | F1-Score |
|------------------|-----------|-------|--------|-----------|
| Classifier Only  | 0.651     | 0.830 | 0.581  | 0.216     |
| All Parameters   | 0.731     | 0.784 | 0.549  | 0.304     |

## Study Two: LLM-based Classification Tuning (GPT-2 Encoder)
| Training Scheme   | Accuracy | RMSE  | MAE    | F1-Score |
|------------------|-----------|-------|--------|-----------|
| Classifier Only  | 0.707     | 0.757 | 0.530  | 0.278     |
| All Parameters   | 0.735     | 0.770 | 0.539  | 0.309     |


# Approach 2: Prompting / Instruction Tuning
## Study Three: Prompting Tuning
| Model          | Prompt Type  | Accuracy  | RMSE  | MAE    | F1-Score  |
|----------------|--------------|-----------|-------|--------|-----------|
| GPT-4o         | few-shot     | 0.710     | 1.050 | 0.735  | 0.281     |
| GPT-4o         | zero-shot    | 0.720     | 1.130 | 0.791  | 0.292     |
| Llama-3-70B    | few-shot     | 0.560     | 1.250 | 0.875  | 0.116     |
| Llama-3-70B    | zero-shot    | 0.630     | 0.990 | 0.693  | 0.193     |
| Gemini 2.5 Pro | zero-shot    | 0.9616    | 1.4977| 1.2026 | 0.5581    |

## Study Four: Instruction Tuning
| Model                           | Prompt Type | Accuracy | RMSE   | MAE    | F1-Score |
|---------------------------------|-------------|----------|--------|--------|----------|
| Gemma-7B                        | zero-shot   | 0.6700   | 1.3300 | 0.9800 | 0.2200   |
| Phi-2                           | zero-shot   | 0.7812   | 1.3500 | 0.6505 | 0.4100   |
| Mistral-7B                      | zero-shot   | 0.7000   | 1.4128 | 1.1214 | 0.2700   |
| Llama-3.1-8B                    | zero-shot   | 0.7596   | 1.0172 | 0.9859 | 0.4511   |
| Llama-3.1-8B                    | 2-shot      | 0.8680   | 0.9400 | 1.0200 | 0.5200   |
| Llama-3.1-8B                    | 4-shot      | 0.8708   | 1.3547 | 0.9436 | 0.5679   |


# Approach 3: k-Instruction Tuning (SFT) + RAG
## Experimental results with RAG on the Llama-3.1-8B model
| Model                                            | k-shot  | Accuracy | F1-Score | RMSE   | MAE    |
|--------------------------------------------------|---------|----------|----------|--------|--------|
| Mistral-7B-Instruct-v0.3 (not finetuned) + RAG   | 2-shot  | 0.9050   | 0.7200   | 1.1800 | 0.8500 |
| Llama-3.1-8B (not finetuned) + RAG               | 2-shot  | 0.9630   | 0.8400   | 0.9800 | 0.7000 |
| Llama-3.1-8B (1-LoRA) + RAG                      | 2-shot  | 0.9750   | 0.8800   | 0.9200 | 0.6600 |
| Llama-3.1-8B (4-LoRA)                            | 2-shot  | 0.9818   | 0.7999   | 0.9947 | 0.7676 |
| Llama-3.1-8B (4-LoRA) + RAG                      | 2-shot  | 0.9902   | 0.9350   | 0.8700 | 0.6200 |


# Approach 4: Supervised Fine-tuning (SFT) + RL (DPO) + RAG  
| Model                             | k-shot  | Accuracy | F1-Score | RMSE   | MAE   |
|-----------------------------------|---------|----------|----------|--------|-------|
| Mistral-7B-Instruct-v0.3 (1-LoRA) | 2-shot  | 0.9470   | 0.8750   | 1.0300 | 0.8300|
| Llama-3.1-8B (1-LoRA) + DPO + RAG | 2-shot  | 0.9870   | 0.9250   | 0.8400 | 0.5800|

------------
### Main Results on Automated Essay Scoring (AES)
| Model / Setting                              | Training / Prompting Scheme        | k-shot    | Accuracy | F1-Score | RMSE   | MAE    |
|----------------------------------------------|------------------------------------|-----------|----------|----------|--------|--------|
| Roberta-base (BERT-based)                    | Classifier Only                    | –         | 0.6510   | 0.2160   | 0.8300 | 0.5810 |
| Roberta-base (BERT-based)                    | All Parameters                     | –         | 0.7310   | 0.3040   | 0.7840 | 0.5490 |
| GPT-2 (LLM-based)                            | Classifier Only                    | –         | 0.7070   | 0.2780   | 0.7570 | 0.5300 |
| GPT-2 (LLM-based)                            | All Parameters                     | –         | 0.7350   | 0.3090   | 0.7700 | 0.5390 |
|----------------------------------------------|------------------------------------|-----------|----------|----------|--------|--------|
| GPT-4o                                       | Prompting Tuning                   | 2-shot    | 0.7100   | 0.2810   | 1.0500 | 0.7350 |
| GPT-4o                                       | Prompting Tuning                   | zero-shot | 0.7200   | 0.2920   | 1.1300 | 0.7910 |
| Llama-3-70B                                  | Prompting Tuning                   | 2-shot    | 0.5600   | 0.1160   | 1.2500 | 0.8750 |
| Llama-3-70B                                  | Prompting Tuning                   | zero-shot | 0.6300   | 0.1930   | 0.9900 | 0.6930 |
| Gemini 2.5 Pro                               | Prompting Tuning                   | zero-shot | 0.9616   | 0.5581   | 1.4977 | 1.2026 |
|----------------------------------------------|------------------------------------|-----------|----------|----------|--------|--------|
| Gemma-7B                                     | Instruction Tuning                 | zero-shot | 0.6700   | 0.2200   | 1.3300 | 0.9800 |
| Phi-2                                        | Instruction Tuning                 | zero-shot | 0.7812   | 0.4100   | 1.3500 | 0.6505 |
| Mistral-7B                                   | Instruction Tuning                 | zero-shot | 0.7000   | 0.2700   | 1.4128 | 1.1214 |
| Llama-3.1-8B                                 | Instruction Tuning                 | zero-shot | 0.7596   | 0.4511   | 1.0172 | 0.9859 |
| Llama-3.1-8B                                 | Instruction Tuning + Few-shot      | 2-shot    | 0.8680   | 0.5200   | 0.9400 | 1.0200 |
| Llama-3.1-8B                                 | Instruction Tuning + Few-shot      | 4-shot    | 0.8708   | 0.5679   | 1.3547 | 0.9436 |
|----------------------------------------------|------------------------------------|-----------|----------|----------|--------|--------|
| Mistral-7B-Instruct-v0.3 (not finetuned)     | k-Instruction Tuning + RAG         | 2-shot    | 0.9050   | 0.7200   | 1.1800 | 0.8500 |
| Llama-3.1-8B (not finetuned)                 | k-Instruction Tuning + RAG         | 2-shot    | 0.9630   | 0.8400   | 0.9800 | 0.7000 |
| Llama-3.1-8B (1-LoRA)                        | k-Instruction Tuning + RAG         | 2-shot    | 0.9750   | 0.8800   | 0.9200 | 0.6600 |
| Llama-3.1-8B (4-LoRA)                        | k-Instruction Tuning (w/o RAG)     | 2-shot    | 0.9818   | 0.7999   | 0.9947 | 0.7676 |
| **Llama-3.1-8B (4-LoRA)**                        | **k-Instruction Tuning + RAG**         | **2-shot**    | **0.9902**   | **0.9350**   | <u>0.8700</u> | <u>0.6200</u> |
|----------------------------------------------|------------------------------------|-----------|----------|----------|--------|--------|
| Mistral-7B-Instruct-v0.3 (1-LoRA)            | SFT + DPO + RAG                    | 2-shot    | 0.9470   | 0.8750   | 1.0300 | 0.8300 |
| Llama-3.1-8B (1-LoRA)                        | SFT + DPO + RAG                    | 2-shot    | <u>0.9870</u>   | <u>0.9250</u>   | **0.8400** | **0.5800** |