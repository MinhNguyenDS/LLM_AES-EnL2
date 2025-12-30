from google import genai
import os
import pandas as pd
import json
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Optional

# --- Định nghĩa json class kết quả trả về từ Gemini --- 
class Criteria(BaseModel):
    Band: str
    Comment: str

class CriteriaWithMistakes(BaseModel):
    Band: str
    Comment: str
    Mistakes: List[str]
    Corrections: List[str]

class Ielt(BaseModel):
    Task_Response: Criteria
    Coherence_and_Cohesion: Criteria
    Lexical_Resource: CriteriaWithMistakes
    Grammatical_Range_and_Accuracy: CriteriaWithMistakes
    Overall_Band_Score: str
    General_Feedback: str


# --- Tạo model ---
client = genai.Client()

model_name = "gemini-2.5-pro"

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

config = {
  "response_mime_type": "application/json",
  "response_schema": Ielt,
  "temperature": 0.8,
  "top_p": 1,
  "top_k": 32, 
  "max_output_tokens": 10000,
  "safety_settings": safety_settings
}

def evaluate_essay(overall_band, essay_prompt, essay_text):
    """
    Gọi Gemini để chấm điểm IELTS Writing Task 2
    Trả về dict chứa các trường đã parse từ JSON
    """
    prompt = f"""
    You are an IELTS Writing Task 2 examiner. 
    The official overall band score of the essay is: {overall_band}.  
    Your task is to evaluate the essay strictly according to IELTS Writing Task 2 band descriptors.  

    Scoring criteria:
    1. Task Response (TR) – Did the candidate address the question fully, provide clear arguments, support with examples?
    2. Coherence and Cohesion (CC) – Is the essay logically organized with effective paragraphing and linking?
    3. Lexical Resource (LR) – Range, appropriacy, and accuracy of vocabulary (please list mistakes and suggest corrections).
    4. Grammatical Range and Accuracy (GRA) – Variety and correctness of grammar and punctuation (please list mistakes and suggest corrections).

    Instructions:
    - Provide a band score (0–9, half bands allowed) for each of the four criteria.
    - The arithmetic mean of the four scores should be as close as possible to the given overall band: {overall_band}.
    - After scoring, provide constructive feedback with strengths and areas for improvement.

    Output format (strict JSON only, no extra text):
    {{
      "Task_Response": {{
        "Band": <score>,
        "Comment": "..."
      }},
      "Coherence_and_Cohesion": {{
        "Band": <score>,
        "Comment": "..."
      }},
      "Lexical_Resource": {{
        "Band": <score>,
        "Mistakes": ["...", "..."],
        "Corrections": ["...", "..."],
        "Comment": "..."
      }},
      "Grammatical_Range_and_Accuracy": {{
        "Band": <score>,
        "Mistakes": ["...", "..."],
        "Corrections": ["...", "..."],
        "Comment": "..."
      }},
      "Overall_Band_Score": <overall_band>,
      "General_Feedback": "..."
    }}

    Essay prompt: {essay_prompt}
    Essay: {essay_text}
    """

    try:
        response = client.models.generate_content(model=model_name, 
                                contents=prompt, 
                                config=config,
                                  
                            ) 

        result_json: Ielt = response.parsed
        return result_json
    except Exception as e:
        return {"error": str(e), "raw_output": response.text if 'response' in locals() else ""}
    

def process_csv_in_batches(input_csv, output_csv, batch_size=10, start_row=0, max_batches=None):
    """
    Đọc file CSV input theo batch, chấm điểm từng essay và append vào file output.
    - input_csv: file CSV nguồn
    - output_csv: file CSV đích (sẽ được append, nếu chưa có thì tạo mới)
    - batch_size: số dòng xử lý mỗi batch
    - start_row: dòng bắt đầu (nếu muốn resume từ giữa)
    - max_batches: số batch tối đa sẽ chạy trong lần này (None = chạy hết)
    """
    reader = pd.read_csv(input_csv, chunksize=batch_size)

    first_batch = True if start_row == 0 else False
    processed_rows = 0
    batch_count = 0

    for chunk in reader:
        # Nếu chưa tới start_row thì skip hoặc cắt chunk
        if processed_rows + len(chunk) <= start_row:
            processed_rows += len(chunk)
            continue
        elif processed_rows < start_row:
            # Cắt bỏ phần đầu của chunk (chỉ lấy từ start_row trở đi)
            offset = start_row - processed_rows
            chunk = chunk.iloc[offset:]
            processed_rows = start_row

        batch_count += 1
        if max_batches is not None and batch_count > max_batches:
            print(f"⏹️ Đã đạt giới hạn {max_batches} batch, dừng lại tại dòng {processed_rows}.")
            break

        results = []
        for _, row in tqdm(chunk.iterrows(), total=len(chunk)):
            overall_band = row["band"]
            essay_prompt = row["prompt"]
            essay_text = row["essay"]

            eval_result = evaluate_essay(overall_band, essay_prompt, essay_text)

            if type(eval_result) is Ielt:
                results.append({
                    "band": overall_band,
                    "prompt": essay_prompt,
                    "essay": essay_text,

                    "TR_Band": eval_result.Task_Response.Band,
                    "TR_Comment": eval_result.Task_Response.Comment,

                    "CC_Band": eval_result.Coherence_and_Cohesion.Band,
                    "CC_Comment": eval_result.Coherence_and_Cohesion.Comment,

                    "LR_Band": eval_result.Lexical_Resource.Band,
                    "LR_Mistakes": "; ".join(eval_result.Lexical_Resource.Mistakes),
                    "LR_Corrections": "; ".join(eval_result.Lexical_Resource.Corrections),
                    "LR_Comment": eval_result.Lexical_Resource.Comment,

                    "GRA_Band": eval_result.Grammatical_Range_and_Accuracy.Band,
                    "GRA_Mistakes": "; ".join(eval_result.Grammatical_Range_and_Accuracy.Mistakes),
                    "GRA_Corrections": "; ".join(eval_result.Grammatical_Range_and_Accuracy.Corrections),
                    "GRA_Comment": eval_result.Grammatical_Range_and_Accuracy.Comment,

                    "Overall_Band": eval_result.Overall_Band_Score,
                    "General_Feedback": eval_result.General_Feedback
                })
            else:
                results.append({
                    "band": overall_band,
                    "prompt": essay_prompt,
                    "essay": essay_text,

                    "TR_Band": None, "TR_Comment": None,
                    "CC_Band": None, "CC_Comment": None,
                    "LR_Band": None, "LR_Mistakes": None, "LR_Corrections": None, "LR_Comment": None,
                    "GRA_Band": None, "GRA_Mistakes": None, "GRA_Corrections": None, "GRA_Comment": None,
                    "Overall_Band": None, "General_Feedback": None,
                    "Error": eval_result.get("error", "Unknown error"),
                    "Raw_Output": eval_result.get("raw_output", "")
                })

        # ghi ngay batch ra file output
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, mode="a", header=first_batch, index=False)
        first_batch = False
        processed_rows += len(chunk)

        print(f"✅ Đã xử lý xong {processed_rows} dòng, batch {batch_count}, kết quả được ghi vào {output_csv}")


# --- Chạy gen file ---
process_csv_in_batches("train.csv", "new_train.csv", start_row=460, batch_size=10, max_batches=1000)