baseline_prompt = """
  You are a specialist in dental radiology.
  Analyze the provided panoramic radiograph image.

  Return ONLY valid JSON (no markdown, no explanations, no extra text).

  Rules:
  - Use only numeric FDI codes (e.g., 11, 16, 26, 36): 
      two-digit numbers where the first digit is the quadrant (1–4) and the second digit is the tooth position (1–8).
  - Sort values in ascending order.
  - Do not include duplicates.
  - Do not add extra fields.

  JSON format:
  {
    "Missing teeth": [],
    "Present teeth": []
  }
  """

prompt_missing = """
  You are a dental radiology specialist.

  Analyze the provided panoramic dental X-ray and identify ONLY missing teeth.

  Definition:
  A tooth is considered missing if no crown or root structure is visible at its expected position.

  Output rules:
  - Return ONLY valid JSON.
  - Include exactly one key: "Missing teeth".
  - Values must be numeric FDI tooth codes as strings (e.g., "11", "36").
  - Sort ascending.
  - No duplicates.
  - If no teeth are missing, return an empty list.
  - No markdown, no explanations, no extra text.

  JSON format:
  {
    "Missing teeth": []
  }
  """

prompt_qwen = """
  You are a dental radiology assistant.

  Analyze the provided panoramic dental X-ray.

  Definition:
  A tooth is considered missing if no crown or root structure is visible at its expected position.

  Output rules:
  - Return ONLY valid JSON.
  - Include exactly the keys listed below, no more and no less.
  - Values must be lists of numeric FDI tooth codes as strings (e.g., "11", "36").
  - Sort values in ascending order.
  - Do not include duplicates.
  - If nothing is detected for a class, return an empty list.
  - No markdown, no explanations, no extra text.

  Classes to report:
  {CLASSES}

  JSON format:
  {{
  {JSON_SCHEMA}
  }}
  """


# TODO: might test this format
prompt_qwen_system = {
  "SYSTEM" : """
    You are a dental radiology assistant.
    Output ONLY valid JSON. No markdown. No extra text."
  """,
  "USER_TEXT": """
    Analyze the provided panoramic dental X-ray and identify ONLY missing teeth.

      Definition:
      A tooth is considered missing if no crown or root structure is visible at its expected position.

      Output rules:
      - Return ONLY valid JSON.
      - Include exactly one key: "Missing teeth".
      - Values must be numeric FDI tooth codes as strings (e.g., "11", "36").
      - Sort ascending.
      - No duplicates.
      - If no teeth are missing, return an empty list.
      - No markdown, no explanations, no extra text.

      JSON format:
      {
        "Missing teeth": []
      }  
  """
}