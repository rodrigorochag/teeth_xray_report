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

prompt_segment = """
  You are a dental radiology specialist.

  Analyze the provided panoramic dental X-ray and identify ONLY missing teeth.
  To help you, we provided X-ray with segmented teeth, please pay attention to it.

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

prompt_openai_upd = """
You are a dental radiology specialist.

Analyze the provided panoramic dental X-ray and identify ONLY missing teeth.

Definition:
A tooth is considered missing ONLY if no crown AND no root structure is visible at its expected anatomical position.

Do NOT mark a tooth as missing if:
- Any part of the crown or root is visible.
- The tooth is impacted or partially erupted.
- Only the crown is absent but the root is visible.
- The tooth has endodontic treatment, a crown, or a large restoration.
- Image quality, overlap, or artifacts make the tooth position ambiguous.

If visibility is uncertain, assume the tooth is PRESENT.

Process:
1. Examine the image quadrant by quadrant in FDI order (11-18, 21-28, 31-38, 41-48).
2. For each expected tooth position, decide whether a tooth is clearly visible.
3. Collect only those teeth that are confidently NOT visible.

Output rules:
- Return ONLY valid JSON.
- Include exactly one key: "Missing teeth".
- Values must be FDI tooth codes as strings (e.g., "11", "36").
- Sort ascending.
- No duplicates.
- If no teeth are missing, return an empty list.
- No markdown, no explanations, no extra text.

JSON format:
{
  "Missing teeth": []
}

If the output is not valid JSON, regenerate internally until it is valid.
"""

prompt_segment_upd = """
You are a dental radiology specialist.

You are given:
1) A panoramic dental X-ray.
2) A segmentation overlay where each visible tooth structure is highlighted.

Your task is to identify ONLY missing teeth.

Definition:
A tooth is considered missing ONLY if:
- No segmented region AND
- No crown AND no root structure
are visible at its expected anatomical position.

Segmentation rules:
- If any segmented region overlaps the expected position of a tooth, the tooth is PRESENT.
- If segmentation is incomplete or noisy but any tooth structure is visible, treat the tooth as PRESENT.
- Segmentation takes priority over visual ambiguity.

Do NOT mark a tooth as missing if:
- Any segmented tooth structure is present.
- The tooth is impacted, partially erupted, or rotated.
- Only the crown is missing but a root is visible.
- The tooth has endodontic treatment, a crown, or restoration.
- Image quality, overlap, or artifacts introduce uncertainty.

If there is uncertainty, assume the tooth is PRESENT.

Process:
1. Scan teeth in strict FDI order (11–18, 21–28, 31–38, 41–48).
2. For each expected tooth position, check for segmented or visible structure.
3. Collect only teeth that are confidently NOT present.

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

If the output is not valid JSON, regenerate internally until it is valid.
"""

prompt_complete = """
You are a specialist in dental radiology.
Analyze the provided panoramic radiograph image and produce a structured report by filling in the following template in JSON format. Write according to World Dental Federation (FDI).
The teeth are segmented in the image, use them as a support tool.

Important instructions:

Fill each field according to the findings visible in the image.
Write each field only with numeric data.

If nothing is detected, do not write.
Do not add extra fields.
Write the dentition type as "0" for permanent and "1" for mixed.
The result must be in JSON format, mirroring the input's structure.

{
  "objects": [
    "Missing teeth:",
    "Present teeth:",
    "Crown lesions:",
    "Type of dentition:",
    "Endodontic treatment:"
  ],
  "time": ""
}
"""

prompt_complete2 = """
You are a specialist in dental radiology.
Analyze the provided panoramic radiograph image and produce a structured report by filling in the following template in JSON format. Write according to World Dental Federation (FDI).
The teeth are segmented in the image, use them as a support tool.

Important instructions:

Fill each field according to the findings visible in the image.
Write each field only with numeric data.

If nothing is detected, do not write.
Do not add extra fields.
Write the dentition type as "0" for permanent and "1" for mixed.
The result must be in JSON format, mirroring the input's structure.

{
  "Missing teeth": [],
  "Present teeth": [],
  "Crown lesions": [],
  "Type of dentition": [],
  "Endodontic treatment": [],
}

JSON format:
{
  "Missing teeth": []
}
"""

prompt_complete3 = """
You are a specialist in dental radiology.
Analyze the provided panoramic radiograph image and produce a structured report by filling in the following template in JSON format. Write according to World Dental Federation (FDI).
The teeth are segmented in the image, use them as a support tool.

Important instructions:

Fill each field according to the findings visible in the image.
Write each field only with numeric data.
Write in ms the "time" field with the time spent to analysis.

If nothing is detected, do not write.
Do not add extra fields.
Write the dentition type as "0" for permanent and "1" for mixed.
The result must be in JSON format, mirroring the input's structure.

{
  "Missing teeth": [],
  "Present teeth": [],
  "Crown lesions": [],
  "Type of dentition": [],
  "Endodontic treatment": [],
  "time": time, ms.
}

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


#---------------------
# Zero-shot vs Guided
#---------------------
prompt_zero_shot = """
You are a specialist in dental radiology.

Analyze the panoramic radiograph and identify the teeth presenting the following characteristics:

- Missing teeth
- Endodontic treatment
- Crown lesions
- Mesial inclination
- Implant

Formatting rules:
- Lists must contain only FDI tooth numbers as integers (e.g., [11, 16, 26, 36]).
- Each list should include the teeth where the corresponding condition is visible.
- A tooth may appear in multiple lists if the corresponding conditions are visible.
- If no teeth present the condition, return an empty list.

Output must be in JSON format:
{
  "Missing teeth": [],
  "Endodontic treatment": [],
  "Crown lesions": [],
  "Mesial inclination": [],
  "Implant": [],
}
"""

prompt_zero_shot_segm = """
You are a specialist in dental radiology.

Analyze the panoramic radiograph and identify the teeth presenting the following characteristics:

- Missing teeth
- Endodontic treatment
- Crown lesions
- Mesial inclination
- Implant

The teeth are segmented in the image, use them as a support tool.

Formatting rules:
- Lists must contain only FDI tooth numbers as integers (e.g., [11, 16, 26, 36]).
- Each list should include the teeth where the corresponding condition is visible.
- A tooth may appear in multiple lists if the corresponding conditions are visible.
- If no teeth present the condition, return an empty list.

Output must be in JSON format:
{
  "Missing teeth": [],
  "Endodontic treatment": [],
  "Crown lesions": [],
  "Mesial inclination": [],
  "Implant": [],
}
"""

prompt_guided = """
You are a specialist in dental radiology.

Analyze the panoramic radiograph and identify the teeth presenting the following characteristics:

- Missing teeth
- Endodontic treatment
- Crown lesions
- Mesial inclination
- Implant

Field definitions (numeric-only output):
1. Missing teeth - absence of a tooth in the expected anatomical position.
2. Endodontic treatment — presence of radiopaque filling material within the root canal.
3. Crown lesions — structural defects affecting the crown of the tooth visible radiographically.
4. Mesial inclination — tooth axis tilted toward the dental midline.
5. Implant — metallic screw-like structure replacing the root of a missing tooth.

Formatting rules:
- Lists must contain only FDI tooth numbers as integers (e.g., [11, 16, 26, 36]).
- Each list should include the teeth where the corresponding condition is visible.
- A tooth may appear in multiple lists if the corresponding conditions are visible.
- If no teeth present the condition, return an empty list.

Output must be in JSON format:
{
  "Missing teeth": [],
  "Endodontic treatment": [],
  "Crown lesions": [],
  "Mesial inclination": [],
  "Implant": [],
}
"""

prompt_guided_segm = """
You are a specialist in dental radiology.

Analyze the panoramic radiograph and identify the teeth presenting the following characteristics:

- Missing teeth
- Endodontic treatment
- Crown lesions
- Mesial inclination
- Implant

The teeth are segmented in the image, use them as a support tool.

Field definitions (numeric-only output):
1. Missing teeth - absence of a tooth in the expected anatomical position.
2. Endodontic treatment — presence of radiopaque filling material within the root canal.
3. Crown lesions — structural defects affecting the crown of the tooth visible radiographically.
4. Mesial inclination — tooth axis tilted toward the dental midline.
5. Implant — metallic screw-like structure replacing the root of a missing tooth.

Formatting rules:
- Lists must contain only FDI tooth numbers as integers (e.g., [11, 16, 26, 36]).
- Each list should include the teeth where the corresponding condition is visible.
- A tooth may appear in multiple lists if the corresponding conditions are visible.
- If no teeth present the condition, return an empty list.

Output must be in JSON format:
{
  "Missing teeth": [],
  "Endodontic treatment": [],
  "Crown lesions": [],
  "Mesial inclination": [],
  "Implant": [],
}
"""