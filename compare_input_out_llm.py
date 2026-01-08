import json
import os
import glob
import csv
from collections import defaultdict

# -----------------------------------------------------------------------------
# CONSTANTS AND CONFIGURATION
# -----------------------------------------------------------------------------

# 1. Definition of the 32 permanent teeth using FDI notation.
# Format: Quadrant (1-4) + Position (1-8). E.g., "18", "46".
FDI_TEETH = set()
for quadrant in [1, 2, 3, 4]:
    for position in range(1, 9):
        FDI_TEETH.add(f"{quadrant}{position}")

# 2. Target classes for evaluation.
# Only findings belonging to these categories will be processed.
TARGET_CLASSES = {
    "Present teeth",
    "Missing teeth",
    "Presence of implants",
    "Endodontic treatment",
    "Tooth with mesial inclination",
    "Crown lesions",
    "Type of dentition"
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def parse_json_content(data):
    """
    Normalizes the JSON content into a dictionary of sets, handling heterogeneous formats.
    
    This function processes the 'objects' key, which may appear as:
    - A dictionary: {"Class": [item1, item2]}
    - A list of strings: ["Class: item1, item2"]

    Args:
        data (dict): The raw JSON data loaded from a file.

    Returns:
        defaultdict(set): A dictionary mapping category names to sets of finding strings.
                          Example: {'Missing teeth': {'18', '28'}, ...}
    """
    parsed = defaultdict(set)
    objects = data.get("objects", [])

    # Case A: 'objects' is a Dictionary
    if isinstance(objects, dict):
        for category, items in objects.items():
            if category in TARGET_CLASSES:
                # Ensure 'items' is iterable even if it's a single value
                if not isinstance(items, list):
                    items = [items]
                for item in items:
                    parsed[category].add(str(item).strip())

    # Case B: 'objects' is a List of Strings
    elif isinstance(objects, list):
        for obj_str in objects:
            if ": " in obj_str:
                category, values_str = obj_str.split(": ", 1)
                if category in TARGET_CLASSES:
                    values = [v.strip() for v in values_str.split(",")]
                    for val in values:
                        parsed[category].add(val)
    
    return parsed

def extract_findings_with_logic(data):
    """
    Extracts findings and applies domain-specific logic rules.

    Rule Applied:
        Complementarity of Teeth: If the category 'Missing teeth' exists in the data,
        'Present teeth' is mathematically derived as the set difference between 
        all valid FDI teeth (32) and the missing teeth. This overwrites any explicit
        'Present teeth' list to ensure consistency.

    Args:
        data (dict): The raw JSON data.

    Returns:
        set: A set of tuples (Category, Item) representing the final findings for evaluation.
             Example: {('Missing teeth', '18'), ('Present teeth', '11'), ...}
    """
    # 1. Initial Parsing
    parsed_data = parse_json_content(data)

    # 2. Logic Application: Deduce Present Teeth from Missing Teeth
    if "Missing teeth" in parsed_data:
        missing_set = parsed_data["Missing teeth"]
        
        # Filter strictly valid FDI teeth to avoid artifacts (e.g., '0' or deciduous codes)
        valid_missing = {t for t in missing_set if t in FDI_TEETH}
        
        # Calculate the complement set
        present_set = FDI_TEETH - valid_missing
        parsed_data["Present teeth"] = present_set

    # 3. Flatten structure for evaluation
    findings = set()
    for category, items in parsed_data.items():
        if category in TARGET_CLASSES:
            for item in items:
                findings.add((category, item))
    
    return findings

def get_common_categories(set_a, set_b):
    """
    Identifies the intersection of categories present in two sets of findings.
    
    Args:
        set_a (set): Set of tuples (Category, Item).
        set_b (set): Set of tuples (Category, Item).

    Returns:
        set: A set of category names (strings) present in both inputs.
    """
    cats_a = {cat for cat, val in set_a}
    cats_b = {cat for cat, val in set_b}
    return cats_a.intersection(cats_b)

# -----------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# -----------------------------------------------------------------------------

def calculate_metrics_logic_enhanced(output_dir, gt_dir, csv_filename="metrics_derived.csv"):
    """
    Calculates detection metrics (Precision, Recall, F1) comparing Output vs. Ground Truth.
    
    Features:
    - Micro-average aggregation.
    - Logic-based data augmentation (deducing present teeth).
    - Intersection-based filtering (evaluates only categories present in both files).
    - CSV export of results.

    Args:
        output_dir (str): Path to the folder containing model output JSON files.
        gt_dir (str): Path to the folder containing ground truth JSON files.
        csv_filename (str): Name of the CSV file to save results.
    """
    # Accumulators for metrics
    class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    global_tp = 0
    global_fp = 0
    global_fn = 0
    files_processed = 0

    output_files = glob.glob(os.path.join(output_dir, "*.json"))
    
    print(f"Processing {len(output_files)} files with Logic Derivation...")

    for out_path in output_files:
        filename = os.path.basename(out_path)
        gt_path = os.path.join(gt_dir, filename)

        if not os.path.exists(gt_path):
            # Skip if no corresponding ground truth file exists
            continue

        # Load JSON files
        with open(out_path, 'r', encoding='utf-8') as f:
            out_data = json.load(f)
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)

        # 1. Extract findings with domain logic (Missing -> Present deduction)
        out_set_raw = extract_findings_with_logic(out_data)
        gt_set_raw = extract_findings_with_logic(gt_data)

        # 2. Filter by Category Intersection
        # Ignore categories that appear in one file but not the other for the current image.
        common_cats = get_common_categories(out_set_raw, gt_set_raw)

        out_set = {(c, v) for c, v in out_set_raw if c in common_cats}
        gt_set = {(c, v) for c, v in gt_set_raw if c in common_cats}

        # 3. Calculate True Positives, False Positives, False Negatives
        tp_set = out_set.intersection(gt_set)
        fp_set = out_set.difference(gt_set)
        fn_set = gt_set.difference(out_set)

        # 4. Update Global Accumulators
        global_tp += len(tp_set)
        global_fp += len(fp_set)
        global_fn += len(fn_set)

        # 5. Update Per-Class Accumulators
        for (cat, val) in tp_set: class_stats[cat]['tp'] += 1
        for (cat, val) in fp_set: class_stats[cat]['fp'] += 1
        for (cat, val) in fn_set: class_stats[cat]['fn'] += 1

        files_processed += 1

    # -------------------------------------------------------------------------
    # EXPORT RESULTS TO CSV
    # -------------------------------------------------------------------------
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Class', 'Precision', 'Recall', 'F1-Score', 'TP', 'FP', 'FN']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write Per-Class Metrics
        for cls in sorted(list(TARGET_CLASSES)):
            stats = class_stats[cls]
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            
            denom_prec = tp + fp
            denom_rec = tp + fn
            
            prec = tp / denom_prec if denom_prec > 0 else 0.0
            rec = tp / denom_rec if denom_rec > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            writer.writerow({
                'Class': cls,
                'Precision': f"{prec:.4f}",
                'Recall': f"{rec:.4f}",
                'F1-Score': f"{f1:.4f}",
                'TP': tp, 'FP': fp, 'FN': fn
            })

        # Write Global Metrics (Micro-Average)
        g_prec = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        g_rec = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0

        writer.writerow({
            'Class': 'GLOBAL',
            'Precision': f"{g_prec:.4f}",
            'Recall': f"{g_rec:.4f}",
            'F1-Score': f"{g_f1:.4f}",
            'TP': global_tp, 'FP': global_fp, 'FN': global_fn
        })

    print(f"Processing complete. Processed {files_processed} images.")
    print(f"Results saved to: {os.path.abspath(csv_filename)}")


if __name__ == "__main__":
    output_folder = "/home/rodrigo/Documents/LLM_INFERENCE/input_llm/prompt/sasha/OUTPUT/complete_report/output_llm_guided"
    gt_folder = "/home/rodrigo/Documents/LLM_INFERENCE/input_llm/prompt/sasha/Ground_truth"
    
    calculate_metrics_logic_enhanced(output_folder, gt_folder, "evaluation_results.csv")
