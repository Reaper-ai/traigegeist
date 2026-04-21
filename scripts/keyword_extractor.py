import re
import numpy as np
import pandas as pd
from typing import List, Set

# --- Configuration Constants ---
ABBREV_MAP = {
    "sob": "shortness of breath", "cp": "chest pain", "loc": "loss of consciousness",
    "ams": "altered mental status", "sz": "seizure", "mi": "myocardial infarction",
    "cva": "stroke", "od": "overdose", "si": "suicidal ideation", "hi": "homicidal ideation",
}

PLACEHOLDER_VALUES = {"", "blank", "unknown", "unknown/blank", "blank unknown", "nan", "none", "na", "n a"}

NEGATION_PHRASES = ["negative for", "no evidence of", "without", "denies", "denied", "free of", "absence of", "absent", "no", "not"]
NEGATION_TOKENS = {"no", "not", "denies", "denied", "without", "negative", "free", "absence", "absent", "never"}

# This list defines both the keywords to search and the columns to keep
TARGET_KEYWORDS = [
    "chest pain", "shortness of breath", "syncope", "assault", "vaginal bleeding",
    "violence", "burn", "head injury", "suicide attempt", "cardiac arrest",
    "gunshot wound", "throat swelling", "paralysis", "sepsis"
]

class EmergencyTextProcessor:
    def __init__(self, keywords: List[str]):
        self.keywords = keywords
        # Pre-compile patterns for efficiency
        self.patterns = {kw: re.compile(rf"(?<!\w){re.escape(kw)}(?!\w)") for kw in keywords}
        self.col_map = {kw: re.sub(r"[^a-z0-9]+", "_", kw.lower()).strip("_") for kw in keywords}

    def normalize(self, text: any) -> str:
        """Cleans text, expands abbreviations, and removes placeholders."""
        if pd.isna(text):
            return ""
        t = str(text).lower().strip()
        t = re.sub(r"[/|_]+", " ", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()

        if t in PLACEHOLDER_VALUES:
            return ""

        for abbr, expanded in ABBREV_MAP.items():
            t = re.sub(rf"\b{re.escape(abbr)}\b", expanded, t)
        
        return t if t not in PLACEHOLDER_VALUES else ""

    def _is_negated(self, text: str, start_idx: int) -> bool:
        """Checks for negation in the immediate context preceding the keyword."""
        left_context = text[max(0, start_idx - 100):start_idx]
        segment = re.split(r"[.;,]", left_context)[-1].strip()
        if not segment:
            return False

        if any(phrase in segment for phrase in NEGATION_PHRASES):
            return True

        tokens = re.findall(r"[a-z0-9']+", segment)
        if any(tok in NEGATION_TOKENS for tok in tokens[-6:]):
            return True
        return False

    def check_presence(self, text: str, pattern: re.Pattern) -> int:
        """Returns 1 if non-negated keyword is found, else 0."""
        for match in pattern.finditer(text):
            if not self._is_negated(text, match.start()):
                return 1
        return 0

    def process_dataframe(self, df: pd.DataFrame, text_cols: List[str]) -> pd.DataFrame:
        """Processes the dataframe and returns only row_index and keyword flags."""
        # 1. Normalize and Combine Text
        temp_combined = df[text_cols].apply(lambda x: " ".join(x.fillna("").map(self.normalize)), axis=1)
        temp_combined = temp_combined.str.replace(r"\s+", " ", regex=True).str.strip()
        
        text_list = temp_combined.tolist()
        results = {"row_index": df.index.astype(np.int64)}

        # 2. Vectorized-ish keyword extraction
        for kw in self.keywords:
            col_name = self.col_map[kw]
            pattern = self.patterns[kw]
            results[col_name] = np.fromiter(
                (self.check_presence(t, pattern) for t in text_list),
                dtype=np.int8, count=len(text_list)
            )

        return pd.DataFrame(results)

def get_emergency_flags(df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
    """
    Modular entry point to process a dataframe directly.
    """
    if text_columns is None:
        text_columns = ["chief_complaint_text", "injury_cause_text"]
    
    processor = EmergencyTextProcessor(TARGET_KEYWORDS)
    return processor.process_dataframe(df, text_columns)

# # --- Example Usage ---
# if __name__ == "__main__":
#     # # Example mock data
#     # data = {
#     #     "chief_complaint_text": ["Chest pain", "No SOB", "Fever"],
#     #     "injury_cause_text": ["nan", "Assaulted by peer", "None"]
#     # }
#     # test_df = pd.DataFrame(data)
    
#     # flags = get_emergency_flags(test_df)
#     # print(flags)
#     pass