"""Descriptive prompt-pattern checks; these are never classifier features."""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
train = pd.read_csv(ROOT / "dataset/train.csv")
test = pd.read_csv(ROOT / "dataset/test.csv")
patterns = {
    "instruction_override_vocabulary": r"(?i)ignore|ignora|ignorare|ignorez|desconsidere|previous|precedent|pr[eé]c[eé]dent|anterior",
    "instruction_vocabulary": r"(?i)instru|system|syst[eè]m|sistema|directive|direttiv",
    "template_placeholders": r"\[(?:PROMPT|TARGETLANGUAGE|LINGUA|LANGUE|IDIOMA)",
    "role_markers": r"(?i)assistant:|system:|\[INST\]|<\|im_start\|>|<start_of_turn>",
    "url": r"https?://",
    "non_latin_script": r"[\u0400-\u04ff\u3040-\u30ff\u4e00-\u9fff\u0600-\u06ff]",
    "vietnamese_specific_letters": r"[\u1ea0-\u1ef9\u0111\u0110]",
}
report = {}
for name, pattern in patterns.items():
    matched = train.text.str.contains(pattern,regex=True)
    report[name] = {
        "train_count":int(matched.sum()),"test_count":int(test.text.str.contains(pattern,regex=True).sum()),
        "training_class_fraction":train.assign(matched=matched).groupby("label").matched.mean().to_dict(),
        "training_language_count":train.loc[matched,"language"].value_counts().to_dict(),
    }
report["interpretation"] = "Language tags do not guarantee a single script; translated templates and placeholders occur. Descriptive correlations are not rules or label overrides. The neural model uses original prompt text."
path = ROOT / "artifacts/audit/patterns.json"
path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(report,indent=2,ensure_ascii=True))
