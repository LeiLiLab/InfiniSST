import re
import pandas as pd
from pathlib import Path

in_path = Path("/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_rank32_iter_0000452_hf_zh_k1_10_k2_sweep_glossary2/zh/k1_10_k2_sweep_glossary2_streamlaal_summary.tsv")
out_path = in_path.with_name(in_path.stem + "_parsed.tsv")

df = pd.read_csv(in_path, sep="\t", dtype=str).fillna("")

# Regex patterns
re_k = re.compile(r"_k2(\d+)_k1(\d+)(?:_|$)")
re_cs = re.compile(r"_cs([0-9]*\.?[0-9]+)(?:_|$)")

def parse_row(p: str):
    glossary_type = ""
    if "gglossary_acl6060" in p:
        glossary_type = "acl6060"
    elif "gextracted_glossary_with_translations" in p:
        glossary_type = "paper_extracted"
    else:
        glossary_type = "unknown"

    m_k = re_k.search(p)
    k2 = m_k.group(1) if m_k else ""
    k1 = m_k.group(2) if m_k else ""

    m_cs = re_cs.search(p)
    cs = m_cs.group(1) if m_cs else ""

    return glossary_type, cs, k2, k1

parsed = df["output_path"].apply(parse_row)
df["glossary_type"] = parsed.apply(lambda x: x[0])
df["chunk_size"] = parsed.apply(lambda x: x[1])

# 回填 K2/K1：如果原列是空，就用 path 解析到的；如果原列非空就保留原值
k2_from_path = parsed.apply(lambda x: x[2])
k1_from_path = parsed.apply(lambda x: x[3])
if "K2" not in df.columns: df["K2"] = ""
if "K1" not in df.columns: df["K1"] = ""
df["K2"] = df["K2"].where(df["K2"].str.strip() != "", k2_from_path)
df["K1"] = df["K1"].where(df["K1"].str.strip() != "", k1_from_path)

num_cols = ["chunk_size", "K2", "K1", "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC"]

# 转成数值
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 统一格式化成 2 位小数
for col in num_cols:
    df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
df.to_csv(out_path, sep="\t", index=False)
print(f"Wrote: {out_path}")