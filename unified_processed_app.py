import datetime
import gc
import io
import json
import logging
import math
import os
import pathlib
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from openai import OpenAI


os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


APP_TITLE = "Cloud RAG Data Assistant"
APP_SUBTITLE = "Unified pre + post process for sanitized or dummy CSV data"
SESSION_ROOT = os.path.join(tempfile.gettempdir(), "streamlit_cloud_rag")
CHUNK_SIZE_MB = 64
UPLOAD_WARNING_MB = 400
HARD_ROW_LIMIT = 1000
MAX_ROWS_FOR_SUMMARY = 50
FULL_SCHEMA_THRESHOLD = 180
DEFAULT_PROVIDER_NAME = "DeepSeek"

PROVIDER_CONFIG = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "secret_key": "DEEPSEEK_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "secret_key": "OPENAI_API_KEY",
    },
    "xAI (Grok)": {
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-2-latest",
        "secret_key": "XAI_API_KEY",
    },
}

DANGEROUS_KEYWORDS = [
    "COPY",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "INSTALL",
    "LOAD",
    "ATTACH",
    "DETACH",
    "EXPORT",
    "IMPORT",
    "CALL",
    "EXECUTE",
    "SET",
    "PRAGMA",
    "HTTPFS",
    "VACUUM",
    "CHECKPOINT",
    "GRANT",
    "REVOKE",
    "TRUNCATE",
    "MERGE",
    "PREPARE",
    "READ_CSV",
    "READ_CSV_AUTO",
    "READ_TEXT",
    "READ_JSON",
    "READ_JSON_AUTO",
    "READ_BLOB",
    "READ_PARQUET_SCHEMA",
    "GLOB",
    "PARQUET_SCAN",
]

PII_COLUMN_PATTERNS = [
    r"(?i)\bname\b",
    r"(?i)\bfirst.?name\b",
    r"(?i)\blast.?name\b",
    r"(?i)\bfull.?name\b",
    r"(?i)\bemail\b",
    r"(?i)\bphone\b",
    r"(?i)\baddress\b",
    r"(?i)\bssn\b",
    r"(?i)social.?sec",
    r"(?i)\bdob\b",
    r"(?i)birth.?date",
    r"(?i)date.?of.?birth",
    r"(?i)\bpassword\b",
    r"(?i)\blogin\b",
    r"(?i)\busername\b",
    r"(?i)\bstudent.?id\b",
    r"(?i)\bemployee.?id\b",
    r"(?i)\bparent.?name\b",
    r"(?i)\bguardian\b",
    r"(?i)\bcontact\b",
    r"(?i)\bpersonal\b",
]


st.set_page_config(page_title=APP_TITLE, layout="wide")


def remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def robust_rmtree(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path, onerror=remove_readonly)
        except Exception:
            logging.warning("Could not fully remove %s", path)


def ensure_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload sanitized data, process it, and then ask a question."}
        ]
    if "dataset_ready" not in st.session_state:
        st.session_state.dataset_ready = False
    if "bundle_dir" not in st.session_state:
        st.session_state.bundle_dir = ""
    if "artifacts_dir" not in st.session_state:
        st.session_state.artifacts_dir = ""
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "starter_questions" not in st.session_state:
        st.session_state.starter_questions = []
    if "pii_redaction" not in st.session_state:
        st.session_state.pii_redaction = True
    if "processing_summary" not in st.session_state:
        st.session_state.processing_summary = {}


def build_session_paths():
    root = os.path.join(SESSION_ROOT, st.session_state.session_id)
    bundle_dir = os.path.join(root, "bundle")
    upload_dir = os.path.join(root, "uploads")
    artifacts_dir = os.path.join(bundle_dir, "artifacts")
    return root, bundle_dir, upload_dir, artifacts_dir


def reset_dataset_state():
    root, _, _, _ = build_session_paths()
    robust_rmtree(root)
    st.session_state.dataset_ready = False
    st.session_state.bundle_dir = ""
    st.session_state.artifacts_dir = ""
    st.session_state.metadata = None
    st.session_state.starter_questions = []
    st.session_state.processing_summary = {}
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload sanitized data, process it, and then ask a question."}
    ]
    execute_validated_sql.clear()


def sanitize_table_name(filename):
    name = os.path.splitext(filename)[0]
    clean = re.sub(r"[^a-zA-Z0-9]", "_", name).lower().strip("_")
    if not clean:
        clean = "table"
    if clean[0].isdigit():
        clean = "t_" + clean
    return clean


def clean_column_names(columns):
    return [str(c).strip().replace('"', "") for c in columns]


def get_all_csvs(root_dir):
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file_name))
    return sorted(csv_files)


def get_parquet_row_count(parquet_path):
    return pq.ParquetFile(parquet_path).metadata.num_rows


def safe_extract(zip_path, target_dir):
    target = pathlib.Path(target_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            member_path = (target / member).resolve()
            if not str(member_path).startswith(str(target)):
                raise ValueError(f"Security alert: ZIP member escapes target directory: {member}")
        zf.extractall(target_dir)


def tokenize(text):
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def normalize_identifier(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def singularize_name(value):
    clean = normalize_identifier(value)
    if clean.endswith("ies") and len(clean) > 3:
        return clean[:-3] + "y"
    if clean.endswith("ses") and len(clean) > 3:
        return clean[:-2]
    if clean.endswith("s") and not clean.endswith("ss") and len(clean) > 1:
        return clean[:-1]
    return clean


def candidate_entity_names(table_name):
    normalized = normalize_identifier(table_name)
    singular = singularize_name(table_name)
    names = {normalized, singular}
    if singular.endswith("history"):
        names.add(singular.replace("history", ""))
    if singular.endswith("enrollment"):
        names.add("user")
    return {name for name in names if name}


def escape_sql_literal(value):
    return value.replace("'", "''")


def detect_pii_columns(df):
    pii_cols = []
    for col in df.columns:
        for pattern in PII_COLUMN_PATTERNS:
            if re.search(pattern, col):
                pii_cols.append(col)
                break
    return pii_cols


def redact_pii(df):
    pii_cols = detect_pii_columns(df)
    if not pii_cols:
        return df, []
    redacted_df = df.copy()
    for col in pii_cols:
        redacted_df[col] = "[REDACTED]"
    return redacted_df, pii_cols


def split_parquet_to_chunks(source_parquet, table_name, rows_per_chunk, output_dir, status):
    pf = pq.ParquetFile(source_parquet)
    schema = pf.schema_arrow
    chunk_idx = 0
    buffered_tables = []
    buffered_rows = 0

    for batch in pf.iter_batches(batch_size=min(rows_per_chunk, 100_000)):
        table = pa.Table.from_batches([batch], schema=schema)
        offset = 0

        while offset < table.num_rows:
            take_rows = min(rows_per_chunk - buffered_rows, table.num_rows - offset)
            piece = table.slice(offset, take_rows)
            buffered_tables.append(piece)
            buffered_rows += take_rows
            offset += take_rows

            if buffered_rows >= rows_per_chunk:
                chunk_path = os.path.join(output_dir, f"{table_name}_{chunk_idx}.parquet")
                pq.write_table(pa.concat_tables(buffered_tables), chunk_path, compression="zstd")
                status.write(f"   • Wrote chunk `{os.path.basename(chunk_path)}`")
                chunk_idx += 1
                buffered_tables = []
                buffered_rows = 0

    if buffered_tables:
        chunk_path = os.path.join(output_dir, f"{table_name}_{chunk_idx}.parquet")
        pq.write_table(pa.concat_tables(buffered_tables), chunk_path, compression="zstd")
        status.write(f"   • Wrote chunk `{os.path.basename(chunk_path)}`")


def detect_relationships(conn, tables_metadata, artifacts_dir, status):
    table_columns = {}
    normalized_columns = {}
    relationships = []
    seen = set()

    for table_name in tables_metadata:
        first_chunk = os.path.join(artifacts_dir, f"{table_name}_0.parquet").replace("\\", "/")
        schema_df = conn.execute(f"DESCRIBE SELECT * FROM read_parquet('{first_chunk}')").df()
        columns = schema_df["column_name"].tolist()
        table_columns[table_name] = columns
        normalized_columns[table_name] = {normalize_identifier(col): col for col in columns}

    table_names = list(table_columns.keys())
    for left_table in table_names:
        for right_table in table_names:
            if left_table == right_table:
                continue

            left_entities = candidate_entity_names(left_table)
            left_norm_map = normalized_columns[left_table]
            right_norm_map = normalized_columns[right_table]

            for right_norm, right_col in right_norm_map.items():
                if not right_norm.endswith("id"):
                    continue

                base_name = right_norm[:-2]
                pk_candidates = ["id"] + [f"{entity}id" for entity in left_entities]
                if base_name in left_entities:
                    for pk_norm in pk_candidates:
                        if pk_norm in left_norm_map:
                            pk_col = left_norm_map[pk_norm]
                            rel_key = (right_table, right_col, left_table, pk_col)
                            if rel_key not in seen:
                                seen.add(rel_key)
                                relationships.append(
                                    {
                                        "from_table": right_table,
                                        "from_column": right_col,
                                        "to_table": left_table,
                                        "to_column": pk_col,
                                    }
                                )
                            break

            common_norms = set(left_norm_map.keys()).intersection(set(right_norm_map.keys()))
            for common_norm in common_norms:
                if common_norm == "id" or not common_norm.endswith("id"):
                    continue
                common_col_left = left_norm_map[common_norm]
                common_col_right = right_norm_map[common_norm]
                rel_key = tuple(sorted([left_table, right_table]) + [common_norm])
                if rel_key not in seen:
                    seen.add(rel_key)
                    relationships.append(
                        {
                            "from_table": left_table,
                            "from_column": common_col_left,
                            "to_table": right_table,
                            "to_column": common_col_right,
                        }
                    )

    if relationships:
        status.write(f"✅ Detected {len(relationships)} likely relationship(s).")
    else:
        status.write("ℹ️ No obvious table relationships detected.")
    return relationships


def extract_schema_metadata(conn, first_chunk_path, table_name):
    schema_df = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{first_chunk_path.replace(os.sep, '/')}')"
    ).df()
    columns_meta = []

    for _, row in schema_df.iterrows():
        col_name = row["column_name"]
        dtype = row["column_type"]
        try:
            samples = conn.execute(
                f"""
                SELECT "{col_name}"::VARCHAR
                FROM read_parquet('{first_chunk_path.replace(os.sep, '/')}')
                WHERE "{col_name}" IS NOT NULL
                LIMIT 3
                """
            ).fetchall()
            sample_str = ", ".join(str(item[0]) for item in samples)
        except Exception:
            sample_str = "N/A"

        desc = f"Table: {table_name}\nColumn: {col_name}\nType: {dtype}\nSamples: {sample_str}"
        columns_meta.append(
            {"name": col_name, "type": dtype, "description": desc, "table": table_name}
        )

    return columns_meta


def analyze_table_overlaps(metadata):
    table_columns = {}
    normalized_columns = {}

    for column in metadata.get("columns", []):
        table_columns.setdefault(column["table"], []).append(column["name"])
        normalized_columns.setdefault(column["table"], {})[normalize_identifier(column["name"])] = column["name"]

    overlaps = []
    table_names = sorted(table_columns.keys())
    for index, left_table in enumerate(table_names):
        for right_table in table_names[index + 1:]:
            left_norm = normalized_columns.get(left_table, {})
            right_norm = normalized_columns.get(right_table, {})
            common_norms = sorted(set(left_norm.keys()).intersection(set(right_norm.keys())))
            if not common_norms:
                continue

            shared_columns = []
            shared_key_columns = []
            for common_norm in common_norms:
                left_col = left_norm[common_norm]
                right_col = right_norm[common_norm]
                shared_columns.append(
                    {
                        "normalized_name": common_norm,
                        "left_column": left_col,
                        "right_column": right_col,
                    }
                )
                if common_norm == "id" or common_norm.endswith("id") or common_norm.endswith("key"):
                    shared_key_columns.append(
                        {
                            "normalized_name": common_norm,
                            "left_column": left_col,
                            "right_column": right_col,
                        }
                    )

            overlaps.append(
                {
                    "left_table": left_table,
                    "right_table": right_table,
                    "shared_column_count": len(shared_columns),
                    "shared_key_count": len(shared_key_columns),
                    "shared_columns": shared_columns,
                    "shared_key_columns": shared_key_columns,
                }
            )

    return overlaps


def inspect_csv_headers(csv_paths):
    header_map = {}
    signature_map = {}
    for csv_path in csv_paths:
        try:
            header_df = pd.read_csv(csv_path, nrows=0, encoding_errors="replace")
            clean_headers = tuple(clean_column_names(header_df.columns))
        except Exception:
            clean_headers = tuple()
        header_map[os.path.basename(csv_path)] = clean_headers
        signature_map.setdefault(clean_headers, []).append(os.path.basename(csv_path))
    return header_map, signature_map


def process_merge_strategy(conn, csv_paths, artifacts_dir, temp_dir, status):
    status.write("🔗 Strategy: Merge all CSV files into one logical table named `data`.")
    temp_master = os.path.join(temp_dir, "master.parquet")
    normalized_paths = [path.replace(os.sep, "/") for path in csv_paths]
    input_files_sql = ", ".join(f"'{path}'" for path in normalized_paths)

    strategies = [
        (
            "UTF-8",
            f"""
            COPY (
                SELECT * FROM read_csv_auto([{input_files_sql}], sample_size=100000)
            ) TO '{temp_master.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
            """,
        ),
        (
            "Latin-1",
            f"""
            COPY (
                SELECT * FROM read_csv_auto([{input_files_sql}], sample_size=100000, encoding='latin-1')
            ) TO '{temp_master.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
            """,
        ),
        (
            "Ignore Errors",
            f"""
            COPY (
                SELECT * FROM read_csv_auto(
                    [{input_files_sql}],
                    sample_size=100000,
                    encoding='latin-1',
                    ignore_errors=true
                )
            ) TO '{temp_master.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
            """,
        ),
    ]

    conversion_success = False
    for label, sql in strategies:
        try:
            if os.path.exists(temp_master):
                os.remove(temp_master)
            conn.execute(sql)
            status.write(f"✅ CSV parse succeeded with the {label} strategy.")
            conversion_success = True
            break
        except Exception as exc:
            logging.info("Merge strategy %s failed: %s", label, exc)

    if not conversion_success:
        status.write("⚠️ DuckDB parsing failed. Falling back to Pandas chunk reads.")
        temp_chunks = []
        for csv_file in csv_paths:
            header_df = pd.read_csv(csv_file, nrows=0, encoding_errors="replace")
            clean_cols = clean_column_names(header_df.columns)
            with pd.read_csv(
                csv_file,
                chunksize=200_000,
                encoding_errors="replace",
                on_bad_lines="skip",
            ) as reader:
                for chunk in reader:
                    chunk.columns = clean_cols
                    chunk_path = os.path.join(temp_dir, f"chunk_{uuid.uuid4().hex}.parquet")
                    chunk.to_parquet(chunk_path, engine="pyarrow", index=False)
                    temp_chunks.append(chunk_path)

        if not temp_chunks:
            raise ValueError("No readable rows were found in the uploaded CSV files.")

        chunk_pattern = os.path.join(temp_dir, "chunk_*.parquet").replace(os.sep, "/")
        conn.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet('{chunk_pattern}')
            ) TO '{temp_master.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
            """
        )

    table_name = "data"
    total_rows = get_parquet_row_count(temp_master)
    file_size_mb = os.path.getsize(temp_master) / (1024 * 1024)

    if file_size_mb < CHUNK_SIZE_MB:
        final_path = os.path.join(artifacts_dir, f"{table_name}_0.parquet")
        os.rename(temp_master, final_path)
    else:
        num_chunks = max(1, math.ceil(file_size_mb / CHUNK_SIZE_MB))
        rows_per_chunk = max(1, math.ceil(total_rows / num_chunks))
        status.write(
            f"✂️ Splitting the merged table into about {num_chunks} Parquet files to keep chunks query-friendly."
        )
        split_parquet_to_chunks(temp_master, table_name, rows_per_chunk, artifacts_dir, status)
        os.remove(temp_master)

    first_chunk = os.path.join(artifacts_dir, f"{table_name}_0.parquet")
    columns_meta = extract_schema_metadata(conn, first_chunk, table_name)
    return {
        "tables": {table_name: {"file_pattern": f"{table_name}_*.parquet", "total_rows": total_rows}},
        "columns": columns_meta,
        "relationships": [],
    }


def process_multi_strategy(conn, csv_paths, artifacts_dir, temp_dir, status):
    status.write("🧩 Strategy: Keep separate tables so related CSV files can be joined later.")
    tables_metadata = {}
    all_columns_meta = []
    progress = status.progress(0.0)

    for index, csv_file in enumerate(csv_paths, start=1):
        raw_name = os.path.basename(csv_file)
        table_name = sanitize_table_name(raw_name)
        temp_parquet = os.path.join(temp_dir, f"{table_name}_temp.parquet")
        input_path = csv_file.replace(os.sep, "/")
        status.write(f"⚙️ Processing table {index}/{len(csv_paths)}: `{table_name}`")

        try:
            conn.execute(
                f"""
                COPY (
                    SELECT * FROM read_csv_auto('{input_path}', sample_size=100000)
                ) TO '{temp_parquet.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
                """
            )
        except Exception as exc:
            logging.info("DuckDB parse failed for %s: %s", table_name, exc)
            status.write(f"⚠️ DuckDB parsing failed for `{table_name}`. Falling back to Pandas chunk reads.")
            temp_chunks = []
            header_df = pd.read_csv(csv_file, nrows=0, encoding_errors="replace")
            clean_cols = clean_column_names(header_df.columns)
            with pd.read_csv(
                csv_file,
                chunksize=200_000,
                encoding_errors="replace",
                on_bad_lines="skip",
            ) as reader:
                for chunk_index, chunk in enumerate(reader):
                    chunk.columns = clean_cols
                    chunk_path = os.path.join(temp_dir, f"{table_name}_{chunk_index}.parquet")
                    chunk.to_parquet(chunk_path, engine="pyarrow", index=False)
                    temp_chunks.append(chunk_path)

            if not temp_chunks:
                raise ValueError(f"No readable rows were found in {raw_name}.")

            chunk_pattern = os.path.join(temp_dir, f"{table_name}_*.parquet").replace(os.sep, "/")
            conn.execute(
                f"""
                COPY (
                    SELECT * FROM read_parquet('{chunk_pattern}')
                ) TO '{temp_parquet.replace(os.sep, "/")}' (FORMAT 'PARQUET', CODEC 'ZSTD')
                """
            )

        total_rows = get_parquet_row_count(temp_parquet)
        file_size_mb = os.path.getsize(temp_parquet) / (1024 * 1024)

        if file_size_mb < CHUNK_SIZE_MB:
            final_path = os.path.join(artifacts_dir, f"{table_name}_0.parquet")
            os.rename(temp_parquet, final_path)
        else:
            num_chunks = max(1, math.ceil(file_size_mb / CHUNK_SIZE_MB))
            rows_per_chunk = max(1, math.ceil(total_rows / num_chunks))
            status.write(
                f"✂️ Splitting `{table_name}` into about {num_chunks} Parquet chunks for cloud-friendly reads."
            )
            split_parquet_to_chunks(temp_parquet, table_name, rows_per_chunk, artifacts_dir, status)
            os.remove(temp_parquet)

        first_chunk = os.path.join(artifacts_dir, f"{table_name}_0.parquet")
        all_columns_meta.extend(extract_schema_metadata(conn, first_chunk, table_name))
        tables_metadata[table_name] = {
            "file_pattern": f"{table_name}_*.parquet",
            "total_rows": total_rows,
        }
        progress.progress(index / len(csv_paths))

    relationships = detect_relationships(conn, tables_metadata, artifacts_dir, status)
    return {
        "tables": tables_metadata,
        "columns": all_columns_meta,
        "relationships": relationships,
    }


def write_artifact_zip(artifacts_dir):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(artifacts_dir):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                arcname = os.path.relpath(full_path, artifacts_dir)
                zf.write(full_path, arcname)
    buffer.seek(0)
    return buffer.getvalue()


def build_local_run_readme():
    return """# Local Execution Package

This package lets you run the same app locally, which is helpful when your sanitized or dummy dataset is too large for Streamlit Community Cloud browser uploads.

## What is included

- `streamlit_app.py`: the app code
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: upload-size setting used by the app
- `.streamlit/secrets.toml.example`: example secrets file
- `run_local_app.bat`: Windows launcher
- `run_local_app.sh`: Mac/Linux launcher

## Local setup steps

1. Install Python 3.12 if you do not already have it.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   Windows:

   `python -m venv .venv`

   `.venv\\Scripts\\activate`

   Mac/Linux:

   `python3 -m venv .venv`

   `source .venv/bin/activate`

4. Install dependencies:

   `pip install -r requirements.txt`

5. Add your API key either in the app sidebar or by copying `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and filling in one or more keys.

6. Run the app:

   `streamlit run streamlit_app.py`

   Or use the included launcher script:

   Windows:

   `run_local_app.bat`

   Mac/Linux:

   `bash run_local_app.sh`

7. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## API keys

You can provide your key either in the app sidebar or via a local secrets file:

Create `.streamlit/secrets.toml` with one or more of the following keys:

```toml
DEEPSEEK_API_KEY = "your-key-here"
OPENAI_API_KEY = "your-key-here"
XAI_API_KEY = "your-key-here"
```

## Recommended usage

- Use `Keep files as separate tables` for multi-entity LMS exports such as users, enrollments, discussion posts, session history, and content objects.
- Use `Merge all files into one table` only when all uploaded CSV files have the same shape and should become one combined dataset.
- Local execution is the better path for larger files because it avoids Community Cloud browser-upload limits and tighter runtime ceilings.
- If the app warns that your uploaded files have mismatched columns, switch to `Keep files as separate tables`.
"""


def build_local_secrets_example():
    return """DEEPSEEK_API_KEY = "your-key-here"
OPENAI_API_KEY = "your-key-here"
XAI_API_KEY = "your-key-here"
"""


def build_windows_launcher():
    return """@echo off
setlocal

if not exist .venv (
  python -m venv .venv
)

call .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
"""


def build_unix_launcher():
    return """#!/usr/bin/env bash
set -e

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
"""


def write_local_execution_zip():
    app_path = os.path.abspath(__file__)
    repo_root = os.path.dirname(app_path)
    requirements_path = os.path.join(repo_root, "requirements.txt")
    config_path = os.path.join(repo_root, ".streamlit", "config.toml")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(app_path):
            zf.write(app_path, "streamlit_app.py")
        if os.path.exists(requirements_path):
            zf.write(requirements_path, "requirements.txt")
        if os.path.exists(config_path):
            zf.write(config_path, ".streamlit/config.toml")
        zf.writestr("README_LOCAL.md", build_local_run_readme())
        zf.writestr(".streamlit/secrets.toml.example", build_local_secrets_example())
        zf.writestr("run_local_app.bat", build_windows_launcher())
        zf.writestr("run_local_app.sh", build_unix_launcher())
    buffer.seek(0)
    return buffer.getvalue()


def process_uploaded_files(uploaded_files, strategy):
    if not uploaded_files:
        raise ValueError("Please upload at least one CSV or ZIP file.")

    root, bundle_dir, upload_dir, artifacts_dir = build_session_paths()
    robust_rmtree(root)
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    status = st.status("🚀 Processing uploaded data...", expanded=True)
    start_time = time.time()
    conn = None

    try:
        total_uploaded_mb = 0.0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(upload_dir, uploaded_file.name)
            file_bytes = uploaded_file.getbuffer()
            total_uploaded_mb += len(file_bytes) / (1024 * 1024)
            with open(file_path, "wb") as handle:
                handle.write(file_bytes)
            status.write(f"📥 Saved upload `{uploaded_file.name}`")

            if uploaded_file.name.lower().endswith(".zip"):
                status.write(f"📂 Extracting `{uploaded_file.name}`")
                safe_extract(file_path, upload_dir)
                os.remove(file_path)

        status.write(f"📦 Total uploaded size: {total_uploaded_mb:.1f} MB")
        if total_uploaded_mb > UPLOAD_WARNING_MB:
            status.write(
                "⚠️ This is a large browser upload for Streamlit Community Cloud. Processing may still fail if the app hits memory or time limits."
            )

        csv_paths = get_all_csvs(upload_dir)
        if not csv_paths:
            raise ValueError("No CSV files were found after upload and extraction.")

        status.write(f"📊 Found {len(csv_paths)} CSV file(s). Converting them to chunked Parquet artifacts.")
        header_map, signature_map = inspect_csv_headers(csv_paths)
        if strategy == "merge" and len(signature_map) > 1:
            preview_groups = []
            for signature_files in list(signature_map.values())[:3]:
                preview_groups.append(", ".join(signature_files[:3]))
            raise ValueError(
                "Merge mode is not recommended for this upload because the CSV files do not share the same column structure. "
                "Please switch to 'Keep files as separate tables'. "
                f"Detected multiple header patterns across files such as: {' | '.join(preview_groups)}"
            )
        conn = duckdb.connect()

        if strategy == "merge":
            result = process_merge_strategy(conn, csv_paths, artifacts_dir, upload_dir, status)
        else:
            result = process_multi_strategy(conn, csv_paths, artifacts_dir, upload_dir, status)

        metadata = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "chunk_size_mb": CHUNK_SIZE_MB,
            "source_mode": "cloud_upload",
            "tables": result["tables"],
            "columns": result["columns"],
            "relationships": result["relationships"],
        }
        metadata["table_overlaps"] = analyze_table_overlaps(metadata)

        metadata_path = os.path.join(artifacts_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        st.session_state.dataset_ready = True
        st.session_state.bundle_dir = bundle_dir
        st.session_state.artifacts_dir = artifacts_dir
        st.session_state.metadata = metadata
        st.session_state.starter_questions = []
        st.session_state.messages = [
            {"role": "assistant", "content": "Dataset processed. Ask a question about the uploaded data."}
        ]
        st.session_state.processing_summary = {
            "csv_count": len(csv_paths),
            "uploaded_mb": round(total_uploaded_mb, 1),
            "tables": len(metadata["tables"]),
            "columns": len(metadata["columns"]),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }

        status.update(label="✅ Processing complete", state="complete", expanded=False)
        st.success("Dataset is ready for chat and query.")
        execute_validated_sql.clear()
        return metadata
    except Exception as exc:
        status.update(label="❌ Processing failed", state="error", expanded=True)
        robust_rmtree(root)
        raise exc
    finally:
        if conn is not None:
            conn.close()
        gc.collect()


def build_table_inventory(metadata, artifacts_dir):
    lines = []
    for table_name, table_info in metadata.get("tables", {}).items():
        pattern = os.path.join(artifacts_dir, table_info["file_pattern"]).replace("\\", "/")
        lines.append(
            f"- {table_name}: read_parquet('{pattern}') | rows={table_info.get('total_rows', 'unknown')}"
        )
    return "\n".join(lines)


def sanitize_sql_for_display(sql_text, metadata, artifacts_dir):
    if not sql_text:
        return sql_text

    clean_sql = sql_text
    for table_name, table_info in metadata.get("tables", {}).items():
        pattern = os.path.join(artifacts_dir, table_info["file_pattern"]).replace("\\", "/")
        replacement = f"ARTIFACT::{table_name}"
        clean_sql = clean_sql.replace(pattern, replacement)
    return clean_sql


def get_stored_relationships(metadata):
    stored = metadata.get("relationships", [])
    return [
        f"- {item['from_table']}.{item['from_column']} -> {item['to_table']}.{item['to_column']}"
        for item in stored
    ]


def get_overlap_relationship_hints(metadata):
    hints = []
    for item in metadata.get("table_overlaps", []):
        if item.get("shared_key_count", 0) > 0:
            key_labels = ", ".join(
                f"{shared['left_column']} / {shared['right_column']}"
                for shared in item.get("shared_key_columns", [])[:5]
            )
            hints.append(
                f"- {item['left_table']} and {item['right_table']} share likely key columns: {key_labels}"
            )
        elif item.get("shared_column_count", 0) > 1:
            col_labels = ", ".join(
                f"{shared['left_column']} / {shared['right_column']}"
                for shared in item.get("shared_columns", [])[:5]
            )
            hints.append(
                f"- {item['left_table']} and {item['right_table']} share columns that may support comparisons: {col_labels}"
            )
    return hints


def build_relationship_context(metadata):
    relationships = get_stored_relationships(metadata)
    overlap_hints = get_overlap_relationship_hints(metadata)

    blocks = []
    if relationships:
        blocks.append(
            "\nKNOWN TABLE RELATIONSHIPS (prefer these JOIN paths when relevant):\n"
            + "\n".join(relationships)
            + "\n"
        )
    if overlap_hints:
        blocks.append(
            "\nTABLE OVERLAP HINTS (shared columns seen during preprocessing):\n"
            + "\n".join(overlap_hints)
            + "\n"
        )
    return "".join(blocks)


def build_context_block(metadata, question):
    columns = metadata.get("columns", [])
    if len(columns) <= FULL_SCHEMA_THRESHOLD:
        selected = columns
    else:
        question_tokens = tokenize(question)
        scored = []
        for column in columns:
            name_tokens = tokenize(column["name"])
            table_tokens = tokenize(column["table"])
            desc_tokens = tokenize(column["description"])
            overlap = len(question_tokens & name_tokens) * 5
            overlap += len(question_tokens & table_tokens) * 3
            overlap += len(question_tokens & desc_tokens)
            if column["name"].lower() in question.lower():
                overlap += 8
            if column["table"].lower() in question.lower():
                overlap += 6
            scored.append((overlap, column))
        selected = [item[1] for item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:12]]

    lines = ["RELEVANT SCHEMA:"]
    for item in selected:
        lines.append(
            f"- Table `{item['table']}` | Column `{item['name']}` | Type `{item['type']}` | Samples `{item['description'].split('Samples: ', 1)[-1]}`"
        )
    return "\n".join(lines)


def _strip_markdown_sql(text):
    clean = (text or "").strip()
    if clean.startswith("```sql"):
        clean = clean[6:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def strip_leading_sql_comments(sql_text):
    if not sql_text:
        return sql_text

    cleaned = sql_text.lstrip()

    while True:
        if cleaned.startswith("--"):
            newline_index = cleaned.find("\n")
            if newline_index == -1:
                return ""
            cleaned = cleaned[newline_index + 1 :].lstrip()
            continue

        if cleaned.startswith("/*"):
            end_index = cleaned.find("*/")
            if end_index == -1:
                return ""
            cleaned = cleaned[end_index + 2 :].lstrip()
            continue

        break

    return cleaned


def find_table_name_in_question(question, metadata):
    question_norm = normalize_identifier(question)
    best_match = None
    best_score = 0

    for table_name in metadata.get("tables", {}).keys():
        candidates = candidate_entity_names(table_name) | {normalize_identifier(table_name)}
        for candidate in candidates:
            if candidate and candidate in question_norm and len(candidate) > best_score:
                best_match = table_name
                best_score = len(candidate)

    return best_match


def find_explicit_table_mentions(question, metadata):
    question_norm = normalize_identifier(question)
    matches = []
    for table_name in metadata.get("tables", {}).keys():
        candidates = candidate_entity_names(table_name) | {normalize_identifier(table_name)}
        if any(candidate and candidate in question_norm for candidate in candidates):
            matches.append(table_name)
    return sorted(set(matches))


def build_table_overview_rows(metadata):
    rows = []
    table_columns = {}
    for column in metadata.get("columns", []):
        table_columns.setdefault(column["table"], []).append(column["name"])

    for table_name, table_info in metadata.get("tables", {}).items():
        rows.append(
            {
                "table_name": table_name,
                "row_count": table_info.get("total_rows"),
                "column_count": len(table_columns.get(table_name, [])),
                "appears_to_represent": infer_table_description(table_name, table_columns.get(table_name, [])),
            }
        )
    return pd.DataFrame(rows)


def infer_table_description(table_name, column_names):
    name = table_name.lower()
    column_set = {col.lower() for col in column_names}

    if "discussion" in name:
        return "Forum or discussion activity such as posts, replies, threads, or engagement details."
    if "enrollment" in name:
        return "User enrollments, roles, and organization or course membership records."
    if "session" in name or "history" in name:
        return "Login sessions or usage-history events with timestamps and access activity."
    if "user" in name:
        return "User account, profile, status, and time-based account activity fields."
    if "content" in name:
        return "Learning content objects with type, status, dates, or hierarchy metadata."
    if "userid" in column_set and "rolename" in column_set:
        return "Records keyed by user with role or participation metadata."
    return "A processed table from the uploaded dataset."


def build_columns_dataframe(metadata, table_name):
    rows = []
    for column in metadata.get("columns", []):
        if column["table"] != table_name:
            continue
        samples = column["description"].split("Samples: ", 1)[-1]
        rows.append(
            {
                "column_name": column["name"],
                "data_type": column["type"],
                "sample_values": samples,
            }
        )
    return pd.DataFrame(rows)


def handle_metadata_question(question, metadata):
    q = question.lower()
    table_name = find_table_name_in_question(question, metadata)
    explicit_tables = find_explicit_table_mentions(question, metadata)
    analysis_markers = [
        " then ",
        " before ",
        " using ",
        " together",
        " summary",
        " diagnostic",
        " percentage",
        " percent",
        " average",
        " order ",
        " exclude ",
        " compare",
        " group by",
        " one row per",
        " posted",
        " include",
        " join",
        " metric",
        " metrics",
    ]
    has_analysis_signal = any(marker in f" {q} " for marker in analysis_markers)

    if not has_analysis_signal and ((("what tables" in q or "which tables" in q) and ("represent" in q or "included" in q)) or (
        "what tables are included" in q
    )):
        df = build_table_overview_rows(metadata)
        lines = ["Here are the processed tables and what they appear to represent:"]
        for _, row in df.iterrows():
            lines.append(
                f"- `{row['table_name']}`: about {row['row_count']:,} rows, {row['column_count']} columns, and appears to represent {row['appears_to_represent'].lower()}"
            )
        return {
            "mode": "metadata",
            "title": "Answered directly from processed metadata",
            "dataframe": df,
            "answer": "\n".join(lines),
        }

    if not has_analysis_signal and "how many total rows" in q and "each table" in q:
        rows = []
        for table, info in metadata.get("tables", {}).items():
            rows.append({"table_name": table, "row_count": info.get("total_rows", 0)})
        df = pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
        lines = ["Row counts by table:"]
        for _, row in df.iterrows():
            lines.append(f"- `{row['table_name']}`: {row['row_count']:,} rows")
        return {
            "mode": "metadata",
            "title": "Answered directly from processed metadata",
            "dataframe": df,
            "answer": "\n".join(lines),
        }

    if not has_analysis_signal and table_name and len(explicit_tables) <= 1 and (
        "which columns" in q or "what columns" in q or "available in" in q
    ):
        df = build_columns_dataframe(metadata, table_name)
        if df.empty:
            return None
        answer = (
            f"The `{table_name}` table has {len(df)} columns. "
            f"The table below lists each column, its type, and sample values captured during preprocessing."
        )
        return {
            "mode": "metadata",
            "title": f"Answered directly from metadata for `{table_name}`",
            "dataframe": df,
            "answer": answer,
        }

    if not has_analysis_signal and "relationships" in q and ("detected" in q or "found" in q or "between tables" in q):
        rows = metadata.get("relationships", [])
        if rows:
            df = pd.DataFrame(rows)
            answer = "These are the table relationships detected during preprocessing."
        else:
            df = pd.DataFrame(columns=["from_table", "from_column", "to_table", "to_column"])
            answer = "No explicit table relationships were detected during preprocessing."
        return {
            "mode": "metadata",
            "title": "Answered directly from processed metadata",
            "dataframe": df,
            "answer": answer,
        }

    if not has_analysis_signal and ("common" in q or "shared" in q or "overlap" in q) and ("columns" in q or "tables" in q):
        rows = []
        for item in metadata.get("table_overlaps", []):
            rows.append(
                {
                    "left_table": item["left_table"],
                    "right_table": item["right_table"],
                    "shared_column_count": item["shared_column_count"],
                    "shared_key_count": item["shared_key_count"],
                    "shared_keys_preview": ", ".join(
                        shared["left_column"] for shared in item.get("shared_key_columns", [])[:5]
                    ),
                }
            )
        df = pd.DataFrame(rows)
        answer = (
            "The table below shows which processed tables share columns or likely key fields. "
            "Pairs with shared key columns are the best candidates for joins."
        )
        return {
            "mode": "metadata",
            "title": "Answered directly from processed metadata",
            "dataframe": df,
            "answer": answer,
        }

    return None


def get_sql_query(
    question,
    table_inventory,
    context_block,
    client,
    model_name,
    conversation_context,
    relationship_context,
    explicitly_named_tables="",
):
    today = datetime.date.today().strftime("%Y-%m-%d")
    prompt = f"""You are an expert DuckDB SQL analyst specializing in education and LMS data.
Current Date: {today}

AVAILABLE TABLES & FILES:
{table_inventory}

{context_block}
{relationship_context}
{explicitly_named_tables}
{conversation_context}
USER QUESTION:
"{question}"

SQL RULES:
1. Output only valid DuckDB SQL. No markdown. No explanation.
2. Use read_parquet('path/pattern') in FROM clauses.
3. Use the exact table and column names from the schema.
4. Use GROUP BY when mixing aggregates with non-aggregates.
5. Default to LIMIT 50 unless the user asks for more.
6. Use ILIKE for fuzzy text search.
7. Use LEFT JOIN when the question asks for missing items.
8. Use date_trunc for month or quarter grouping.
9. Exclude nulls where appropriate.
10. When joining tables, prefer the known relationships supplied above.
11. First identify the grain of the answer before counting. Count rows or distinct IDs at that grain to avoid accidental duplication from joins.
12. For questions like "how many users", "how many authors", or "how many students", prefer COUNT(DISTINCT user-like ID).
13. For questions like "how many enrollments are associated with users who ...", filter the enrollments table to the relevant users, then count enrollment rows from the enrollments table.
14. When grouping activity by role or category, count the activity/event ID from the activity table, not the user ID unless the user asked for distinct users.
15. If a join can multiply rows, use a subquery or COUNT(DISTINCT ...) when needed.
16. If the user explicitly names tables, use all of those tables unless one is clearly irrelevant to the requested output.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You generate DuckDB SQL only. Use read_parquet() for data access.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _strip_markdown_sql(response.choices[0].message.content)


def fix_sql_query(
    question,
    failed_sql,
    error_msg,
    table_inventory,
    context_block,
    client,
    model_name,
    relationship_context,
    explicitly_named_tables="",
):
    prompt = f"""The following DuckDB SQL query failed.

FAILED SQL:
{failed_sql}

ERROR MESSAGE:
{error_msg}

AVAILABLE TABLES & FILES:
{table_inventory}

{context_block}
{relationship_context}
{explicitly_named_tables}
ORIGINAL QUESTION:
"{question}"

Fix the SQL. Use only read_parquet() for data access. Output only corrected SQL.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a SQL debugger. Output only corrected DuckDB SQL."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return _strip_markdown_sql(response.choices[0].message.content)


def validate_sql(sql_query, artifacts_dir):
    clean_sql = strip_leading_sql_comments(sql_query).strip().rstrip(";").strip()
    if not clean_sql:
        raise ValueError("Security alert: empty SQL query.")

    sql_without_strings = re.sub(r"'[^']*'", "", clean_sql)
    if ";" in sql_without_strings:
        raise ValueError("Security alert: multiple SQL statements are not allowed.")

    upper_sql = clean_sql.upper().lstrip()
    if not (upper_sql.startswith("SELECT") or upper_sql.startswith("DESCRIBE") or upper_sql.startswith("WITH")):
        raise ValueError("Security alert: only SELECT, DESCRIBE, or WITH queries are permitted.")

    for keyword in DANGEROUS_KEYWORDS:
        if re.search(r"\b" + keyword + r"\b", upper_sql):
            raise ValueError(f"Security alert: forbidden keyword detected: {keyword}")

    allowed_prefix = os.path.realpath(artifacts_dir).replace("\\", "/")
    string_literals = re.findall(r"'([^']*)'", clean_sql.replace("''", ""))
    for literal in string_literals:
        if literal.lower().startswith("http://") or literal.lower().startswith("https://") or literal.lower().startswith("s3://"):
            raise ValueError("Security alert: remote URLs are blocked in this cloud-safe version.")

        looks_like_path = (
            "/" in literal
            or "\\" in literal
            or literal.endswith(".parquet")
            or literal.endswith(".csv")
        )
        if looks_like_path:
            normalized = literal.replace("\\", "/")
            if "*" in normalized:
                target = os.path.realpath(os.path.dirname(normalized)).replace("\\", "/")
            else:
                target = os.path.realpath(normalized).replace("\\", "/")
            if not target.startswith(allowed_prefix):
                raise ValueError("Security alert: the query references files outside the processed dataset.")

    if "LIMIT" not in upper_sql:
        clean_sql = f"SELECT * FROM ({clean_sql}) AS _limited LIMIT {HARD_ROW_LIMIT}"
    return clean_sql


@st.cache_data(show_spinner=False, ttl=3600)
def execute_validated_sql(clean_sql, artifacts_dir):
    conn = duckdb.connect(database=":memory:")
    try:
        conn.execute("SET autoinstall_known_extensions=false")
        conn.execute("SET autoload_known_extensions=false")
        return conn.execute(clean_sql).df()
    finally:
        conn.close()


def summarize_answer(question, df, client, model_name, pii_redaction_enabled):
    total_rows = len(df)
    summary_df = df.head(MAX_ROWS_FOR_SUMMARY) if total_rows > MAX_ROWS_FOR_SUMMARY else df.copy()
    redacted_cols = []
    if pii_redaction_enabled:
        summary_df, redacted_cols = redact_pii(summary_df)

    truncation_note = ""
    if total_rows > MAX_ROWS_FOR_SUMMARY:
        truncation_note = f"\nOnly the first {MAX_ROWS_FOR_SUMMARY} rows of {total_rows} are shown to the model."

    redaction_note = ""
    if redacted_cols:
        redaction_note = f"\nThese columns were redacted before summarization: {', '.join(redacted_cols)}."

    prompt = f"""You are an expert education data analyst summarizing query results for a non-technical audience.

User question:
{question}

Query results:
{summary_df.to_string(index=False)}{truncation_note}{redaction_note}

Rules:
1. Keep the summary concise and practical.
2. Call out concrete counts, rates, and patterns.
3. If relevant, suggest a sensible next question.
4. Do not speculate about redacted values.
"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def attempt_visualization(df):
    try:
        if df.empty:
            return

        if len(df) < 2:
            return

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if not date_cols:
            for col_name in list(categorical_cols):
                converted = pd.to_datetime(df[col_name], errors="coerce", format="mixed", utc=True)
                if converted.notna().mean() > 0.8:
                    df = df.copy()
                    df[col_name] = converted
                    date_cols.append(col_name)
                    categorical_cols.remove(col_name)
                    break

        if len(df.columns) <= 2 and len(df) <= 5 and len(numeric_cols) >= 1:
            return

        st.caption("Auto-visualization")
        if date_cols and numeric_cols:
            st.line_chart(df.set_index(date_cols[0])[numeric_cols[:3]].sort_index())
            return
        if categorical_cols and numeric_cols:
            chart_df = df.head(25).set_index(categorical_cols[0])[numeric_cols[0]]
            st.bar_chart(chart_df)
            return
        if len(numeric_cols) >= 2:
            st.scatter_chart(df, x=numeric_cols[0], y=numeric_cols[1])
            return
        if len(numeric_cols) == 1:
            series = df[numeric_cols[0]].dropna()
            if 0 < series.nunique() <= 50:
                st.bar_chart(series.value_counts().sort_index())
    except Exception as exc:
        logging.info("Visualization skipped: %s", exc)


def _format_chat_export():
    lines = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"### {role}\n{msg['content']}")
    return "\n\n---\n\n".join(lines)


def generate_starter_questions(metadata, client, model_name):
    columns = metadata.get("columns", [])
    lines = []
    for item in columns[:40]:
        lines.append(f"- {item['table']}.{item['name']} ({item['type']})")

    prompt = f"""Generate 5 short starter questions for a non-technical user exploring an education dataset.
Use only the schema below.

SCHEMA:
{chr(10).join(lines)}

Output one question per line and nothing else.
"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        raw_lines = response.choices[0].message.content.splitlines()
        cleaned = []
        for line in raw_lines:
            line = re.sub(r"^\s*[-0-9.]+\s*", "", line).strip()
            if line:
                cleaned.append(line)
        return cleaned[:5]
    except Exception as exc:
        logging.info("Starter question generation failed: %s", exc)
        return []


def render_sidebar():
    with st.sidebar:
        st.title("Configuration")

        provider_names = list(PROVIDER_CONFIG.keys())
        provider_name = st.selectbox(
            "LLM provider",
            provider_names,
            index=provider_names.index(DEFAULT_PROVIDER_NAME),
        )
        provider_config = PROVIDER_CONFIG[provider_name]

        api_key = ""
        try:
            if provider_config["secret_key"] in st.secrets:
                api_key = st.secrets[provider_config["secret_key"]]
        except Exception:
            pass

        if api_key:
            st.success(f"{provider_name} key loaded from secrets")
        else:
            api_key = st.text_input(f"{provider_name} API key", type="password")

        model_name = st.text_input("Model name", value=provider_config["default_model"])

        st.markdown("---")
        st.session_state.pii_redaction = st.toggle(
            "PII redaction before summarization",
            value=st.session_state.pii_redaction,
            help="When enabled, matching columns are replaced with [REDACTED] before rows are sent to the LLM.",
        )

        st.markdown("---")
        if st.button("Clear processed dataset", use_container_width=True):
            reset_dataset_state()
            st.rerun()

        if len(st.session_state.messages) > 1:
            st.download_button(
                "Export chat history",
                data=_format_chat_export(),
                file_name="chat_history.md",
                mime="text/markdown",
                use_container_width=True,
            )

        return provider_name, api_key, model_name, provider_config


def render_processing_ui():
    st.subheader("1. Upload and preprocess")
    st.caption(
        "This cloud version assumes the source data is already sanitized or dummy data. It processes uploads into chunked local Parquet files inside the running Streamlit session."
    )

    with st.expander("How to use this app", expanded=False):
        st.markdown(
            """
            1. Choose your LLM provider in the sidebar.
            2. Upload sanitized or dummy CSV files. You can also upload ZIP files that contain CSVs.
            3. Pick a preprocessing strategy:
               - `Merge all files into one table` for same-shape files that should become one dataset.
               - `Keep files as separate tables` for related LMS exports such as users, enrollments, discussion posts, and content objects.
            4. Click `Process uploads` and wait for the dataset summary to appear.
            5. Review the processed table list and then ask a specific question in plain English.
            6. Start with counts, comparisons, and date trends before moving into more complex joins.

            Example questions:
            - How many users have posted discussion posts?
            - How many discussion post authors are also in the users table?
            - Show discussion post counts by user role.
            - How many enrollments are associated with users who have posted discussions?
            - Which content object types are included in the dataset?
            - Are there any content objects marked as deleted?
            - Show the count of logins or events by month.

            Tips:
            - Short, specific questions work best.
            - For grouped answers, name the grouping field you care about, such as role, course, month, or completion status.
            - This app is intended for exploratory analysis of sanitized data, not for raw confidential data.
            """
        )

    with st.expander("Run this app locally for larger files", expanded=False):
        st.markdown(
            """
            Local execution is the best option when your sanitized dataset is too large for Streamlit Community Cloud uploads or when you want more control over runtime resources.

            Local quick start:
            1. Download the local execution package below.
            2. Unzip it on your machine.
            3. Install Python 3.12.
            4. Run `pip install -r requirements.txt`
            5. Run `streamlit run streamlit_app.py`

            Included in the download:
            - the app code
            - `requirements.txt`
            - `.streamlit/config.toml`
            - `.streamlit/secrets.toml.example`
            - Windows and Mac/Linux launcher scripts
            - a local README

            You can provide your API key in the sidebar or by creating `.streamlit/secrets.toml` locally.
            Multi-table LMS exports should usually use `Keep files as separate tables`.
            """
        )
        st.download_button(
            "Download local execution package",
            data=write_local_execution_zip(),
            file_name="cloud_rag_data_assistant_local.zip",
            mime="application/zip",
        )

    strategy = st.radio(
        "Preprocessing strategy",
        options=["merge", "separate"],
        index=1,
        format_func=lambda value: "Merge all files into one table" if value == "merge" else "Keep files as separate tables",
        horizontal=True,
    )

    uploaded_files = st.file_uploader(
        "Upload CSV or ZIP files",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="ZIP files are extracted in the app. Large browser uploads may still run into Streamlit Community Cloud limits.",
    )

    if uploaded_files:
        total_mb = sum(len(item.getbuffer()) for item in uploaded_files) / (1024 * 1024)
        st.info(f"Selected {len(uploaded_files)} file(s), about {total_mb:.1f} MB total.")

    if st.button("Process uploads", type="primary", use_container_width=True):
        process_uploaded_files(uploaded_files, strategy)
        st.rerun()


def render_dataset_summary():
    if not st.session_state.dataset_ready or not st.session_state.metadata:
        return

    metadata = st.session_state.metadata
    summary = st.session_state.processing_summary
    st.subheader("2. Processed dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CSV files", summary.get("csv_count", 0))
    col2.metric("Tables", len(metadata.get("tables", {})))
    col3.metric("Columns", len(metadata.get("columns", [])))
    col4.metric("Upload size (MB)", summary.get("uploaded_mb", 0.0))

    with st.expander("Artifact summary", expanded=False):
        st.json(
            {
                "processing_summary": summary,
                "tables": metadata.get("tables", {}),
                "relationships": metadata.get("relationships", []),
                "table_overlaps": metadata.get("table_overlaps", []),
            }
        )

    artifact_zip = write_artifact_zip(st.session_state.artifacts_dir)
    st.download_button(
        "Download processed artifacts",
        data=artifact_zip,
        file_name="processed_artifacts.zip",
        mime="application/zip",
    )


def render_chat_ui(provider_name, api_key, model_name, provider_config):
    st.subheader("3. Ask questions")
    if not st.session_state.dataset_ready:
        st.info("Process a dataset first.")
        return

    if not api_key:
        st.warning("Add an API key in the sidebar to enable SQL generation and summarization.")
        return

    client = OpenAI(api_key=api_key, base_url=provider_config["base_url"])
    metadata = st.session_state.metadata
    artifacts_dir = st.session_state.artifacts_dir
    table_inventory = build_table_inventory(metadata, artifacts_dir)
    relationship_context = build_relationship_context(metadata)

    if not st.session_state.starter_questions:
        with st.spinner("Generating starter questions..."):
            st.session_state.starter_questions = generate_starter_questions(metadata, client, model_name)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if st.session_state.starter_questions and len(st.session_state.messages) <= 2:
        st.caption("Suggested questions")
        selected_question = None
        for question in st.session_state.starter_questions:
            if st.button(question, key=f"starter_{question}", use_container_width=True):
                selected_question = question
        if selected_question:
            st.session_state.messages.append({"role": "user", "content": selected_question})
            st.session_state.starter_questions = []
            st.rerun()

    user_input = st.chat_input("Ask about the processed dataset")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    st.session_state.starter_questions = []

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=False) as status:
            try:
                direct_response = handle_metadata_question(user_input, metadata)
                if direct_response is not None:
                    status.write(direct_response["title"])
                    df = direct_response["dataframe"]
                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                        st.download_button(
                            "Download results as CSV",
                            data=df.to_csv(index=False),
                            file_name="metadata_results.csv",
                            mime="text/csv",
                        )
                    status.update(label="✅ Answer ready", state="complete")
                    st.write(direct_response["answer"])
                    st.session_state.messages.append({"role": "assistant", "content": direct_response["answer"]})
                    return

                context_block = build_context_block(metadata, user_input)
                explicit_tables = find_explicit_table_mentions(user_input, metadata)
                explicit_table_block = ""
                if explicit_tables:
                    explicit_table_block = (
                        "USER-REFERENCED TABLES:\n"
                        + "\n".join(f"- {table_name}" for table_name in explicit_tables)
                        + "\n"
                    )
                conversation_context = ""
                msgs = st.session_state.messages
                if len(msgs) >= 4 and msgs[-3]["role"] == "user" and msgs[-2]["role"] == "assistant":
                    conversation_context = (
                        f'\nPREVIOUS QUESTION: "{msgs[-3]["content"]}"\n'
                        f'PREVIOUS ANSWER SUMMARY: "{msgs[-2]["content"][:500]}"\n'
                    )

                status.write(f"Generating SQL with {provider_name} ({model_name})")
                sql = get_sql_query(
                    user_input,
                    table_inventory,
                    context_block,
                    client,
                    model_name,
                    conversation_context,
                    relationship_context,
                    explicit_table_block,
                )
                display_sql = sanitize_sql_for_display(sql, metadata, artifacts_dir)
                st.code(display_sql, language="sql")

                clean_sql = validate_sql(sql, artifacts_dir)
                status.write("Executing query")

                try:
                    df = execute_validated_sql(clean_sql, artifacts_dir)
                except Exception as first_error:
                    logging.info("First SQL attempt failed: %s", first_error)
                    status.write("Retrying with SQL repair")
                    sql_retry = fix_sql_query(
                        user_input,
                        sql,
                        str(first_error),
                        table_inventory,
                        context_block,
                        client,
                        model_name,
                        relationship_context,
                        explicit_table_block,
                    )
                    retry_display_sql = sanitize_sql_for_display(sql_retry, metadata, artifacts_dir)
                    st.code(retry_display_sql, language="sql")
                    clean_sql = validate_sql(sql_retry, artifacts_dir)
                    df = execute_validated_sql(clean_sql, artifacts_dir)

                if len(df) >= HARD_ROW_LIMIT:
                    st.warning(
                        f"Results were capped at {HARD_ROW_LIMIT:,} rows. Add a narrower filter or aggregation for more precise answers."
                    )

                st.dataframe(df, use_container_width=True)
                if not df.empty:
                    st.download_button(
                        "Download results as CSV",
                        data=df.to_csv(index=False),
                        file_name="query_results.csv",
                        mime="text/csv",
                    )
                    attempt_visualization(df)

                if not df.empty:
                    status.write("Summarizing results")
                    answer = summarize_answer(
                        user_input,
                        df,
                        client,
                        model_name,
                        st.session_state.pii_redaction,
                    )
                else:
                    answer = "The query returned no rows. Try broadening the question or asking for a different slice of the data."

                status.update(label="✅ Answer ready", state="complete")
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except ValueError as exc:
                status.update(label="🔒 Blocked", state="error")
                st.error(str(exc))
            except Exception as exc:
                logging.exception("Unhandled chat error")
                status.update(label="❌ Failed", state="error")
                st.error(str(exc))


def main():
    ensure_session_state()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.expander("Important cloud notes", expanded=True):
        st.markdown(
            f"""
            - This unified version is designed for **sanitized or dummy data only**.
            - It preprocesses and queries data **inside a single Streamlit Community Cloud session**.
            - Chunking reduces Parquet file size and helps query performance, but it **does not remove Streamlit Cloud resource limits**.
            - Streamlit documents a default upload limit of **200 MB**, configurable through `.streamlit/config.toml`, and approximate Community Cloud resource ceilings of **up to about 2.7 GB memory and 50 GB storage** as of February 2024.
            - In practice, browser upload and execution time will become the real bottlenecks before truly huge local-style workloads. This version is the safest realistic cloud adaptation, not a full replacement for your local heavy-ingest setup.
            """
        )

    provider_name, api_key, model_name, provider_config = render_sidebar()
    render_processing_ui()
    render_dataset_summary()
    render_chat_ui(provider_name, api_key, model_name, provider_config)


if __name__ == "__main__":
    main()
