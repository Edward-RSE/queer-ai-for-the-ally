import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd

MIN_WEIGHT = 1

INSTRUCTION_LABELS = {
    "provide_answer": "Answering a Question",
    "provide_response": "Responding to a Statement",
    "provide_examples": "Providing Examples",
    "provide_definition": "Providing a Definition",
    "provide_guidance": "Providing Guidance",
    "provide_pushback": "Providing Pushback",
    "": "General Guidance",
}

TYPE_LABELS = {
    "question": "Question",
    "statement": "Statement",
    "request": "Request",
    "": "General Topic",
}

# Numeric category codes used by the Reddit dataset
CATEGORY_LABELS = {
    7: "Personal Experience",
    8: "Family & Relationships",
    9: "Society & Identity",
}


def normalise(value):
    if pd.isna(value):
        return ""
    # there is so funky stuff going on with line endings
    return str(value).strip().replace("_x000D_", "").strip()


def detect_schema(df: pd.DataFrame) -> str:
    """Try and figure out if the DataFrame is the Reddit or HuggingFace dataset.

    The different datasets have different columns of data, so we need to figure
    out which is which. That's because we need to normalise the schema so we can
    process both files with the same script.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing one of the datasets.

    Returns
    -------
    str
        The name of the dataset.

    """
    hf_cols = {"Input", "Output", "Instruction", "Type", "Weight"}
    reddit_cols = {"Post (full)", "Reply", "Category"}

    stripped = set(df.columns)
    if hf_cols <= stripped:
        return "huggingface"
    if reddit_cols <= stripped:
        return "reddit"

    raise ValueError(
        f"Unrecognised schema. Expected columns for HuggingFace ({hf_cols}) "
        f"or Reddit ({reddit_cols}), but found: {list(stripped)}"
    )


def load_huggingface(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the HuggingFace dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the HuggingFace dataset.

    Returns
    -------
    pd.DataFrame
        The re-normalised DataFrame.

    """
    required = {"Weight", "Input", "Output", "Instruction", "Type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns not found: {missing}\nFound: {list(df.columns)}"
        )

    df = df.copy()
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    nan_count = df["Weight"].isna().sum()
    if nan_count:
        print(f"Warning: {nan_count} row(s) had non-numeric Weight and were skipped.")

    ally = df[df["Weight"] >= MIN_WEIGHT].copy()
    ally["Input"] = ally["Input"].apply(normalise)
    ally["Output"] = ally["Output"].apply(normalise)
    ally["Instruction"] = ally["Instruction"].apply(normalise).str.lower()
    ally["Type"] = ally["Type"].apply(normalise).str.lower()

    return ally[["Input", "Output", "Instruction", "Type"]]


def load_reddit(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the Reddit dataset to the HuggingFace schema.

    The file has two Weight columns (one for the post, one for the reply).
    Instruction is absent in this dataset; every row is mapped to "" so it
    groups under "General Guidance".

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the Reddit datasets.

    Returns
    -------
    pd.DataFrame
        The re-normalised DataFrame.

    """
    df = df.copy()
    cols = list(df.columns)
    cols[4] = "post_weight"
    cols[6] = "reply_weight"
    df.columns = cols

    df["post_weight"] = pd.to_numeric(df["post_weight"], errors="coerce")
    df["reply_weight"] = pd.to_numeric(df["reply_weight"], errors="coerce")

    for col in ("post_weight", "reply_weight"):
        nan_count = df[col].isna().sum()
        if nan_count:
            print(
                f"Warning: {nan_count} row(s) had non-numeric {col} and were skipped."
            )

    ally = df[
        (df["post_weight"] >= MIN_WEIGHT) & (df["reply_weight"] >= MIN_WEIGHT)
    ].copy()

    ally["Input"] = ally["Post (full)"].apply(normalise)
    ally["Output"] = ally["Reply"].apply(normalise)
    ally["Instruction"] = ""

    def map_category(raw):
        if pd.isna(raw):
            return ""
        try:
            key = int(raw)
            return CATEGORY_LABELS.get(key, str(key))
        except (ValueError, TypeError):
            return normalise(raw).lower()

    ally["Type"] = ally["Category"].apply(map_category)

    return ally[["Input", "Output", "Instruction", "Type"]]


def write_markdown_files(
    output_dir: str | Path, grouped: pd.api.typing.DataFrameGroupBy
) -> int:
    """Write DataFrame to multiple Markdown files.

    Parameters
    ----------
    output_dir : str | Path
        The directory to write the Markdown files to.
    grouped : pd.api.typing.DataFrameGroupBy
        The grouped together DataFrame to be converted into Markdown files.

    Returns
    -------
    int
        The number of Markdown files written to disk.

    """
    os.makedirs(output_dir, exist_ok=True)

    existing = [
        int(m.group(1))
        for f in os.listdir(output_dir)
        if (m := re.fullmatch(r"example_(\d+)\.md", f))
    ]
    example_count = max(existing, default=0)

    files_written = 0
    for (instruction, type_), group in grouped:
        for _, row in group.iterrows():
            example_count += 1
            files_written += 1
            lines = [
                "# AI Ally Input-Response Example\n",
                f'**Context:** "{type_ if type_ else "General Topic"}"\n',
                f'**Input:** "{row["Input"]}"\n',
                f'**Response:** "{row["Output"]}"\n',
            ]
            pad = max(4, len(str(example_count)))
            file_name = f"example_{example_count:0{pad}d}.md"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
    return files_written


def main(output_dir: str | Path) -> None:
    """Main function of the script.

    Parameters
    ----------
    output_dir : str
        The directory to write the Markdown files to.

    """
    input_files = [
        "./datasets/DGN_data-coded-HuggingFace.xlsx",
        "./datasets/DGN_data-reddit-manualSelection.xlsx",
    ]

    created_files = 0

    shutil.rmtree(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    for input_path in input_files:
        df = pd.read_excel(input_path)
        df.columns = [c.strip() for c in df.columns]

        schema = detect_schema(df)
        if schema == "huggingface":
            ally = load_huggingface(df)
        else:
            ally = load_reddit(df)

        ally = ally[(ally["Input"] != "") & (ally["Output"] != "")]
        grouped = ally.groupby(["Instruction", "Type"], sort=True)

        created_files += write_markdown_files(output_dir, grouped)

    print(f"Created {created_files} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert Excel to OpenWebUI Markdown RAG Knowledge Base."
    )
    ap.add_argument("output_dir", help="The directory to write the Markdown files to.")
    args = ap.parse_args()
    main(args.output_dir)
