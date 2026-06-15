import argparse
import os

import pandas as pd

MIN_WEIGHT = 1

INSTRUCTION_LABELS = {
    "provide_answer": "Answering a Question",
    "provide_response": "Responding to a Statement",
    "provide_examples": "Providing Examples",
    "provide_definition": "Providing a Definition",
    "provide_guidance": "Providing Guidance",
    "": "General Guidance",
}
TYPE_LABELS = {
    "question": "Question",
    "statement": "Statement",
    "request": "Request",
    "": "General Topic",
}


def normalise(value):
    if pd.isna(value):
        return ""
    return str(value).strip().replace("_x000D_", "").strip()


def section_heading(instruction: str, type_: str) -> str:
    # Safely handle empty strings or missing keys
    inst_label = INSTRUCTION_LABELS.get(
        instruction,
        instruction.replace("_", " ").title() if instruction else "General Guidance",
    )
    type_label = TYPE_LABELS.get(type_, type_.title() if type_ else "Context")
    return f"{inst_label} — {type_label}"


def write_markdown_files(output_dir, dataframe):
    os.makedirs(output_dir, exist_ok=True)
    example_count = 0

    for (instruction, type_), group in dataframe:
        for _, row in group.iterrows():
            example_count += 1

            lines = [
                f"Context: {type_ if type_ else 'General Topic'}",
                f"Scenario: {row['Input']}",
                f"Ally Response: {row['Output']}",
            ]

            file_name = f"example_{example_count:04d}.md"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    print(f"Total files generated: {example_count}")


def main(input_path: str, output_dir: str):
    print(f"Reading: {input_path}")
    df = pd.read_excel(input_path)

    df.columns = [c.strip() for c in df.columns]

    required = {"Weight", "Input", "Output", "Instruction", "Type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns not found: {missing}\nFound: {list(df.columns)}"
        )

    # Ensure Weight is numeric before filtering to prevent type comparison errors
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    ally = df[df["Weight"] >= MIN_WEIGHT].copy()

    ally["Input"] = ally["Input"].apply(normalise)
    ally["Output"] = ally["Output"].apply(normalise)
    ally["Instruction"] = ally["Instruction"].apply(normalise).str.lower()
    ally["Type"] = ally["Type"].apply(normalise).str.lower()

    # Filter out empty records
    ally = ally[(ally["Input"] != "") & (ally["Output"] != "")]

    grouped_ally = ally.groupby(["Instruction", "Type"], sort=True)
    write_markdown_files(output_dir, grouped_ally)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert Excel to OpenWebUI Markdown RAG Knowledge Base"
    )
    ap.add_argument("input_file", help="The Excel spreadsheet to convert to Markdown")
    ap.add_argument("output_dir", help="The directory to write the Markdown files to")
    args = ap.parse_args()

    main(args.input_file, args.output_dir)
