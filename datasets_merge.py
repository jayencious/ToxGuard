import pandas as pd
import re

print("Loading datasets...")

jigsaw_df = pd.read_csv("preprocessed_data/jigsaw-data/train.csv")
github_df = pd.read_csv("preprocessed_data/github-data/labeled_data.csv")

# Processsing of Jigsaw Dataset
print("Processing Jigsaw dataset...")

toxic_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# We are creating a single "label" column: if any of the above is > 0, value is 1, else 0
jigsaw_df["label"] = jigsaw_df[toxic_columns].max(axis=1)

# Renaming the text column to a standard name
jigsaw_df = jigsaw_df.rename(
    columns={ "comment_text": "text" }
)

# We'll keep only the columns that we need
jigsaw_clean = jigsaw_df[["text", "label"]].copy()

# Data Sampling or Balancing
# Jigsaw dataset is huge, so we'll take 10,000 clean comments and 10,000 toxic comments
jigsaw_toxic = jigsaw_clean[jigsaw_clean["label"] == 1].sample(
    n=10000, random_state=42
)
jigsaw_non_toxic = jigsaw_clean[jigsaw_clean["label"] == 0].sample(
    n=10000, random_state=42
)

jigsaw_final = pd.concat(
    [jigsaw_toxic, jigsaw_non_toxic]
)

# Processing of Github (Davidson) Dataset
print("Processing Github dataset...")
# Davidson labels (class): 0 = hate speech, 1 = offensive language, 2 = neither
# We will map 0 and 1 to our '1' (toxic) and 2 to our '0' (clean)
def map_davidson_labels(c):
    if c in [0, 1]:
        return 1
    
    return 0

github_df["label"] = github_df["class"].apply(
    map_davidson_labels
)

# Renaming the text column to our standard name
github_df = github_df.rename(
    columns={"tweet": "text"}
)

# We will keep only the columns that we need
github_final = github_df[["text", "label"]].copy()

# Merging and shuffling the datasets
print("Merging the datasets...")

merged_df = pd.concat(
    [jigsaw_final, github_final],
    ignore_index=True
)

# Shuffling the dataset so the model doesn't learn based on the order
merged_df = merged_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Text Cleaning
print("Cleaning the textual data...")

def clean_text(text):
    text = str(text).lower()    # to convert the text to lowercase
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text,
        flags=re.MULTILINE
    )   # to remove URLs
    text = re.sub(
        r"\@\w+|\#",
        "",
        text
    )   # to remove mentions and hashtags
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )   # to remove special characters and numbers
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()   # to remove extra spaces

    return text

merged_df["text"] = merged_df["text"].apply(clean_text)

# To drop any rows that became empty after cleaning the text
merged_df = merged_df[merged_df["text"].astype(bool)]

# Exporting the final dataset
output_filename = "final_merged_dataset.csv"
merged_df.to_csv(output_filename, index=False)

print(f"SUCCESS! Final dataset saved as '{output_filename}'.")
print(f"Total rows: {len(merged_df)}")
print(merged_df["label"].value_counts())