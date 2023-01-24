# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:nomarker
#     text_representation:
#       extension: .py
#       format_name: nomarker
#       format_version: '1.0'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Merging all data sources

from pathlib import Path

import pandas as pd

# # Load processed tables

rawdata_file = Path.cwd() / "mimic-iv-1.0.zip"
dataset_path = Path.cwd() / "processed"
rawdata_path = Path.cwd() / "raw"


with pd.option_context("string_storage", "pyarrow"):
    admissions = pd.read_parquet(dataset_path / "admissions_processed.parquet")

    labevents = pd.read_parquet(
        dataset_path / "labevents_processed.parquet",
        columns=["subject_id", "hadm_id", "charttime", "valuenum", "label"],
    )
    inputevents = pd.read_parquet(
        dataset_path / "inputevents_processed.parquet",
        columns=["subject_id", "hadm_id", "charttime", "amount", "label"],
    )
    outputevents = pd.read_parquet(
        dataset_path / "outputevents_processed.parquet",
        columns=["subject_id", "hadm_id", "charttime", "value", "label"],
    )
    prescriptions = pd.read_parquet(
        dataset_path / "prescriptions_processed.parquet",
        columns=["subject_id", "hadm_id", "charttime", "dose_val_rx", "drug"],
    )

admissions

for table in (labevents, inputevents, outputevents, prescriptions):
    display(table.shape)
    display(pd.DataFrame({"type": table.dtypes, "uniques": table.nunique()}))

# ## Change the name of amount. Valuenum for every table

inputevents = inputevents.rename(columns={"amount": "valuenum"})
outputevents = outputevents.rename(columns={"value": "valuenum"})
prescriptions = prescriptions.rename(columns={"dose_val_rx": "valuenum"})
prescriptions = prescriptions.rename(columns={"drug": "label"})

# ## Merge the tables

tables = {
    "inputevent": inputevents,
    "labevent": labevents,
    "outputevent": outputevents,
    "prescription": prescriptions,
}

merged_df = pd.concat(tables, names=["type"]).reset_index(drop=True)
assert all(merged_df.notna())
merged_df

# ## Validate that all labels have different names.

assert merged_df["label"].nunique() == (
    inputevents["label"].nunique()
    + labevents["label"].nunique()
    + outputevents["label"].nunique()
    + prescriptions["label"].nunique()
)

# ## Validate that all subject_id / hadm_id pairs are unique

assert all(merged_df.groupby("subject_id")["hadm_id"].nunique() == 1)
assert all(merged_df.groupby("hadm_id")["subject_id"].nunique() == 1)

# ## Create Metadata tensor

metadata = admissions.copy().sort_values(by=["subject_id"])

for key in ["hadm_id", "subject_id"]:
    mask = metadata[key].isin(merged_df[key])
    metadata = metadata[mask]
    print(f"Removing {(~mask).sum()} {key}")
    print(f"Number of patients remaining: {metadata['subject_id'].nunique()}")
    print(f"Number of admissions remaining: {metadata['hadm_id'].nunique()}")
    print(f"Number of events remaining: {metadata.shape}")

# # Filter tables

# ## Only keep data with duration in bounds

mintime = metadata.set_index("subject_id")[["admittime", "edregtime"]].min(axis=1)

delta = (
    merged_df.groupby("subject_id")["charttime"].max()
    - merged_df.groupby("subject_id")["charttime"].min()
)
mask = delta < metadata.set_index("subject_id")["elapsed_time"]
mask.mean()

# ## Only keep data chose `charttime` > `admittime`

mask = (
    merged_df.groupby("subject_id")["charttime"].min()
    >= metadata.set_index("subject_id")["admittime"]
)
mask.mean()

mask = (
    merged_df.groupby("subject_id")["charttime"].min()
    >= metadata.set_index("subject_id")["edregtime"]
)
mask.mean()

mask = merged_df.groupby("subject_id")["charttime"].min() >= mintime
mask.mean()

# ## Only keep data chose `charttime` < `dischtime`

mask &= (
    merged_df.groupby("subject_id")["charttime"].max()
    <= metadata.set_index("subject_id")["dischtime"]
)
mask.mean()

# ## Only keep data chose `charttime` ends within the (2d, 29d) bound

lb = mintime + pd.Timedelta("2d")
ub = mintime + pd.Timedelta("29d")
et = merged_df.groupby("subject_id")["charttime"].max()
mask &= (lb <= et) & (et <= ub)
mask.mean()

# ### Note: combined masks ⟹ only ~ 70 % of data remains

# # Add timestamps and Label Codes

# ## Create timestamps

reftime = merged_df.groupby("subject_id")["charttime"].min()
reftime = reftime.rename("reftime")
metadata = metadata.join(reftime, on="subject_id")
merged_df = pd.merge(reftime, merged_df, left_index=True, right_on="subject_id")
merged_df["time_stamp"] = merged_df["charttime"] - merged_df["reftime"]
merged_df = merged_df.drop(columns=["reftime"])

# ## Create label codes.

merged_df["label"] = merged_df["label"].astype("string").astype("category")
merged_df["label_code"] = merged_df["label"].cat.codes
merged_df = merged_df.sort_values(["hadm_id", "valuenum", "time_stamp", "label_code"])
merged_df

# ## select only values within first 48 hours

mask = merged_df["time_stamp"] < pd.Timedelta(48, "h")
merged_df = merged_df[mask].copy()
print(f"Number of patients considered: {merged_df['hadm_id'].nunique()}")
assert all(merged_df["time_stamp"] < pd.Timedelta(48, "h"))

# ## Convert time_stamp to minutes

merged_df["time_stamp"] = merged_df["time_stamp"].dt.total_seconds().div(60).astype(int)

# # Finalize and Serialize Tensors

# ## Select columns used in final dataset

LABELS = merged_df["label"].dtype
LABELS

selection = ["subject_id", "time_stamp", "label", "valuenum"]
timeseries = merged_df[selection].copy()
timeseries = timeseries.sort_values(by=selection)
timeseries = timeseries.set_index(["subject_id", "time_stamp"])
timeseries.to_parquet(dataset_path / "timeseries_triplet.parquet")
print(timeseries.shape, timeseries.dtypes)
timeseries

# ## Sparse Representation

from tsdm.encoders import TripletDecoder

timeseries.label = timeseries.label.astype(LABELS)
encoder = TripletDecoder(value_name="valuenum", var_name="label")
encoder.fit(timeseries)
encoded = encoder.encode(timeseries)
assert len(encoded.index.unique()) == len(encoded)
encoded.columns = encoded.columns.astype("string")
encoded.to_parquet(dataset_path / "timeseries.parquet")
encoded.columns = encoded.columns.astype(LABELS)
encoded

# ## Save Metadata Tensor

selection = [
    "subject_id",
    "reftime",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "ethnicity",
    "hospital_expire_flag",
    "gender",
    "anchor_age",
    "anchor_year",
    "anchor_year_group",
]
metadata = metadata[selection]
metadata = metadata.set_index("subject_id")
metadata = metadata.sort_index()
metadata.to_parquet(dataset_path / "metadata.parquet")
print(metadata.shape, metadata.dtypes)
metadata

# ## Create label table

labels = pd.Series(LABELS.categories, name="label", dtype=LABELS)
labels = labels.to_frame()
label_origin = pd.Series(
    {
        key: name
        for name, table in tables.items()
        for key in table["label"].cat.categories
    },
    name="origin",
    dtype="category",
)
label_origin.index.name = "label"
label_origin.index = label_origin.index.astype(LABELS)
labels = pd.merge(labels, label_origin, right_index=True, left_on="label")
labels["code"] = labels["label"].cat.codes
missing = encoded.isna().mean().rename("missing").astype("float32")
means = encoded.mean().rename("mean").astype("float32")
stdvs = encoded.std().rename("stdv").astype("float32")
labels = labels.join(missing, on="label")
labels = labels.join(means, on="label")
labels = labels.join(stdvs, on="label")
labels.to_parquet(dataset_path / "labels.parquet")
print(labels.dtypes)
labels
