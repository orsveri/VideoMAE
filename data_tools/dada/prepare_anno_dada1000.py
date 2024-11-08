import pandas as pd
from natsort import natsorted

dada1000_anno = "/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA1000/testing/testing-text.txt"
out_anno = "/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA1000/testing/test_fragments.csv"
save_split_for_videomae = False

# Initialize lists to store each column's data
directories = []
labels = []
starts = []
ends = []
toas = []

# Read the text file and parse each line
with open(dada1000_anno, "r") as file:
    for line in file:
        # Split on the comma to separate the main data from the text portion
        main_data, text = line.strip().split(",", 1)

        # Split the main data into individual components
        directory, label, start, end, toa = main_data.split()

        # Append data to respective lists
        directories.append(directory)
        labels.append(int(label))
        starts.append(int(start))
        ends.append(int(end))
        toas.append(int(toa))

# Create a DataFrame
df = pd.DataFrame({
    "directory": directories,
    "label": labels,
    "start": starts,
    "end": ends,
    "toa": toas
})


if save_split_for_videomae:
    unique_clips = natsorted(list(set(directories)))

    with open("/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000/annotation/val_split.txt", "w") as file:
        for item in unique_clips:
            file.write(f"{item}\n")

# Save DataFrame to a .csv file
df.to_csv(out_anno, index=False)


