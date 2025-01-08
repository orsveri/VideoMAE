import zipfile
import json
import pandas as pd
import os

# Paths
clip_name = "bFGmOp9H3MA_000355"
input_dir = f"/mnt/experiments/sorlova/datasets/DoTA_refined/frames/{clip_name}"
output_dir1 = f"/mnt/experiments/sorlova/datasets/DoTA_refined/frames/{clip_name}_p1"
output_dir2 = f"/mnt/experiments/sorlova/datasets/DoTA_refined/frames/{clip_name}_p2"

# Frame splits
split1 = range(0, 125+1)  # Frames 000000–000153
split2 = range(126, 237+1)  # Frames 000154–000257

images_zip = "images.zip"
depth_zip = "dany2_rel.zip"
detections_csv = "det_yolov5.csv"
detections_json = "detections_yolov5.json"


def create_zip_subset(input_zip, output_zip, frame_range, file_extension):
    """
    Create a subset ZIP file based on frame range.

    Args:
        input_zip (str): Path to the input ZIP file.
        output_zip (str): Path to the output ZIP file.
        frame_range (range): Range of frames to include in the subset.
        file_extension (str): Extension of files to filter (e.g., '.jpg', '.png').
    """

    with zipfile.ZipFile(input_zip, "r") as zf_in:
        with zipfile.ZipFile(output_zip, "w") as zf_out:
            first_i = min(list(frame_range))
            for frame_num in frame_range:
                filename = f"{frame_num:06d}{file_extension}"
                new_filename = f"{frame_num-first_i:06d}{file_extension}"
                if filename in zf_in.namelist():
                    with zf_in.open(filename) as file_in:
                        zf_out.writestr(new_filename, file_in.read())


first_frame_part1 = min(split1)
first_frame_part2 = min(split2)

# Step 0
os.makedirs(output_dir1, exist_ok=False)
os.makedirs(output_dir2, exist_ok=False)

# Step 1: Create ZIP subsets for frames and depth maps
create_zip_subset(os.path.join(input_dir, images_zip), os.path.join(output_dir1, images_zip), split1, ".jpg")
create_zip_subset(os.path.join(input_dir, images_zip), os.path.join(output_dir2, images_zip), split2, ".jpg")
create_zip_subset(os.path.join(input_dir, depth_zip), os.path.join(output_dir1, depth_zip), split1, ".png")
create_zip_subset(os.path.join(input_dir, depth_zip), os.path.join(output_dir2, depth_zip), split2, ".png")

# Step 2: Split CSV metadata
detections_df = pd.read_csv(os.path.join(input_dir, detections_csv))
detections_part1 = detections_df[detections_df["frame"].astype(int).isin(split1)].copy()
detections_part1["frame"] = detections_part1["frame"] - first_frame_part1
detections_part2 = detections_df[detections_df["frame"].astype(int).isin(split2)].copy()
detections_part2["frame"] = detections_part2["frame"] - first_frame_part2

detections_part1.index.name = ""
detections_part1.to_csv(os.path.join(output_dir1, detections_csv), index=False)
detections_part2.index.name = ""
detections_part2.to_csv(os.path.join(output_dir2, detections_csv), index=False)

# Step 3: Split JSON metadata
with open(os.path.join(input_dir, detections_json), "r") as f:
    detections = json.load(f)

# For Part 1
detections_json_part1 = []
for entry in detections:
    frame_num = int(entry["Frame"])
    if frame_num in split1:
        new_entry = entry.copy()
        new_entry["Frame"] = f"{frame_num - first_frame_part1:06d}"
        detections_json_part1.append(new_entry)

with open(os.path.join(output_dir1, detections_json), "w") as f:
    json.dump(detections_json_part1, f, indent=4)

# For Part 2
detections_json_part2 = []
for entry in detections:
    frame_num = int(entry["Frame"])
    if frame_num in split2:
        new_entry = entry.copy()
        new_entry["Frame"] = f"{frame_num - first_frame_part2:06d}"
        detections_json_part2.append(new_entry)

with open(os.path.join(output_dir2, detections_json), "w") as f:
    json.dump(detections_json_part2, f, indent=4)

print("Splitting complete!")
