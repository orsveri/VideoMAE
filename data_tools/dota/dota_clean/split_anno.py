import json
import os
from copy import deepcopy


clip = "bFGmOp9H3MA_000355"

anno_dir = "/mnt/experiments/sorlova/datasets/DoTA_refined/dataset/annotations"


# Input and output paths
annotation_file = os.path.join(anno_dir, f"{clip}.json")
output_annotation_file_part1 = os.path.join(anno_dir, f"{clip}_p1.json")
output_annotation_file_part2 = os.path.join(anno_dir, f"{clip}_p2.json")

# Frame splits
split1 = range(0, 125+1)
split2 = range(126, 237+1)


# Adjust paths and frame IDs for the given split
def adjust_annotations(input_file, output_file, frame_range):
    with open(input_file, "r") as f:
        annotations = json.load(f)

    first_frame = min(frame_range)

    # Update annotations for the specified frame range
    old_video_name = annotations["video_name"]
    new_video_name = os.path.splitext(os.path.basename(output_file))[0]
    annotations["video_name"] = new_video_name
    anno_labels = deepcopy([frame for frame in annotations["labels"] if frame["frame_id"] in frame_range])
    for i, frame in enumerate(anno_labels):
        frame["image_path"] = frame["image_path"].replace(
            f"{old_video_name}/images/{frame['frame_id']:06d}.jpg",
            f"{new_video_name}/images/{i:06d}.jpg"
        )
        frame["frame_id"] = i
    annotations["labels"] = anno_labels

    # Save the adjusted annotations
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=2)


# Process part 1
adjust_annotations(
    annotation_file,
    output_annotation_file_part1,
    split1,
)

# Process part 2
adjust_annotations(
    annotation_file,
    output_annotation_file_part2,
    split2,
)

print("Annotation files adjusted successfully!")
