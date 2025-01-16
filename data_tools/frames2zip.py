import os
import re
import shutil
import zipfile
from tqdm import tqdm


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def folder_to_zip(folder, zip_path, remove_folder=False, existing_action="err"):
    if os.path.exists(zip_path):
        if existing_action == "del":
            print(f"Removing existing zip file {zip_path}")
            os.remove(zip_path)
        elif existing_action == "err":
            raise FileExistsError(f"Zip file exists: {zip_path}")
        elif existing_action == "skip":
            print(f"Skipping existing zip file {zip_path}")
            return
        else:
            raise AssertionError("Incorrect existing_action argument. Can be del, err or skip")

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filename in tqdm(sorted(os.listdir(folder), key=natural_key), leave=False, position=1):
            if filename in ("Thumbs.db",):
                continue
            zipf.write(os.path.join(folder, filename), arcname=filename)

    # Remove folder
    if remove_folder:
        shutil.rmtree(folder)
        print(f"Removed folder {folder}")


def compress_folders(scenario_folder, subfolder):
    for sim_run in tqdm(sorted(os.listdir(scenario_folder), key=natural_key), leave=True, position=0):
        sim_run_folder = os.path.join(scenario_folder, sim_run, subfolder)
        out_zip_path = os.path.join(scenario_folder, sim_run, f"{subfolder}.zip")
        folder_to_zip(folder=sim_run_folder, zip_path=out_zip_path, remove_folder=True, existing_action="skip")


if __name__ == "__main__":
    inp_folders = "/gpfs/work3/0/tese0625/RiskNetData/DoTA/frames"
    compress_folders(scenario_folder=inp_folders, subfolder="images")
    print("Done!")



