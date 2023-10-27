#%%

import zipfile
import os

zfile = zipfile.ZipFile("UCF101_videos.zip", "r")
video_names = zfile.namelist()

print(video_names[4])
print(len(video_names))
#%%
for dir_path, _, train_names in os.walk("./UCF101_subset/train1/"):
    if train_names:
        for train_name in train_names:
            print("UCF101/" + train_name)
            video_names.remove("UCF101/" + train_name)
for dir_path, _, val_names in os.walk("./UCF101_subset/val"):
    if val_names:
        for val_name in val_names:
            video_names.remove("UCF101/" + val_name)
print(len(video_names))
#%%
for video_name in video_names:
    if video_name.endswith(".avi"):
        classname = video_name.split('_')[1]
        video = zfile.read(video_name)
        os.makedirs("./UCF101_subset/test/" + classname, exist_ok=True)
        with open("./UCF101_subset/test/" + classname + video_name[6:], "wb") as file:
            file.write(video)
        # zfile.extract(video_name, "./UCF101_subset/test/" + classname + "/")
        print("./UCF101_subset/test/" + classname + video_name[6:])
# %%
