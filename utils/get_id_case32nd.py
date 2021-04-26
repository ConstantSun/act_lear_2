import sys
import glob
import json
from tqdm import tqdm

def add_image_id_to_pool(id: str, filename="pooling_data.json"):
    """id: image name, e.g: GEMS_IMG__2010_MAR__12__HA122541__F8HB4A50_24"""
    with open(filename, 'r+') as f:
        dic = json.load(f)
        dic["ids"].append(id)
    with open(filename, 'w') as file:
        json.dump(dic, file)

full_data = "/home/hang/Documents/active_learning/data/data_train/"
count = 165
for file in tqdm(glob.glob(full_data+"imgs/*")):
    id = file[file.rfind("/"):]
    add_image_id_to_pool(id[1:][:-4], "../tmp/data_165.json")
    count = count - 1
    if count == 0:
        break
