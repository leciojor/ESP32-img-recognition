import os
import numpy as np
from PIL import Image
import pandas as pd
import h5py
from tqdm import tqdm

def get_tensors(img):
    tensor = np.array(img)
    return tensor

def main(df=True, limit=1):
    label_map = {"dirt": 0, "rocky": 1, "grassy": 2, "sandy": 3}

    with h5py.File("data/final_dataset.h5", "w") as f:
        img_dset = f.create_dataset("images", (0, 563, 1000 , 3), maxshape=(None, 563, 1000, 3), compression="gzip", dtype=np.uint8)
        label_dset = f.create_dataset("labels", (0, 4), maxshape=(None, 4), compression="gzip", dtype=np.uint8)
        data_df = {
                        'R': [],
                        'G': [],
                        'B': [],
                        'class': []
                    }
        
        for tgt in ["dirt", "rocky", "grassy", "sandy"]:
            count = 0
            for file in tqdm(os.listdir(f"data/full_clean/{tgt}")):
                if count == limit:
                    break
                image = Image.open(f"data/full_clean/{tgt}/{file}").resize((563, 1000))
                img_tensor = get_tensors(image)

                if df:
                    r = img_tensor[:, :, 0]
                    g = img_tensor[:, :, 1]
                    b = img_tensor[:, :, 2]
                    data_df['R'].append(r)
                    data_df['B'].append(b)
                    data_df['G'].append(g)
                    data_df['class'].append(tgt)

                else:

                    label = np.eye(4)[label_map[tgt]]

                    img_dset.resize((img_dset.shape[0] + 1), axis=0)
                    label_dset.resize((label_dset.shape[0] + 1), axis=0)

                    img_dset[-1] = img_tensor  
                    label_dset[-1] = label   

                count +=1  

        if data_df:
            df = pd.DataFrame(data_df)
            df.to_csv("report_display.csv")            

if __name__ == '__main__':
    main()
