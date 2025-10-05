import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_hdf5(file_path, dataset_name='observations', num_samples=20):
    with h5py.File(file_path, 'r') as f:
        # print(f.keys())
        print(f.attrs.keys())
        for key in f.attrs.keys():
            print("INFO: ", key, f.attrs[key])

            # '0' 그룹 아래 키 확인
        group = f['0']
        print(group.keys())  # 여기에 'observations', 'actions' 등이 나올 거예요
        if dataset_name not in group:
            print(f"Dataset '{dataset_name}' not found in the file.")
            return
        
        data = group[dataset_name][:]
        print(f"Loaded {len(data)} samples from '{dataset_name}'.")
        
        for i in range(min(num_samples, len(data))):
            img_array = data[i]

            img = Image.fromarray(img_array)
            img.save(f"misc/{i}.png")
            print(f"INFO: image saved to misc/{i}.png")
        print("INFO: actions", group['actions'].shape, list(group['actions'])[:num_samples])
        print("INFO: states", group['states'].shape, list(group['states'])[:num_samples])
        obs = np.array(group['obs'])
        print("INFO: obs.min, max, mean", obs.min(), obs.max(), obs.mean(), obs.shape)
        print("states[0]", group['states'][0])

if __name__ == "__main__":
    file_path = "/shared/s2/lab01/dataset/DMC/cheetah-run-scale-easy-video-hard-64px-5k.hdf5"  # HDF5 파일 경로
    visualize_hdf5(file_path, dataset_name='obs')  # 필요 시 dataset_name 변경 