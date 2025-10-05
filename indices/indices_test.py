import pickle

file_path = '/home/s2/youngjoonjeong/github/laom/indices/traj_splits_labeled_seed_0_train_2_val_4.pkl'

# 'rb'는 이진 파일(binary file)을 읽기 모드로 연다는 뜻입니다.
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(data)
# train_indices 리스트를 오름차순으로 정렬
data['train_indices'].sort()

# val_indices 리스트를 오름차순으로 정렬
data['val_indices'].sort()

print(data['train_indices'])
print(data['val_indices'])
