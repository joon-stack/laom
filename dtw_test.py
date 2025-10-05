# 터미널에서 먼저 라이브러리를 설치해주세요.
# pip install numpy tslearn matplotlib

import numpy as np
from tslearn.metrics import dtw
import matplotlib.pyplot as plt

# --- 데이터 준비 ---
# Action = (dx, dy) 변화량
# sin 곡선을 따라 움직이는 경로를 생성합니다.

# 공통 파라미터
n_steps = 50
t = np.linspace(0, np.pi, n_steps + 1)

# --- 시퀀스 A: 정속 주행 (시간을 선형적으로 사용) ---
x_a = t
y_a = np.sin(x_a)
# (x, y) 위치 좌표 생성
points_a = np.vstack([x_a, y_a]).T
# (dx, dy) action sequence 생성
sequence_a = np.diff(points_a, axis=0)


# --- 시퀀스 B: 가변 속도 주행 (시간을 비선형적으로 사용) ---
# ease-in, ease-out 효과를 주기 위해 sin 함수로 시간 축을 왜곡
t_warped = (np.sin((t - 0.5) * np.pi) + 1) / 2 * np.pi
x_b = t_warped
y_b = np.sin(x_b)
# (x, y) 위치 좌표 생성
points_b = np.vstack([x_b, y_b]).T
# (dx, dy) action sequence 생성
sequence_b = np.diff(points_b, axis=0)


# --- 거리 계산 ---

# 1. L2 거리 (엄격한 1:1 비교)
l2_distance = np.linalg.norm(sequence_a - sequence_b)

# 2. DTW 거리 (유연한 정렬 비교)
dtw_distance = dtw(sequence_a, sequence_b)


# --- 결과 출력 및 시각화 ---

print("--- 거리 계산 결과 비교 ---")
print(f"L2 거리 (엄격한 1:1 비교): {l2_distance:.4f}")
print(f"DTW 거리 (유연한 정렬 비교): {dtw_distance:.4f}")

reduction_ratio = (l2_distance - dtw_distance) / l2_distance
print(f"\nDTW를 사용했을 때 거리가 L2 대비 약 {reduction_ratio:.1%} 감소했습니다.")

# 경로 시각화 (두 경로가 동일한지 확인)
plt.figure(figsize=(12, 5))

# Plot 1: 경로 비교
plt.subplot(1, 2, 1)
path_a = np.cumsum(sequence_a, axis=0)
path_b = np.cumsum(sequence_b, axis=0)
plt.plot(path_a[:, 0], path_a[:, 1], 'b-o', markersize=3, label='경로 A (정속)')
plt.plot(path_b[:, 0], path_b[:, 1], 'r--x', markersize=3, label='경로 B (가변속)')
plt.title('두 시퀀스가 만드는 최종 경로')
plt.xlabel('X 위치')
plt.ylabel('Y 위치')
plt.legend()
plt.grid(True)
plt.axis('equal')


# Plot 2: Action의 크기(속도) 비교
plt.subplot(1, 2, 2)
plt.plot(np.linalg.norm(sequence_a, axis=1), 'b-o', markersize=3, label='시퀀스 A 속도')
plt.plot(np.linalg.norm(sequence_b, axis=1), 'r--x', markersize=3, label='시퀀스 B 속도')
plt.title('시간에 따른 Action의 크기(속도) 변화')
plt.xlabel('스텝 (시간)')
plt.ylabel('Action 크기 (||dx, dy||)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()