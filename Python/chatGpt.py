import numpy as np

def translation_matrix(tx, ty, tz):
    """
    移動を行う行列を作成する関数
    """
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

def rotation_matrix_x(angle):
    """
    X軸周りの回転を行う行列を作成する関数
    """
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([
        [1, 0, 0, 0],
        [0, cos, -sin, 0],
        [0, sin, cos, 0],
        [0, 0, 0, 1]
    ])

def rotation_matrix_y(angle):
    """
    Y軸周りの回転を行う行列を作成する関数
    """
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([
        [cos, 0, sin, 0],
        [0, 1, 0, 0],
        [-sin, 0, cos, 0],
        [0, 0, 0, 1]
    ])

def rotation_matrix_z(angle):
    """
    Z軸周りの回転を行う行列を作成する関数
    """
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([
        [cos, -sin, 0, 0],
        [sin, cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# 移動ベクトル
tx, ty, tz = 0, 0, 0

# 回転角度
angle_x = np.pi/0.001  # X軸周りの回転角度
angle_y = np.pi/0.001  # Y軸周りの回転角度
angle_z = np.pi/0.001  # Z軸周りの回転角度

# 移動行列を作成
T = translation_matrix(tx, ty, tz)

# X軸周りの回転行列を作成
R_x = rotation_matrix_x(angle_x)

# Y軸周りの回転行列を作成
R_y = rotation_matrix_y(angle_y)

# Z軸周りの回転行列を作成
R_z = rotation_matrix_z(angle_z)

# X軸周りの回転行列から始めて、Y軸、Z軸の順に回転
# 行列の乗算は逆順に適用されることに注意してください
rotation_matrix = np.dot(np.dot(R_x, R_y), R_z)

# 移動と回転を組み合わせる
affine_matrix = np.dot(rotation_matrix, T)

print("アフィン変換行列:")
print(affine_matrix)
