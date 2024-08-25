import math

# 角度を度からラジアンに変換
angle_degrees = 40
angle_radians = math.radians(angle_degrees)

# コサインを計算
cos_value = math.cos(angle_radians)

print("Cos(", angle_degrees, "degrees) =", cos_value)