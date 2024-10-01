# 定义求解函数
def find_integer():
    for i in range(2, 85):  # i 的范围从 2 到 84
        if 168 % i == 0:  # i 是 168 的因子
            j = 168 // i  # 计算对应的 j
            if i > j and (i + j) % 2 == 0 and (i - j) % 2 == 0:  # 确保 i 和 j 同为偶数或者奇数
                m = (i + j) // 2
                n = (i - j) // 2
                x = n * n - 100
                print(f"x: {x}, m: {m}, n: {n}")

# 调用求解函数
find_integer()