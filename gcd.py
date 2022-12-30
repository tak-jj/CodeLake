# 최대공약수, 유클리드 호제법
# greatest common divisor, Euclidean algorithm
def gcd(a, b):
    if b > a:
        a, b = b, a
    while b != 0:
        a = a % b
        a, b = b, a
    return a

if __name__ == '__main__':
    print(gcd(60, 144))