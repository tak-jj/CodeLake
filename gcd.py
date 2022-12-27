# 최대공약수, 유클리드 호제법
# greatest common divisor, Euclidean algorithm

def gcd(n1, n2):
    if n1 > n2:
        n1, n2 = n2, n1
    while n2 % n1 != 0:
        n1, n2 = n2, n2 % n1
    
    return n1

if __name__ == '__main__':
    print(gcd(15, 20))