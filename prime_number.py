# 소수 판정
# prime number judgement
def is_prime(number):
    if number < 2:
        return False
    else:
        for i in range(2, int(number**(1/2))+1):
            if number % i == 0:
                return False
            else:
                return True