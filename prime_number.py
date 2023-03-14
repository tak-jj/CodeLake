# 소수 판정
# prime number judgement
def isPrime(number):
    if number < 2:
        return False
    else:
        for i in range(2, int(number**(1/2))+1):
            if number % i == 0:
                return False
        return True
            
if __name__ == '__main__':
    print(isPrime(1), isPrime(2), isPrime(327), isPrime(7523))