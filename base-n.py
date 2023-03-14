# 10진법 수를 n진법 수로 바꾸기
# change base-10 number to base-n number

def changeBase(number, n=2):
    result = ''
    while number != 0:
        temp = number % n
        number = number // n
        result += str(temp)
    
    return result[::-1]

if __name__ == '__main__':
    print(changeBase(10, 2))
    print(changeBase(25, 3))