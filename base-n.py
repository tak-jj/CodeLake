# 10진법 수를 n진법 수로 바꾸기
# change base-10 number to base-n number

def change_base(number, n=2):
    result = ''
    while number != 0:
        temp = number % n
        number = number // n
        result += str(temp)
    
    return result[::-1]

if __name__ == '__main__':
    print(change_base(10, 2))
    print(change_base(25, 3))