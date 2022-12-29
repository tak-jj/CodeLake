# 이미지를 정사각형으로 변환, 빈 부분은 색으로 채우기
# change image to square, fill empty space with color.
from PIL import Image

def expand2square(image_path, background_color=(0,0,0)):
    image = Image.open(image_path)
    x, y = image.size
    if x == y:
        return image
    elif x > y:
        result = Image.new(image.mode, (x, x), background_color)
        result.paste(image, (0, (x-y)//2))
        return result
    else:
        result = Image.new(image.mode, (y, y), background_color)
        result.paste(image, ((y-x)//2, 0))
        return result