import cv2

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 가로 설정 // set width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 세로 설정 // set height

if not cam.isOpened():
    print('Cannot get cam')
    exit()

while cam.isOpened():
    ret, frame = cam.read()

    frame = cv2.flip(frame, 1) # 0 상하반전, 1 좌우반전 // 0 vertical, 1 horizontal
    cv2.imshow('cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
