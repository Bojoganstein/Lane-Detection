import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

left_top = 0, 0
right_top = 0, 0
left_bottom = 0, 0
right_bottom = 0, 0

while True:

    ret, frame = cam.read()

    # ret (bool): Return code of the `read` operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)

    # frame (array): The actual frame as an array.
    #                Height x Width x 3 (3 colors, BGR) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc.

    if ret is False:
        break



#2+#3
    width = 320
    height = 180

    dim = (width, height)

    resizedGrey = cv2.cvtColor(cv2.resize(frame, (dim)), cv2.COLOR_RGB2GRAY)

    cv2.imshow('Original', resizedGrey)
#4
    mask = np.zeros((height, width), dtype=np.uint8)
    pt1 = (int(width * 0.45), int(height * 0.76))
    pt2 = (int(width*0.52), int(height * 0.76))
    pt3 = (int(0), int(height * 1.0))
    pt4 = (int(width-1), int(height * 1.0))

    trapezoid = np.array([pt2, pt1, pt3, pt4], dtype=np.int32)

    FINAL = cv2.fillConvexPoly(mask, trapezoid, (1, 1, 1))
    cv2.imshow("Trapezoid2", mask*255)


    inmultita = resizedGrey * mask

    cv2.imshow("Inmultire pas 4", inmultita)

#5
    screenCorners = np.array([(width, 0), (0, 0), (0, height), (width, height)], dtype=np.int32)
    strechedFirst = cv2.getPerspectiveTransform(np.float32(trapezoid), np.float32(screenCorners))
    strechedFinal = cv2.warpPerspective(inmultita, strechedFirst, dim)
    cv2.imshow("Intindere", strechedFinal)


#6
    blur = cv2.blur(strechedFinal, ksize=(3, 3))
    cv2.imshow("Blur", blur)

#7
    sobel_vertical = np.float32([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)
    # DE CE INT8 de ce FLOAT 32 - > We need to convert the frames to float32 because since the frames are uint8 (unsigned) they don’t play well with negative values.
    sobel1 = cv2.filter2D(np.float32(blur), -1, sobel_vertical)
    sobel2 = cv2.filter2D(np.float32(blur), -1, sobel_horizontal)

    sobelFinal = cv2.convertScaleAbs(np.sqrt(sobel1**2+sobel2**2))
    cv2.imshow("Sobel", sobelFinal)

#8
    retval, threshold = cv2.threshold(sobelFinal, int(255*0.40), 256, cv2.THRESH_BINARY)
    cv2.imshow("Threshold", threshold)

#9
    treshholdCopy = threshold.copy()

    treshholdCopy[0:height-1, 0:int(width*0.05)] = 0;
    treshholdCopy[0:height-1, int(width*0.95):width-1] = 0;

    left_xs = []
    left_ys = []
    right_xs = []
    right_ys = []

    leftSlice = treshholdCopy[0:height-1, 0:int(width*0.5)]
    rightSlice = treshholdCopy[0:height-1, int(width*0.5+1):width-1]

    leftValues = np.argwhere(leftSlice > 1)
    rightValues = np.argwhere(rightSlice > 1)

    for i in range(0, len(leftValues)):
        left_xs.append(leftValues[i][1])
        left_ys.append(leftValues[i][0])

    for i in range(0, len(rightValues)):
        right_xs.append(rightValues[i][1]+int(width*0.5+1))
        right_ys.append(rightValues[i][0])

#10
    try:
        leftLine = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
        rightLine = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

        left_top_y = 0
        left_top_x = (left_top_y - (leftLine[0])) / (leftLine[1])

        left_bottom_y = height - 1
        left_bottom_x = (left_bottom_y - (leftLine[0])) / (leftLine[1])

        right_top_y = 0
        right_top_x = (right_top_y - (rightLine[0])) / (rightLine[1])

        right_bottom_y = height - 1
        right_bottom_x = (right_bottom_y - (rightLine[0])) / (rightLine[1])

        #if -(10 ** 8) <= left_bottom_x <= 10 ** 8:
        if (left_top[0]-(10 ** 8) <= left_top_x <= 10 ** 8+left_top[0]) and (left_top[0]-(10 ** 8) <= left_top_x <= 10 ** 8+left_top[0]):
            left_top = int(left_top_x), int(left_top_y)
            left_bottom = int(left_bottom_x), int(left_bottom_y)

        #if -(10 ** 8) <= right_bottom_x <= 10 ** 8:
        if (right_bottom[0]-(10 ** 8) <= right_bottom_x <= 10 ** 8+right_bottom[0]) and (right_top[0]-(10 ** 8) <= right_top_x <= 10 ** 8+right_top[0]):
            right_top = int(right_top_x), int(right_top_y)
            right_bottom = int(right_bottom_x), int(right_bottom_y)

        cv2.line(treshholdCopy, left_top, left_bottom, (255, 0, 0), 3)
        cv2.line(treshholdCopy, right_top, right_bottom, (128, 0, 0), 3)
        cv2.imshow("ThresholdCopyLines", treshholdCopy)

        # 11
        blankFrameLeft = np.zeros((height, width), dtype=np.uint8)
        cv2.line(blankFrameLeft, left_top, left_bottom, (255, 0, 0), 5)

        reverseMatrix = cv2.getPerspectiveTransform(np.float32(screenCorners), np.float32(trapezoid))
        FINALframeLeft = cv2.warpPerspective(blankFrameLeft, reverseMatrix, dim)

        blankFrameRight = np.zeros((height, width), dtype=np.uint8)
        cv2.line(blankFrameRight, right_top, right_bottom, (255, 255, 0), 5)

        reverseMatrix = cv2.getPerspectiveTransform(np.float32(screenCorners), np.float32(trapezoid))
        FINALframeRight = cv2.warpPerspective(blankFrameRight, reverseMatrix, dim)

        cv2.imshow('FINAL FRAME BW', FINALframeLeft++FINALframeRight)

        FINALFRAME = cv2.resize(frame, (dim))

        FINALFRAME[FINALframeLeft == 255] = (0, 0, 255)
        FINALFRAME[FINALframeRight == 255] = (0, 255, 0)

        #cv2.line(FINALFRAME, left_top, left_bottom, (255, 0, 0), 3)
        #cv2.line(FINALFRAME, right_top, right_bottom, (0, 255, 0), 3)

        cv2.imshow('FINAL FRAME!!!', FINALFRAME)
    except:
        print("eroare la un frame")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


