import cv2


def mouse_callback(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("Distance is ",disp[y,x])


if __name__ == '__main__':
    sub_dir = "data/data_202107301058/"
    im_i = 6

    disp = cv2.imread(sub_dir + f"depth/{im_i}.png",-1)
    iml = cv2.imread(sub_dir + f"img/{im_i}.jpg",-1)
    iml = cv2.resize(iml,(disp.shape[1],disp.shape[0]))

    # 展示距离
    cv2.namedWindow('distance_map', cv2.WINDOW_NORMAL)
    cv2.imshow("distance_map", iml)
    cv2.setMouseCallback("distance_map", mouse_callback)
    cv2.waitKey(0)

    print(1)