import numpy as np
import pyzed.sl as sl
import cv2
import os
import math

txt_path = 'data/info.txt'

with open(txt_path,'a') as file:
    file.write('frame,time,alpha,beta,gamma\n')

depth_path = 'data/depth'
img_path = 'data/img'

path_list = [img_path,depth_path]

for path_i in path_list:
    if not os.path.exists(path_i):
        os.makedirs(path_i)

zed = sl.Camera()           # Create a ZED camera object
input_type = sl.InputType() # Set configuration parameters

init = sl.InitParameters(input_t=input_type)  # 初始化
init.camera_resolution = sl.RESOLUTION.HD2K # 相机分辨率(默认-HD720) 弄成2K
init.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式  (默认-PERFORMANCE)
init.coordinate_units = sl.UNIT.MILLIMETER    # 毫米级    (默认-MILLIMETER)

# Open the camera
err = zed.open(init)
if err != sl.ERROR_CODE.SUCCESS:
    print('Cannot open the camera!')
    exit(-1)

image_size = zed.get_camera_information().camera_resolution

a = image_size.width
b = image_size.height
# 定义图像数据
image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
depth_image_zed = sl.Mat(image_size.width/2, image_size.height/2, sl.MAT_TYPE.U16_C1) # 16位进行保存

# 定义测量数据
depth_zed = sl.Mat(image_size.width/2, image_size.height/2, sl.MAT_TYPE.U16_C1)
point_cloud_zed = sl.Mat(image_size.width/2, image_size.height/2)
key = ''

frame = 0
num = 0



while key != 113:
    print(frame)
    frame += 1
    zed.grab()
    # 拿取图像
    zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)       # Retrieve the "left image, depth image" in the half-resolution
    zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH,sl.MEM.CPU, image_size)
    # 拿去测量的信息
    zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA)
    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
    x = image_zed.get_width() / 2
    y = image_zed.get_height() / 2
    point_cloud_value = point_cloud_zed.get_value(x, y)[1]
    depth_map = depth_zed.get_data()

    image_ocv = image_zed.get_data()
    depth_image_ocv = depth_image_zed.get_data()


    # depth_map 处理
    depth_map[np.isnan(depth_map)] = 0  # 将nan填充为0
    depth_map[np.isinf(depth_map)] = 65535
    depth_map = depth_map.astype(np.uint16)
    depth_map = cv2.resize(depth_map,(int(image_size.width/ 2), int( image_size.height/ 2)))

    # image处理
    image_ocv = image_ocv.astype(np.uint8)
    image_ocv_2 = image_ocv[:,:,0:3]

    a = image_ocv_2.shape[0]
    b = image_ocv_2.shape[1]

    c = image_size.width
    d = image_size.height


    cv2.imshow("img",image_ocv)
    cv2.imshow("img2",image_ocv_2)
    cv2.imshow('depth',depth_map)
    cv2.waitKey(0)

    print(image_ocv.shape,depth_image_ocv.shape,depth_map.shape)

    img = np.hstack([image_ocv, depth_image_ocv])
    img = cv2.resize(img, (800, 400))
    cv2.imshow("ZED-depth", img)
    key = cv2.waitKey(5)
    if frame % 3 == 0:
        cv2.imwrite(os.path.join(img_path, f'{num}.jpg'), image_ocv)
        cv2.imwrite(os.path.join(depth_path, f'{num}.png'), depth_map)
        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
        num += 1

cv2.destroyAllWindows()
zed.close()
