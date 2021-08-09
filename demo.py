import numpy as np
import pyzed.sl as sl
import cv2
import os
import datetime
from configparser import ConfigParser
from ast import literal_eval

def get_config_section(conf_path = 'conf.ini',conf_type = 'DEFAULT'):
    config = ConfigParser()
    config.read(conf_path,encoding='utf8')
    conf_dict = dict(config.items(conf_type))
    for k,v in conf_dict.items():
        try:
            v_new = literal_eval(v)
            conf_dict[k] = v_new
        except:
            pass
    return conf_dict



class ZedGrapData:
    def __init__(self,conf):
        self.txt_path = conf['txt_path']
        self.depth_path = conf['depth_path']
        self.img_path = conf['img_path']

        self.make_data_dir()
        self.init_zed_cam()


    def make_data_dir(self):
        """
        构建采集数据的文件夹，主文件夹命名为data_20210730这种
        :return:
        """
        now_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        sub_dir = f'data/data_{now_time}'
        self.txt_path = os.path.join(sub_dir,self.txt_path)
        self.depth_path = os.path.join(sub_dir,self.depth_path)
        self.img_path = os.path.join(sub_dir,self.img_path)

        # 判断下是否存在存储路径
        for path_i in [self.depth_path,self.img_path]:
            if not os.path.exists(path_i):
                os.makedirs(path_i)

    def init_zed_cam(self):
        self.zed = sl.Camera()  # Create a ZED camera object
        input_type = sl.InputType()  # Set configuration parameters

        init = sl.InitParameters(input_t=input_type)  # 初始化
        init.camera_resolution = sl.RESOLUTION.HD2K  # 相机分辨率(默认-HD720) 弄成2K
        init.depth_mode = sl.DEPTH_MODE.ULTRA  # 深度模式  (默认-PERFORMANCE)
        init.coordinate_units = sl.UNIT.MILLIMETER  # 毫米级    (默认-MILLIMETER)

        # Open the camera
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            print('Cannot open the camera!')
            exit(-1)
        else:
            print('ZED camera init completed...')

    @staticmethod
    def img_transfer(depth,img):
        """
        将图像数据转成需要的形式，其中景深图为16位，前景图为8位
        :param depth: 由zed相机采集的景深图，其shape = (1242,2088)，浮点数
        :param img: 由zed相机采集的前景图，其shape = (1242,2088，4)
        :return:
        """
        # 1. 前景图处理
        img = img.astype(np.uint8)[:, :, 0:3]

        # 2. 景深图处理
        depth[np.isnan(depth)] = 0  # 将nan填充为0
        depth[np.isinf(depth)] = 65535 # 将inf填充为65535
        depth = depth.astype(np.uint16)
        depth = cv2.resize(depth, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
        return depth,img


    def run(self):
        image_size = self.zed.get_camera_information().camera_resolution
        img = sl.Mat()
        depth = sl.Mat()
        depth_show = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        key = ''
        num = 0
        frame = 0

        while key != 113:
            num += 1
            if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # 拿取图像
                self.zed.retrieve_image(img, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
                self.zed.retrieve_image(depth_show, sl.VIEW.DEPTH,sl.MEM.CPU, image_size)
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

                # 转成numpy形式的图像
                depth_np = depth.get_data()
                depth_show_np = depth_show.get_data()
                img_np = img.get_data()

                # 图像展示
                img_hstack = np.hstack([img_np, depth_show_np])
                img_hstack = cv2.resize(img_hstack, (1000, 600))
                cv2.imshow("depth_show", img_hstack)
                key = cv2.waitKey(5)

                # 转成需要的形式
                depth_transfered, img_tranfered = self.img_transfer(depth_np, img_np)

                if num % 3 == 0:
                    cv2.imwrite(os.path.join(self.img_path, f'{frame}.jpg'), img_tranfered)
                    cv2.imwrite(os.path.join(self.depth_path, f'{frame}.png'), depth_transfered)
                    timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)
                    frame += 1

        cv2.destroyAllWindows()
        self.zed.close()


if __name__ == '__main__':
    conf = get_config_section()
    zedCap = ZedGrapData(conf)
    zedCap.run()
