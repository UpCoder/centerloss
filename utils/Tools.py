# -*- coding: utf-8 -*-
import numpy as np
import pydicom
import os
import SimpleITK as itk
from PIL import Image, ImageDraw
import Queue
import gc
import copy
from Config import Config
import glob
import cv2


def image_expand(image, size):
    image = np.asarray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    image = cv2.dilate(image, kernel)
    return image


def image_erode(image, size):
    image = np.asarray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    image = cv2.erode(image, kernel)
    return image

# 调整窗宽 窗位
def rejust_pixel_value(image):
    image = np.array(image)
    ww = np.float64(250)
    wc = np.float64(55)
    ww = max(1, ww)
    lut_min = 0
    lut_max = 255
    lut_range = np.float64(lut_max) - lut_min

    minval = wc - ww / 2.0
    maxval = wc + ww / 2.0
    image[image < minval] = minval
    image[image > maxval] = maxval
    to_scale = (minval <= image) & (image <= maxval)
    image[to_scale] = ((image[to_scale] - minval) / (ww * 1.0)) * lut_range + lut_min
    return image


# 读取单个DICOM文件
def read_file(file_name):
    header = pydicom.read_file(file_name)
    image = header.pixel_array
    image = header.RescaleSlope * image + header.RescaleIntercept
    return image


# 读取DICOM文件序列
def read_dicom_series(dir_name):
    print 'read dicom ', dir_name
    files = list(os.listdir(dir_name))
    files.sort()
    res = []
    for file in files:
        if file.endswith('DCM'):
            cur_file = os.path.join(dir_name, file)
            res.append(read_file(cur_file))
    return res


# 读取mhd文件
def read_mhd_image(file_path, rejust=False):
    header = itk.ReadImage(file_path)
    image = np.array(itk.GetArrayFromImage(header))
    if rejust:
        image[image < -70] = -70
        image[image > 180] = 180
        image = image + 70
    return np.array(image)


# 保存mhd文件
def save_mhd_image(image, file_name):
    print 'image type is ', type(image)
    header = itk.GetImageFromArray(image)
    itk.WriteImage(header, file_name)


# 一次读取多个ｍｈｄ文件，然后统一放缩至指定大小
def read_mhd_images(paths, new_size=None, avg_liver_values=None):
    images = []
    for index, path in enumerate(paths):
        # print path
        cur_image = read_mhd_image(path)
        cur_img = np.asarray(cur_image, np.float32)
        if avg_liver_values is not None:
            for i in range(0):
                cur_img = cur_img * cur_img
                cur_img = cur_img / avg_liver_values[index]
        if new_size is not None:
            cur_img = Image.fromarray(np.asarray(cur_img, np.float32))
            print new_size
            cur_img = cur_img.resize(new_size)
            #print np.shape(cur_img), path
            cur_image = np.array(cur_img)
        images.append(cur_image)
    return images


# 将灰度图像转化为RGB通道
def conver_image_RGB(gray_image):
    shape = list(np.shape(gray_image))
    image_arr_rgb = np.zeros(shape=[shape[0], shape[1], 3])
    image_arr_rgb[:, :, 0] = gray_image
    image_arr_rgb[:, :, 1] = gray_image
    image_arr_rgb[:, :, 2] = gray_image
    return image_arr_rgb


# 将一个矩阵保存为图片
def save_image(image_arr, save_path):
    if len(np.shape(image_arr)) == 2:
        image_arr = conver_image_RGB(image_arr)
    image = Image.fromarray(np.asarray(image_arr, np.uint8))
    image.save(save_path)


# 将图像画出来，并且画出标记的病灶
def save_image_with_mask(image_arr, mask_image, save_path):
    shape = list(np.shape(image_arr))
    image_arr_rgb = np.zeros(shape=[shape[0], shape[1], 3])
    image_arr_rgb[:, :, 0] = image_arr
    image_arr_rgb[:, :, 1] = image_arr
    image_arr_rgb[:, :, 2] = image_arr
    image = Image.fromarray(np.asarray(image_arr_rgb, np.uint8))
    image_draw = ImageDraw.Draw(image)
    [ys, xs] = np.where(mask_image != 0)
    miny = np.min(ys)
    maxy = np.max(ys)
    minx = np.min(xs)
    maxx = np.max(xs)
    ROI = image_arr_rgb[miny-1:maxy+1, minx-1:maxx+1, :]
    ROI_Image = Image.fromarray(np.asarray(ROI, np.uint8))

    for index, y in enumerate(ys):
        image_draw.point([xs[index], y], fill=(255, 0, 0, 128))
    if save_path is None:
        image.show()
    else:
        image.save(save_path)
        ROI_Image.save(os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split('.')[0]+'_ROI.jpg'))
        del image, ROI_Image
        gc.collect()
# 获取单位方向的坐标，
# 比如dim=2，则返回的数组就是[[-1, -1],...[1, 1]]
def get_direction_index(dim=3, cur_dir=[]):
    res = []
    for i in range(-1, 2):
        cur_dir.append(i)
        if dim != 1:
            res.extend(get_direction_index(dim-1, cur_dir))
            cur_dir.pop()
        else:
            res.append(copy.copy(cur_dir))
            cur_dir.pop()
    return res


# 验证数组arr1的值是否在arr_top 和 arr_dowm之间
# arr_top[i] > arr1[i] >= arr_down[i]
def value_valid(arr1, arr_top, arr_down):
    for index, item in enumerate(arr1):
        if arr_down[index] <= item < arr_top[index]:
            continue
        else:
            return False
    return True


# 将mask文件中的多个病灶拆分出来
def split_mask_image(total_mask_image_path, save_paths):
    directions = get_direction_index(dim=3)

    def find_connected_components(position, mask_image, flag):
        queue = Queue.Queue()
        points = []
        queue.put(position)
        while not queue.empty():
            cur_position = queue.get()
            points.append(cur_position)
            for direction in directions:
                new_z = cur_position[0] + direction[0]
                new_y = cur_position[1] + direction[1]
                new_x = cur_position[2] + direction[2]
                if value_valid([new_z, new_y, new_x], np.shape(mask_image), [0, 0, 0])\
                        and flag[new_z, new_y, new_x] == 1 \
                        and mask_image[new_z, new_y, new_x] != 0:
                    queue.put([new_z, new_y, new_x])
                    flag[new_z, new_y, new_x] = 0
        return points
    mask_image = read_mhd_image(total_mask_image_path)
    mask_image = np.array(mask_image)
    [z, y, x] = np.shape(mask_image)
    flag = np.ones(
        shape=[
            z, y, x
        ]
    )
    flag[np.where(mask_image == 0)] = 0
    index = 0
    for o in range(z):
        for n in range(y):
            for m in range(x):
                if mask_image[o, n, m] != 0 and flag[o, n, m] == 1:
                    flag[o, n, m] = 0
                    points = find_connected_components([o, n, m], mask_image, flag)
                    new_mask = np.zeros(
                        shape=[z, y, x]
                    )
                    for point in points:
                        new_mask[point[0], point[1], point[2]] = 1
                    save_mhd_image(new_mask, save_paths[index])
                    index += 1
                    print len(points)


# 根据Srrid判断所属的类别
def get_lesion_type_by_srrid(srrid):
    for key in Config.LESION_TYPE_RANGE.keys():
        for cur_range in Config.LESION_TYPE_RANGE[key]:
            if srrid in cur_range:
                return key
    return None


# 根据根目录读取NC，ART、PV三个phase的数据
def get_diff_phases_images(dir_path):
    images = {}
    for phase in Config.QIXIANGS:
        mhd_path = os.path.join(dir_path, phase, phase+'.mhd')
        print os.path.join(dir_path, phase)
        if os.path.exists(mhd_path):
            images[phase] = np.array(read_mhd_image(mhd_path))
        else:
            images[phase] = np.array(read_dicom_series(os.path.join(dir_path, phase)))
        if Config.ADJUST_WW_WC:
            images[phase] = rejust_pixel_value(images[phase])
    return images


# 根据目录读取LiverMask 和 所有的TumorMask
def get_total_masks(dir_path, dirs=['LiverMask', 'TumorMask']):
    tumors = {}
    if 'LiverMask' in dirs:
        liver_mask = {}
        cur_dir_path = os.path.join(dir_path, 'LiverMask')
        files = os.listdir(cur_dir_path)
        for cur_file in files:
            if not cur_file.endswith('.mhd'):
                continue
            phase_name = cur_file[cur_file.find('_', cur_file.find('_')+1) + 1: cur_file.find('.mhd')]
            liver_mask[phase_name] = read_mhd_image(os.path.join(cur_dir_path, cur_file))
        tumors['LiverMask'] = liver_mask
    if 'TumorMask' in dirs:
        tumors_mask = []
        cur_dir_path = os.path.join(dir_path, 'TumorMask')
        files = os.listdir(cur_dir_path)
        tumors_num = len(files) / 6
        for i in range(tumors_num):
            cur_tumor_mask = {}
            for phase in Config.QIXIANGS:
                mask_file_path = glob.glob(os.path.join(cur_dir_path, '*_' + phase + '_' + str(i+1) + '.mhd'))[0]
                cur_tumor_mask[phase] = read_mhd_image(mask_file_path)
            tumors_mask.append(cur_tumor_mask)
        tumors['TumorMask'] = tumors_mask
    return tumors


# 获取label的分布
def get_distribution_label(labels):
    min_value = np.min(labels)
    max_value = np.max(labels)
    my_dict = {}
    for label in labels:
        if label in my_dict:
            my_dict[label] += 1
        else:
            my_dict[label] = 0
    return my_dict

def compress22dim(image):
    shape = list(np.shape(image))
    if len(shape) == 3:
        return np.squeeze(image)
    else:
        return image
def shuffle_array(paths):
    '''
    将一个数组打乱
    :param paths: 数组
    :return: 打乱之后的数组
    '''
    paths = np.array(paths)
    indexs = range(len(paths))
    np.random.shuffle(indexs)
    return paths[indexs]

#　将数据打乱
def shuffle_image_label(images, labels):
    labels = np.array(labels)
    if len(images) == 2:
        random_index = range(len(images[0]))
    else:
        random_index = range(len(images))
    np.random.shuffle(random_index)
    labels = labels[random_index]
    new_images = []
    for cur_index in random_index:
        new_images.append(images[cur_index])
    return new_images, labels
def resize_image(image, size):
    image = Image.fromarray(np.asanyarray(image, np.uint8))
    return image.resize((size, size))
def get_boundingbox(mask_image):
    '''
    返回mask生成的bounding box
    :param mask_image:mask 文件
    :return:[xmin, xmax, ymin, ymax]
    '''
    xs, ys = np.where(mask_image == 1)
    return [
        np.min(xs),
        np.max(xs),
        np.min(ys),
        np.max(ys)
    ]


def cal_liver_average(mhd_image, liver_mask_image):
    liver_mask_image[liver_mask_image == 1] = 2
    mhd_image_copy = copy.copy(mhd_image)
    liver_mask_image[mhd_image_copy < 30] = 0
    # liver_mask_image[mhd_image_copy]
    return (1.0 * np.sum(mhd_image[liver_mask_image == 2])) / (1.0 * np.sum(liver_mask_image == 2))
# 将数据按照指定的方式排序
def changed_shape(image, shape):
    new_image = np.zeros(
        shape=shape
    )
    batch_size = shape[0]
    for z in range(batch_size):
        for phase in range(shape[-1]):
            if shape[-1] == 1:
                new_image[z, :, :, phase] = image[z]
            else:
                new_image[z, :, :, phase] = image[z, phase]
    del image
    gc.collect()
    return new_image


# 将一幅图像的mask外部全部标记为０
def mark_outer_zero(image, mask_image):
    def is_in(x, y, mask):
        sum1 = np.sum(mask[0:x, y])
        sum2 = np.sum(mask[x:, y])
        sum3 = np.sum(mask[x, 0:y])
        sum4 = np.sum(mask[x, y:])
        if sum1 != 0 and sum2 != 0 and sum3 != 0 and sum4 != 0:
            return True
        return False
    def fill_region(mask, x, y):
        queue = Queue.Queue()
        queue.put([x, y])
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        while not queue.empty():
            point = queue.get()
            for direction in directions:
                new_x = point[0] + direction[0]
                new_y = point[1] + direction[1]
                if value_valid([new_x, new_y], np.shape(mask), [0, 0]) and mask[new_x, new_y] == 0:
                    mask[new_x, new_y] = 1
                    queue.put([new_x, new_y])
        return mask
    [w, h] = np.shape(image)
    mask_image_copy = mask_image.copy()
    for i in range(w):
        # find = False
        for j in range(h):
            if mask_image_copy[i, j] == 0:
                if is_in(i, j, mask_image_copy):
                    print i, j
                    mask_image_copy[i, j] = 1
                    mask_image_copy = fill_region(mask_image_copy, i, j)
                    # find = True
                    image[np.where(mask_image_copy == 0)] = 0
                    return image, mask_image_copy
    print 'Error'
    return image, mask_image_copy


# 显示一幅图像
def show_image(image_arr, title=None):
    image = Image.fromarray(np.asarray(image_arr, np.uint8))
    image.show(title=title)


# 计算针对二分类错了多少个
def acc_binary_acc(logits, label):
    acc_count = 0.0
    logits = copy.copy(logits)
    label = copy.copy(label)
    logits = np.array(logits)
    label = np.array(label)
    logits[logits == 1] = 0
    logits[logits == 3] = 0
    logits[logits == 2] = 1
    logits[logits == 4] = 1

    label[label == 1] = 0
    label[label == 3] = 0
    label[label == 2] = 1
    label[label == 4] = 1
    for index, logit in enumerate(logits):
        if label[index] == logit:
            acc_count += 1
    return (1.0 * acc_count) / (1.0 * len(logits))

# 计算Ａｃｃｕｒａｃｙ，并且返回每一类最大错了多少个
def calculate_acc_error(logits, label, show=True):
    error_index = []
    error_dict = {}
    error_dict_record = {}
    error_count = 0
    error_record = []
    label = np.array(label).squeeze()
    logits = np.array(logits).squeeze()
    for index, logit in enumerate(logits):
        if logit != label[index]:
            error_count += 1
            if label[index] in error_dict.keys():
                error_dict[label[index]] += 1   # 该类别分类错误的个数加１
                error_dict_record[label[index]].append(logit)   # 记录错误的结果
            else:
                error_dict[label[index]] = 1
                error_dict_record[label[index]] = [logit]
            error_index.append(index)
            error_record.append(logit)
    acc = (1.0 * error_count) / (1.0 * len(label))
    if show:
        for key in error_dict.keys():
            print 'label is %d, error number is %d, all number is %d, acc is %g'\
                  % (key, error_dict[key], np.sum(label == key), 1-(error_dict[key]*1.0)/(np.sum(label == key) * 1.0))
            print 'error record　is ', error_dict_record[key]
    return error_dict, error_dict_record, acc, error_index, error_record


def get_shuffle_index(n):
    random_index = range(n)
    np.random.shuffle(random_index)
    return random_index


# 对所有病灶进行线性增强
def linear_enhancement(path='/home/give/PycharmProjects/MedicalImage/imgs/LiverAndLesion_bg/TRAIN'):
    image_dirs = os.listdir(path)
    for image_dir in image_dirs:
        for phase in ['nc', 'art', 'pv']:
            liver_path = os.path.join(path, image_dir, 'liver_' + phase + '.jpg')
            tumor_path = os.path.join(path, image_dir, 'tumor_' + phase + '.jpg')
            liver_image = Image.open(liver_path)
            tumor_image = Image.open(tumor_path)
            from Slice.TwoFolderMaxSlice.Slice_Base_Liver_Tumor import Liver_Tumor_Operations
            new_tumor = Liver_Tumor_Operations.tumor_linear_enhancement_only_tumor(tumor_image=np.array(tumor_image))
            new_path = os.path.join(path, image_dir, 'tumor_' + phase + '_enhancement.jpg')
            image = Image.fromarray(np.asarray(new_tumor, np.uint8))
            image.save(new_path)


def extract_avg_liver_dict(txt_path='/home/give/Documents/dataset/MedicalImage/MedicalImage/average_pixel_value.txt'):
    filed = open(txt_path)
    lines = filed.readlines()
    res_dict = {}
    for line in lines:
        split_res = line.split(',')
        srrid = split_res[0]
        srrid = int(srrid.split('-')[0])
        avg_liver = [float(split_res[i]) for i in range(1, 4)]
        res_dict[srrid] = avg_liver
    return res_dict


def calculate_tp(logits, labels):
    count = 0
    for index, logit in enumerate(logits):
        if logit == labels[index] and logit == 1:
            count += 1
    return count

def calculate_recall(logits, labels):
    tp = calculate_tp(logits=logits, labels=labels)
    recall = (tp * 1.0) / (np.sum(labels == 1) * 1.0)
    return recall


def calculate_precision(logits, labels):
    tp = calculate_tp(logits=logits, labels=labels)
    precision = (tp * 1.0) / (np.sum(logits == 1) * 1.0)
    return precision


def get_game_evaluate(logits, labels, argmax=None):
    logits = np.array(logits)
    labels = np.array(labels)
    if argmax is not None:
        logits = np.argmax(logits, argmax)
        labels = np.argmax(labels, argmax)
    recall = calculate_recall(logits=logits, labels=labels)
    precision = calculate_precision(logits=logits, labels=labels)
    f1_score = (2*precision*recall) / (precision + recall)
    return recall, precision, f1_score


def convert2depthlaster(mask_image):
    '''
    将数组调整为ｄｅｐｔｈ在第三个通道
    :param mask_image: depth width height
    :return:  width height depth
    '''
    mask_image = np.array(mask_image)
    shape0 = list(np.shape(mask_image[0]))
    for i in range(len(mask_image)):
        if shape0 != list(np.shape(mask_image[i])):
            print 'The size of each channal is not equal.'
            return mask_image
    res = np.zeros([shape0[0], shape0[1], len(mask_image)])
    for i in range(len(mask_image)):
        res[:, :, i] = mask_image[i, :, :]
    return res

def test_show_regression(type_name='HEM'):
    from glob import glob
    '''
    可视化不同的类型病灶ｒｅｇｒｅｓｓｉｏｎ之后的结果
    :return:
    '''
    data_dir = '/home/give/Documents/dataset/MedicalImage/MedicalImage/SL_TrainAndVal/train/1887735_2842841_0_0_3'
    phasenames = ['NC', 'ART', 'PV']
    mhd_images = []
    for phasename in phasenames:
        image_path = glob(os.path.join(data_dir, phasename + '_Image*.mhd'))[0]
        mask_path = os.path.join(data_dir, phasename + '_Registration.mhd')
        mhd_image = read_mhd_image(image_path, rejust=True)
        mhd_image = np.squeeze(mhd_image)
        mask_image = read_mhd_image(mask_path)
        mask_image = np.squeeze(mask_image)
        [xmin, xmax, ymin, ymax] = get_boundingbox(mask_image)
        # xmin -= 15
        # xmax += 15
        # ymin -= 15
        # ymax += 15
        mask_image = mask_image[xmin: xmax, ymin: ymax]
        mhd_image = mhd_image[xmin: xmax, ymin: ymax]
        mhd_image[mask_image != 1] = 0
        mhd_images.append(mhd_image)
        img = Image.fromarray(np.asarray(mhd_image, np.uint8))
        img.save('./' + type_name + '_' + phasename + '.jpg')
    mhd_images = convert2depthlaster(mhd_images)
    img = Image.fromarray(np.asarray(mhd_images, np.uint8))
    img.save('./' + type_name + '.jpg')
    show_image(mhd_images)

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**1
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out


def check_save_path(path):
    '''
    检查将要存储文件的路径,判断该路径所在的父文件夹是否存在，如不存在则建立
    :param path:
    :return:
    '''
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        print 'mkdir: ', path_dir
        os.makedirs(path_dir)


def split_array(arr, num, rate=None):
    '''
    将一个数组拆分成多个数组
    :param arr: 待拆分的数组
    :param num: 需要拆分成多少个
    :param rate: None 代表的是均匀划分，否则按照规则来划分(NUM-1)
    :return:
    '''

    result = []
    length = len(arr)
    if rate is None:
        pre_num = length / num
        for i in range(num):
            if i != (num - 1):
                cur_group = arr[i * pre_num: (i + 1) * pre_num]
            else:
                cur_group = arr[i * pre_num: length]
            result.append(cur_group)
        return result
    else:
        start = 0
        for i in range(num):
            if i != (num-1):
                cur_num = int(length * rate[i])
                result.append(arr[start: start + cur_num])
                start = start + cur_num
            else:
                result.append(arr[start:])
        return result


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return [np.eye(nb_classes)[int(target)] for target in targets]

if __name__ == '__main__':
    test_show_regression()