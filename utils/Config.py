# -*- coding: utf-8 -*-


class Config:
    QIXIANGS = ['NC', 'ART', 'PV']
    LESION_TYPE = ['CYST', 'FNH', 'HCC', 'HEM', 'METS']
    LABEL_NUMBER_MAPPING = {
        'CYST': 0,
        'FNH': 1,
        'HCC': 2,
        'HEM': 3,
        'METS': 4,
    }
    ADJUST_WW_WC = True     # 判断是否需要调整窗窗位
    NPY_SAVE_PATH = 'E:\\work\\MedicalImage\\data'    # 存储的NPY文件的地址
    RECOVER = True
    BOUNDING_BOX_NOT_LEASION = 0    # 0代表补0 1代表补肝脏密度
    EXCEL_PATH = '/home/give/Documents/dataset/MedicalImage/MedicalImage/data_origanal.xlsx'
    IMAGE_SAVE_PATH = '/home/give/PycharmProjects/MedicalImage/imgs'
    LESION_TYPE_RANGE = {
        'CYST': [range(0, 19), range(100, 120)],
        'FNH': [range(19, 29), range(120, 132)],
        'HCC': [range(29, 39), range(132, 152)],
        'HEM': [range(39, 49), range(152, 172)],
        'METS': [range(49, 59), range(172, 185)],
    }
    DATASET_PATH = '/home/give/Documents/dataset/MedicalImage/MedicalImage'
    MaxSlice_Base = {
        'BASE_DATA_PATH': '/home/give/PycharmProjects/MedicalImage/data',
        'splited': {
            'statue': True,
            'ranges': [
                range(0, 88),
                range(99, 200)
            ],
            'save_paths': [
                [
                    '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Image1.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Mask1.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Label1.npy'
                ],
                [
                    '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Image2.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Mask2.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Label2.npy'
                ]
            ]
        },
        'VALIDATION_DISTRIBUTION': [
            5,  # CYST
            5,  # FNH
            5,  # HCC
            5,  # HEM
            5  # METS
        ]
    }
    MaxSliceDataPATH = [
        '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Image.npy',
        '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Mask.npy',
        '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Label.npy'
    ]

    MaxSlice_Resize = {

        'SAVE_PATH': [
            '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Resize_ROI.npy',
            '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Resize_Label.npy'
        ],
        'RESIZE': [
            45, 45
        ],
        'IMAGE_SAVE_PATH': '/home/give/PycharmProjects/MedicalImage/imgs/resized',
    }

    MaxSlice_Resize_Zero = {
        'IMAGE_SAVE_PATH': '/home/give/PycharmProjects/MedicalImage/imgs/resized_zero',
        'SAVE_PATH': [
            '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Resize_Zero_ROI.npy',
            '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Resize_Zero_Label.npy'
        ],
        'RESIZE': [
            45, 45
        ],
    }

    MaxSlice_R_Z_AVG = {
        'IMAGE_SAVE_PATH': '/home/give/PycharmProjects/MedicalImage/imgs/resized_zero_avg',
    }

    MaxSlice_Multi_Scale = {
        'sizes': [
            [45, 45],
            [20, 20],
            [100, 100]
        ]
    }

    MaxSlice_Multi_Scale_Zero = {
        'sizes': [
            [45, 45],
            [20, 20],
            [100, 100]
        ],
        'NPY_SAVE_PATH': [
            [
                '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Multi_Scale_Zero/scale1.npy',
                '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Multi_Scale_Zero/label1.npy'
            ],
            [
                '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Multi_Scale_Zero/scale2.npy',
                '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Multi_Scale_Zero/label2.npy'
            ],
            [
                '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Multi_Scale_Zero/scale3.npy',
                '/home/give/PycharmProjects/MedicalImage/data/MaxSlice_Multi_Scale_Zero/label3.npy'
            ]
        ],
        'IMAGE_SAVE_PATH': '/home/give/PycharmProjects/MedicalImage/imgs/resize_zero_multi_scale'
    }

    MaxSlice_Liver_Base = {
        'BASE_DATA_PATH': '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver',
        'splited': {
            'statue': True,
            'ranges': [
                range(0, 88),
                range(99, 200)
            ],
            'save_paths': [
                [
                    '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Image1.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Mask1.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Label1.npy'
                ],
                [
                    '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Image2.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Mask2.npy',
                    '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Label2.npy'
                ]
            ]
        },
        'VALIDATION_DISTRIBUTION': [
            5,  # CYST
            5,  # FNH
            5,  # HCC
            5,  # HEM
            5  # METS
        ]
    }
    MaxSlice_Liver_Resize_Zero = {
        'SAVE_NPY_PATH': [
            '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Resize_Zero_ROI.npy',
            '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/MaxSlice_Resize_Zero_Label.npy',
            '/home/give/PycharmProjects/MedicalImage/data/Sub_Liver/exist.npy',
        ],
        'IMAGE_SAVE_PATH': '/home/give/PycharmProjects/MedicalImage/imgs/resize_liver_zero'
    }