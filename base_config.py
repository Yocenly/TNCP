# -*- coding: utf-8 -*-
# @Time    : 2022/11/25 15:58
# @Author  : lv yuqian
# @File    : base_config.py
# @Email   : lvyuqian_email@163.com
NONE_SUPPORT = 0
WEAK_SUPPORT = 1
STRONG_SUPPORT = 2
GENERAL_SUPPORT = 3

dataset_names = ['TVShow', 'LastFM', 'Facebook', 'DeezerEU', 'HepPh', 'AstroPh', 'CondMat',
                 'USAir', 'USPower', 'EDU', 'Indo', 'Arabic', 'Gowalla', 'Citeseer', 'Google', 'RoadNet']

method_names = ['RED', 'RND', 'KNM', 'SV', 'TNC', 'ATNC']

DATASET_NAME = 'soc_advogato'

TIME_DATA = {
    'RED': {
        'TVShow': 153.9833, "LastFM": 274.4071, 'Facebook': 5227.3609, 'DeezerEU': 5039.9592, 'Gowalla': 0,
        'HepPh': 1822.0215, 'AstroPh': 4435.6485, 'CondMat': 2961.5092, 'Citeseer': 0,
        'USAir': 0.7645, 'USPower': 76.2927, 'RoadNet': 0,
        'EDU': 84.373, 'Indo': 1300.1213, 'Arabic': 0, 'Google': 0,
    },
    'RND': {
        'TVShow': 4.0487, "LastFM": 6.4717, 'Facebook': 71.5931, 'DeezerEU': 51.5023, 'Gowalla': 2010.5024,
        'HepPh': 15.3183, 'AstroPh': 43.1637, 'CondMat': 22.8171, 'Citeseer': 2164.6183,
        'USAir': 0.3172, 'USPower': 2.4715, 'RoadNet': 0,
        'EDU': 6.3479, 'Indo': 6.9505, 'Arabic': 537.0778, 'Google': 91031.1019,
    },
    'KNM': {
        'TVShow': 53.4286, "LastFM": 120.7546, 'Facebook': 1408.8041, 'DeezerEU': 4578.3134, 'Gowalla': 0,
        'HepPh': 255.6605, 'AstroPh': 722.9806, 'CondMat': 1718.5364, 'Citeseer': 0,
        'USAir': 1.7575, 'USPower': 323.828, 'RoadNet': 0,
        'EDU': 117.38, 'Indo': 308.8882, 'Arabic': 33325.8677, 'Google': 0,
    },
    'SV': {
        'TVShow': 34.7373, "LastFM": 49.0364, 'Facebook': 527.5443, 'DeezerEU': 208.0432, 'Gowalla': 3338.7493,
        'HepPh': 1251.8854, 'AstroPh': 667.0024, 'CondMat': 194.5611, 'Citeseer': 2603.5115,
        'USAir': 2.8399, 'USPower': 11.5037, 'RoadNet': 53088.8177,
        'EDU': 11.5234, 'Indo': 115.6866, 'Arabic': 10386.5748, 'Google': 21102.37915,
    },
    'TNC': {
        'TVShow': 36.842, "LastFM": 156.8493, 'Facebook': 1192.3135, 'DeezerEU': 3963.7506, 'Gowalla': 0,
        'HepPh': 230.3329, 'AstroPh': 597.3365, 'CondMat': 1246.5723, 'Citeseer': 0,
        'USAir': 1.8701, 'USPower': 339.9841, 'RoadNet': 0,
        'EDU': 157.9355, 'Indo': 173.9469, 'Arabic': 30670.3935, 'Google': 0,
    },
    'ATNC': {
        'TVShow': 7.21329, "LastFM": 6.61722, 'Facebook': 86.54346, 'DeezerEU': 14.00491, 'Gowalla': 1264.83832,
        'HepPh': 57.196897, 'AstroPh': 73.26963, 'CondMat': 10.76043, 'Citeseer': 238.1117,
        'USAir': 0.91244, 'USPower': 1.33094, 'RoadNet': 486.12759,
        'EDU': 0.94899, 'Indo': 11.7986, 'Arabic': 201.10644, 'Google': 2214.96579,
    },
}

CIS_DATA = {
    'ORI': {
        'TVShow': 3.60000, "LastFM": 5.30628, 'Facebook': 8.03915, 'DeezerEU': 4.19929, 'Gowalla': 4.11160,
        'HepPh': 4.01997, 'AstroPh': 6.56763, 'CondMat': 4.05706, 'Citeseer': 3.05694,
        'USAir': 6.05882, 'USPower': 1.64299, 'RoadNet': 1.86045,
        'EDU': 2.47377, 'Indo': 1.32374, 'Arabic': 1.28501, 'Google': 4.71771,
    },
    'ATNC': {
        'TVShow': 2.23590, "LastFM": 2.96335, 'Facebook': 4.60053, 'DeezerEU': 3.64876, 'Gowalla': 2.86114,
        'HepPh': 1.59235, 'AstroPh': 3.52929, 'CondMat': 2.89991, 'Citeseer': 1.90089,
        'USAir': 2.58824, 'USPower': 1.61728, 'RoadNet': 1.85833,
        'EDU': 2.45167, 'Indo': 1.22011, 'Arabic': 1.23581, 'Google': 3.04452,
    },
    'SV': {
        'TVShow': 2.84103, "LastFM": 3.60471, 'Facebook': 6.69573, 'DeezerEU': 4.07633, 'Gowalla': 3.25961,
        'HepPh': 1.92180, 'AstroPh': 4.24175, 'CondMat': 3.77362, 'Citeseer': 3.02556,
        'USAir': 4.00000, 'USPower': 1.63894, 'RoadNet': 1.86044,
        'EDU': 2.46849, 'Indo': 1.31141, 'Arabic': 1.28204, 'Google': 4.65759,
    },
}