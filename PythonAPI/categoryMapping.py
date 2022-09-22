coco_cid_to_FACSED_cid_map = {
    1: 793,
    2: 94,
    3: 207,
    4: 703,
    5: 3,
    6: 173,
    7: 1115,
    8: 1123,
    9: 118,
    10: 1112,
    11: 445,
    13: 1019,
    14: 766,
    15: 90,
    16: 99,
    17: 225,
    18: 378,
    19: 569,
    20: 943,
    21: 80,
    22: 422,
    23: 76,
    24: 1202,
    25: 496,
    27: 34,
    28: 1133,
    31: 35,
    32: 138,
    33: 36,
    34: 474,
    35: 964,
    36: 976,
    37: 41,
    38: 611,
    39: 58,
    40: 60,
    41: 962,
    42: 1037,
    43: 1079,
    44: 133,
    46: 1190,
    47: 344,
    48: 469,
    49: 615,
    50: 1000,
    51: 139,
    52: 45,
    53: 12,
    54: 912,
    55: 735,
    56: 154,
    57: 217,
    58: 1219,
    59: 816,
    60: 387,
    61: 183,
    62: 232,
    63: 982
}

hypersim_cid_to_FACSED_cid_map = {
    1: 1204,
    2: 1205,
    3: 181,
    4: 77,
    5: 232,
    6: 982,
    7: 1050,
    8: 1206,
    9: 1207,
    10: 1208,
    11: 748,
    12: 1209,
    13: 1218,
    14: 361,
    15: 1210,
    16: 350,
    17: 395,
    18: 804,
    19: 694,
    20: 386,
    21: 1211,
    22: 1212,
    23: 127,
    24: 421,
    25: 1077,
    26: 719,
    27: 1108,
    28: 957,
    29: 143,
    30: 1213,
    31: 793,
    32: 1214,
    33: 1097,
    34: 961,
    35: 626,
    36: 68,
    37: 953,
    38: 1215,
    39: 1216,
    40: 1217
}

human_parsing_cid_to_FACSED_cid_map = {
    1: 1220,
    2: 1221,
    3: 1222,
    4: 1223
}

def map_hypersim_cid(hypersim_cid):
    """
    @params
    hypersim_cid : int
        The semantic category id of the hypersim dataset (NYU40 label id)
    @returns
    int
        The mapped category id for FACSED
    """
    return hypersim_cid_to_FACSED_cid_map[hypersim_cid]

def map_coco_cid(coco_cid):
    """
    @params
    coco_cid : int
        The semantic category id of the COCO dataset
    @returns
    int
        The mapped category id for FACSED
    """
    return coco_cid_to_FACSED_cid_map[coco_cid]

def map_human_parsing_cid(human_parsing_cid):
    """
    @params
    human_parsing : int
        The semantic category id of the human parsing dataset
    @returns
    int
        The mapped category id for FACSED
    """
    return human_parsing_cid_to_FACSED_cid_map[human_parsing_cid]
