import numpy as np


def eyebrow_ver_len(l_coord, r_coord):
    """ The vertical length of the eyebrow """
    l_coord = np.array(l_coord).reshape((4, 2))
    r_coord = np.array(r_coord).reshape((4, 2))

    l_len = max(l_coord[:][0]) - min(l_coord[:][0])
    r_len = max(r_coord[:][0]) - min(r_coord[:][0])

    return l_len, r_len


def cal_distance(coord1, coord2):
    coord1 = coord1.flatten()
    coord2 = coord2.flatten()
    
    dis = np.sqrt(
        np.sum(np.square(coord1[0]-coord2[0])+np.square(coord1[1]-coord2[1])))
    return dis


def cal_angle(l_coord, min_coord, r_coord):
    """ The angle of the l_coord-min_coord-r_coord """
    cosine_value = (np.square(cal_distance(l_coord, min_coord)) + np.square(cal_distance(min_coord, r_coord)) -
                    np.square(cal_distance(l_coord, r_coord))) / (2 * cal_distance(l_coord, min_coord) * cal_distance(min_coord, r_coord))
    angle = np.arccos(cosine_value)
    return angle


def cal_changing_rate(past_value, next_value, time_period):
    """ calculate the rate of change in the discrete data """
    present_cr = (next_value - past_value) / (time_period)
    return present_cr


def cal_area(coord, i):
    """ calculate the area of a polygon using the vector product
    the coordinate should be listed anticlockwise
    i is the number of the sides
    """
    coord = np.array(coord).reshape((i, 2))
    temp_det = 0
    for idx in range(i - 1):
        temp = np.array([coord[idx], coord[idx+1]])
        temp_det += np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([coord[-1], coord[0]]))
    return temp_det*0.5


def generate_mocap_new_feature(f_present):
    f_present = np.array(f_present).astype(np.float)
    new_constant_feature = []
    # cal ver eyebrow len
    lbro_coord = np.array([[f_present[113], f_present[115]], [f_present[116], f_present[118]], [f_present[119], f_present[121]], [f_present[122], f_present[124]]])
    rbro_coord = np.array([[f_present[125], f_present[127]], [f_present[128], f_present[130]], [f_present[131], f_present[133]], [f_present[134], f_present[136]]])
    l_len, r_len = eyebrow_ver_len(lbro_coord, rbro_coord)
    new_constant_feature.append(l_len)
    new_constant_feature.append(r_len)

    # cal FH angle
    FH_angle_coord = np.array([[f_present[11],f_present[13]],[f_present[14], f_present[16]], [f_present[17], f_present[19]]])
    new_constant_feature.append(cal_angle(FH_angle_coord[0], FH_angle_coord[1], FH_angle_coord[2]))

    # LBM
    LBM_coord = np.array([[f_present[89], f_present[91]], [f_present[98], f_present[100]], [f_present[95], f_present[97]], [f_present[92], f_present[94]]])
    LBM_area = cal_area(LBM_coord, 4)
    new_constant_feature.append(LBM_area)

    # RBM
    RBM_coord = np.array([[f_present[101], f_present[103]], [f_present[104], f_present[106]], [f_present[107], f_present[109]], [f_present[110], f_present[112]]])
    RBM_area = cal_area(RBM_coord, 4)
    new_constant_feature.append(RBM_area)

    # MAX (RBRO1 to MH, LBRO1 to MH)
    LBRO1_c = np.array([[f_present[113],f_present[115]]])
    RBRO1_c = np.array([[f_present[125],f_present[127]]])
    MH_c = np.array([[f_present[74],f_present[76]]])
    LM = cal_distance(LBRO1_c, MH_c)
    RM = cal_distance(RBRO1_c, MH_c)
    new_constant_feature.append(max(LM, RM))

    # LC
    LC_coord = np.array([[f_present[20], f_present[22]], [f_present[23], f_present[25]], [f_present[32], f_present[34]], [f_present[38], f_present[40]], [f_present[41], f_present[43]], [f_present[35], f_present[37]], [f_present[29], f_present[31]]])
    LC_area = cal_area(LC_coord, 7)
    new_constant_feature.append(LC_area)

    # RC
    RC_coord = np.array([[f_present[44], f_present[46]], [f_present[53], f_present[55]], [f_present[59], f_present[61]], [f_present[65], f_present[67]], [f_present[62], f_present[64]], [f_present[56], f_present[58]], [f_present[47], f_present[49]]])
    RC_area = cal_area(RC_coord, 7)
    new_constant_feature.append(RC_area)

    # LC4 to average of (LNSTRL+TNOSE+MNOSE)
    LC4_c = np.array([[f_present[29], f_present[31]]])
    MNOSE_c = np.array([[f_present[77], f_present[79]]])
    LNSTRL = np.array([[f_present[80], f_present[82]]])
    TNOSE_c = np.array([[f_present[83], f_present[85]]])
    avg_len_l = (cal_distance(LC4_c,MNOSE_c) + cal_distance(LC4_c, LNSTRL) + cal_distance(LC4_c, TNOSE_c)) / 3
    new_constant_feature.append(avg_len_l)

    # RC4 to average of (RNSTRL+TNOSE+MNOSE)
    RC4_c = np.array([[f_present[53], f_present[55]]])
    MNOSE_c = np.array([[f_present[77], f_present[79]]])
    RNSTRL_c = np.array([[f_present[86], f_present[88]]])
    TNOSE_c = np.array([[f_present[83], f_present[85]]])
    avg_len_r = (cal_distance(RC4_c,MNOSE_c) + cal_distance(RC4_c, RNSTRL_c) + cal_distance(RC4_c, TNOSE_c)) / 3
    new_constant_feature.append(avg_len_r)

    # max(length of LC3 to LC4, length of RC3 to LR4)
    LC3_c = np.array([[f_present[26], f_present[28]]])
    RC3_c = np.array([[f_present[50], f_present[52]]])
    new_constant_feature.append(max(cal_distance(LC3_c, LC4_c), cal_distance(RC3_c, RC4_c)))

    # the length of MH to MNOSE
    new_constant_feature.append(cal_distance(MH_c, MNOSE_c))

    # cal LC angle
    LC_angle_coord = np.array([[f_present[41], f_present[43]],[f_present[38], f_present[40]], [f_present[26], f_present[28]]])
    LC_angle = cal_angle(LC_angle_coord[0], LC_angle_coord[1], LC_angle_coord[2])

    # cal RC angle
    RC_angle_coord = np.array([[f_present[50], f_present[52]],[f_present[62], f_present[64]], [f_present[65], f_present[67]]])
    RC_angle = cal_angle(RC_angle_coord[0], RC_angle_coord[1], RC_angle_coord[2])

    new_constant_feature.append(max(LC_angle, RC_angle))

    # the area of the MOU1~MOU8
    MOU_coord = np.array([[f_present[158], f_present[160]], [f_present[155], f_present[157]], [f_present[152], f_present[154]], [f_present[149], f_present[151]], [f_present[146], f_present[148]], [f_present[143], f_present[145]], [f_present[140], f_present[142]], [f_present[137], f_present[139]]])
    MOU_area = cal_area(MOU_coord, 8)
    new_constant_feature.append(MOU_area)
    
    # cal CH angle
    CH_angle_coord = np.array([[f_present[2],f_present[4]],[f_present[5], f_present[7]], [f_present[8], f_present[10]]])
    new_constant_feature.append(cal_angle(CH_angle_coord[0], CH_angle_coord[1], CH_angle_coord[2]))

    return new_constant_feature


def generate_cr_feature(f_onelabel):
    f_onelabel = np.array(f_onelabel)
    time = f_onelabel[:, 1]
    # FH_angle = f_onelabel[:][169]
    # LBM = f_onelabel[:][170]
    # RBM = f_onelabel[:][171]
    # LC_area = f_onelabel[:][173]
    # RC_area = f_onelabel[:][174]
    total_cr = []
    for idx in range(167, 182):
        if idx == 172:
            continue
        x = f_onelabel[:, idx]
        cr = []
        for j in range(1, (len(x) - 1)):
            cr.append(cal_changing_rate(x[j-1], x[j+1], time[j+1] - time[j-1]))
        cr_f = [cr[0]]
        cr_l = [cr[-1]]
        cr = (cr_f + cr + cr_l)
        total_cr.append(np.array(cr).astype(np.float))
    total_cr = np.array(total_cr)
    return total_cr.T
