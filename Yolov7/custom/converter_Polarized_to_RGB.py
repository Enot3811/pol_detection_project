import cv2
import numpy as np
from timeit import Timer
import time
import os
import pickle
import math
from math import cos as cos
from math import asin as asin
from math import sin as sin
from math import radians as d2r
from math import degrees as r2d


file_list_all=os.listdir("./dataset_POL/")
file_list = []
for i in file_list_all:
   if i.endswith('.npy'):
      file_list.append(i)

A1 = 90
A2 = 45
A3 = 135
A4 = 0



def Sum_Vect(A1, F1, A2, F2):
    if  F1 < F2:
        FF = F2
        F2 = F1
        F1 = FF
        FF = A2
        A2 = A1
        A1 = FF
    A = A2 - A1
    AF = 180 - A
    R = (F1**2 + F2**2 - 2*F1*F2*cos(d2r(AF)))**0.5
    if R == 0:

       AR = A2
       return AR, R
    else:
       AR = d2r(A1) + asin(F2*sin(d2r(AF))/R)
    if AF == 0 and F2 > F1:
       AR = AR + d2r(180)
    return AR, R

def Sum_Vect_All(A1, F1, A2, F2, A3, F3, A4, F4):
    AR_1, R_1 = Sum_Vect(A1, F1, A2, F2)
    AR_2, R_2 = Sum_Vect(r2d(AR_1), R_1, A3, F3)
    AR_3, R_3 = Sum_Vect(r2d(AR_2), R_2, A4, F4)

    return AR_3, R_3 


# Razbienie to channels
def split_RAW_image_to_channels(image):
    ch_90 = image[::2, ::2].astype(np.float32)
    ch_45 = image[::2, 1::2].astype(np.float32)
    ch_135 = image[1::2, ::2].astype(np.float32)
    ch_0 = image[1::2, 1::2].astype(np.float32)
    return ch_0, ch_45, ch_90, ch_135


# Vichislenie param Stocks
def calc_Stocks_param_old(ch_0, ch_45, ch_90, ch_135):
    # param Stocks
    S_0 = ch_0 + ch_90  # polariz 0.0<=S_0<=1.0  nepolariz 0.0<=S_0<=2.0
    Mask = np.nonzero(S_0 == 0)
    # избегаем деления на ноль в дальнейших расчетах
    S_0[Mask[0], Mask[1]] = 0.00000000000000000000000000001
    #print('np.min(S_0)', np.min(S_0))
    #print('np.max(S_0)', np.max(S_0))
    S_1 = ch_0 - ch_90  # polariz -1.0<=S_1<=1.0 nepolariz -1.0<=S_1<=1.0
    #print('np.min(S_1)', np.min(S_1))
    #print('np.max(S_1)', np.max(S_1))
    S_2 = ch_45 - ch_135  # polariz -1.0<=S_2<=1.0 nepolariz -1.0<=S_2<=1.0
    #print('np.min(S_2)', np.min(S_2))
    #print('np.max(S_2)', np.max(S_2))
    return S_0, S_1, S_2


# AoLP
def calc_AoLP(S_1, S_2):
    AoLP = 0.5 * np.arctan((S_2 / S_1))
    AoLP = AoLP * 255
    AoLP = cv2.convertScaleAbs(AoLP)

    return AoLP

# AoLP
def calc_AoLP_1(S_1, S_2):
    AoLP = 0.5 * np.arctan2(S_2, S_1)
    AoLP = cv2.convertScaleAbs(AoLP)
    return AoLP

# DoLP
def calc_DoLP(S_0, S_1, S_2):
    DoLP = ((S_1**2 + S_2**2)**0.5) / S_0
    DoLP = DoLP * (255 / np.max(DoLP))
    DoLP = DoLP.astype(np.uint8)
    return DoLP


# Poluchenie blikov
def give_anblik_image(ch_0, ch_45, ch_90, ch_135):
    #Reflect (bliki) delete on image
    reflect_1 = np.minimum(ch_90, ch_45)
    reflect_2 = np.minimum(ch_0, ch_135)
    reflect = np.minimum(reflect_1, reflect_2)
    return reflect

def give_max_blik_image(ch_0, ch_45, ch_90, ch_135):
    #max blick
    reflect_1 = np.maximum(ch_90, ch_45)
    reflect_2 = np.maximum(ch_0, ch_135)
    an_reflect = np.maximum(reflect_1, reflect_2)
    return an_reflect


def give_magic_1(ch_0, ch_45, ch_90, ch_135, reflect):
    #!MAGIC!MAGIC!MAGIC!
    ch_90_anmin = ch_90 - reflect
    ch_45_anmin = ch_45 - reflect
    ch_135_anmin = ch_135 - reflect
    ch_0_anmin = ch_0 - reflect


    result_magic_1 = ch_90_anmin + ch_45_anmin + ch_135_anmin + ch_0_anmin
    #result_magic_1 = cv2.convertScaleAbs(result_magic_1)
    result_magic_1 = result_magic_1.astype(np.uint8)

    return result_magic_1

def give_magic_2(ch_0, ch_45, ch_90, ch_135, reflect):
    #!MAGIC!MAGIC!MAGIC!!!
    ch_90_anmin = ch_90 - reflect
    ch_45_anmin = ch_45 - reflect
    ch_135_anmin = ch_135 - reflect
    ch_0_anmin = ch_0 - reflect
    result_magic_2_1 = np.maximum(ch_90_anmin, ch_45_anmin)
    result_magic_2_2 = np.maximum(ch_0_anmin, ch_135_anmin)
    result_magic_2 = np.maximum(result_magic_2_1, result_magic_2_2)

    return result_magic_2


def sum_by_channel(ch_0, ch_45, ch_90, ch_135, reflect):
    reflect = reflect.astype(np.float32)
    ch_90 = ch_90.astype(np.float32)
    ch_45 = ch_45.astype(np.float32)
    ch_135 = ch_135.astype(np.float32)
    ch_0 = ch_0.astype(np.float32)

    ch_90_anmin = ch_90 - reflect
    ch_45_anmin = ch_45 - reflect
    ch_135_anmin = ch_135 - reflect
    ch_0_anmin = ch_0 - reflect

    sum_ch_90 = np.sum(ch_90_anmin)
    sum_ch_45 = np.sum(ch_45_anmin)
    sum_ch_135 = np.sum(ch_135_anmin)
    sum_ch_0 = np.sum(ch_0_anmin)
    sum_all = np.array([sum_ch_90,  sum_ch_45,  sum_ch_135,  sum_ch_0])
    return sum_all, sum_ch_90, sum_ch_45, sum_ch_135, sum_ch_0 # float


def give_background_image_ones(AoLP):
    background_image = np.ones((AoLP.shape[0], AoLP.shape[1]))
    background_image = background_image*255
    background_image = background_image.astype('uint8')  
    return background_image 


def give_threshold_image(AoLP):
    ret, threshold_image = cv2.threshold(AoLP, 0,255,0)
    test_AoLP = cv2.bitwise_not(threshold_image)
    return test_AoLP


def find_contours(test_AoLP, background_image):
    #contours_1, hierarchy = cv2.findContours(test_AoLP, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_1, hierarchy = cv2.findContours(test_AoLP, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print('contours_1', len(contours_1))    
    count = 0
    if len(contours_1) != 0:
       if len(contours_1) == 1:
          contours_1 = contours_1[0]  
       else:
          res = np.copy(background_image)
          #res_1 = np.copy(background_image)
          #res_2 = np.copy(background_image)
          c_max = np.array([])
          for c in contours_1:
               area = cv2.contourArea(c)
               #print('area', area)
               if count == 0:
                  c_max = c
               else:
                  if c.shape[0] > c_max.shape[0]:
                     c_max = c
               
               if area > 50:

                  res = cv2.fillPoly(res, pts = [c], color = (128,128,128))
                  #res_1 = cv2.fillPoly(res_1, pts = [c], color = (128,128,128))
                  #res = cv2.drawContours(res, c, -1, (0,0,0), 1)
                  #res = cv2.drawContours(res, c, -1, (0,0,0), 1)
                  count = count + 1
    #print ('count', count)
    file_name_AoLP = end_frame +'_res_' + '.png'
    cv2.imwrite(file_name_AoLP, res)
    #res_2 = cv2.applyColorMap(res_2, cv2.COLORMAP_PINK)   
    #cv2.imshow('test_AoLP', test_AoLP)
    #cv2.imshow('res', res)
    return contours_1, res, area, count






    if np.argmax(sum_all) == 0:
       print('max_sum_ch', '90')
    if np.argmax(sum_all) == 1:
       print('max_sum_ch', '45')
    if np.argmax(sum_all) == 2:
       print('max_sum_ch', '135')
    if np.argmax(sum_all) == 3:
       print('max_sum_ch', '0')
  
    sum_all_sort = np.sort(sum_all)

    if sum_all_sort[-1] == sum_ch_90:
       print('max_sum__angle', '90')
       print('max_sum_ch', sum_ch_90)
    if sum_all_sort[-1] == sum_ch_45:
       print('max_sum_angle', '45')
       print('max_sum_ch', sum_ch_45)
    if sum_all_sort[-1] == sum_ch_135:
       print('max_sum_angle', '135')
       print('max_sum_ch', sum_ch_135)
    if sum_all_sort[-1] == sum_ch_0:
       print('max_sum__angle', '0')
       print('max_sum_ch', sum_ch_0)

    if sum_all_sort[-2] == sum_ch_90:
       print('sec_max_sum_angle', '90')
       print('sec_max_sum_ch', sum_ch_90)
    if sum_all_sort[-2] == sum_ch_45:
       print('sec_max_sum_angle', '45')
       print('sec_max_sum_ch', sum_ch_45)
    if sum_all_sort[-2] == sum_ch_135:
       print('sec_max_sum_angle', '135')
       print('sec_max_sum_ch', sum_ch_135)
    if sum_all_sort[-2] == sum_ch_0:
       print('sec_max_sum_angle', '0')
       print('sec_max_sum_ch', sum_ch_0)


    AR, R = (A1, sum_ch_90, A2, sum_ch_45, A3, sum_ch_135, A4, sum_ch_0) 
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!AR', r2d(AR))

 
    ######################################
    ########NNNN NNNNNN NNNNNNNNN#########
    ######################################

   
 

def line(vx, vy, x, y, res_1, res_2):
    rows, cols = res_1.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(c_max, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((- x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    
    ############
    line = cv2.line(res_1, (cols-1, righty), (0, lefty), (0,0,0), 3) 

    ############
    line_1 = cv2.line(res_2, (cols-1, righty), (0, lefty), (0,0,0), 3)

    x_1 = cols-1
    y_1 = righty
    x_2 = 0
    y_2 = lefty

    k = (y_2 - y_1) / (x_2 - x_1) # tan(fi)
    print('k', k)

    k_rad = np.arctan(k) #fi
    print('k_rad', k_rad)

    k_deg = np.rad2deg(k_rad)
    print('k_deg', end_frame, k_deg)

    h = res_2.shape[0]
    w = res_2.shape[1]
    x_center = int(w/2)
    y_center = int(h/2)
    cv2.circle(res_2, (x_center, y_center), 10, (128,128,128), -1)
    l = 100
    l_1 = 150
    y = l * k
    print('y', y)
    y = l * np.tan(k_rad)
    print('y', y)
    line_vect_road = cv2.line(res_2, (x_center - l, y_center - int(y)), (x_center, y_center), (0,0,0), 3) ############


    pfi_sun = 60
    y = l * np.tan(np.deg2rad(180) - (np.deg2rad(pfi_sun) - k_rad))
    line_vect_sun = cv2.line(res_2, (x_center - l, y_center - int(y)), (x_center, y_center), (128,128,128), 3) ############


    pfi_AR = r2d(AR)
    y = l_1 * np.tan(np.deg2rad(pfi_AR) + (np.deg2rad(90) )) 
    line_vect_AR = cv2.line(res_2, (x_center - l_1, y_center - int(y)), (x_center, y_center), (50,50,50), 3) ############

    return  line, line_1, line_vect_road, line_vect_sun, line_vect_AR
    


    print ('count', count)
    '''
    file_name_AoLP = end_frame +'_res_' + '.png'
    cv2.imwrite(file_name_AoLP, res)
    cv2.imshow('test_AoLP', test_AoLP)
    cv2.imshow('res', res)
    cv2.waitKey(10)
    '''

    res_2 = cv2.applyColorMap(res_2, cv2.COLORMAP_PINK)





    file_name_AoLP = end_frame +'_line_' + '.png'
    cv2.imwrite(file_name_AoLP, line)

    file_name_AoLP = end_frame +'_line_1' + '.png'
    cv2.imwrite(file_name_AoLP, res_2)

    file_name_AoLP = end_frame +'magic_1' + '.png'
    cv2.imwrite(file_name_AoLP, result_magic_1)

    file_name_AoLP = end_frame +'magic_2' + '.png'
    cv2.imwrite(file_name_AoLP, result_magic_2)

    #cv2.imshow('test_AoLP', test_AoLP)
    #cv2.imshow('line', line)
    cv2.waitKey(1)


'''
laplacian_AoLP = cv2.Laplacian(AoLP, cv2.CV_64F)
SobelX_AoLP = cv2.Sobel(AoLP, cv2.CV_8U, 1, 0, ksize = 5)
SobelY_AoLP = cv2.Sobel(AoLP, cv2.CV_8U, 0, 1, ksize = 5)

cv2.imshow('SobelX_AoLP', SobelX_AoLP)
cv2.imshow('SobelY_AoLP', SobelY_AoLP)
'''



#cv2.imshow('DoLP_0', DoLP_0)
#cv2.imshow('reflect', reflect)

#cv2.imshow('test_DoLP', test_DoLP)
#cv2.imshow('test_AoLP', test_AoLP)


#file_name_DoLP = end_frame +'_DoLP_' + '.png'
#cv2.imwrite(file_name_DoLP, test_DoLP)

for i in range(len(file_list) - 1643):
    end_frame = ("./dataset_POL/" + file_list[i]) 
    raw_image = np.load(end_frame)
    image_size = raw_image.shape
    #print('image_size', image_size)
    image = np.copy(raw_image)
    ch_0, ch_45, ch_90, ch_135 = split_RAW_image_to_channels(image)
    S_0, S_1, S_2 = calc_Stocks_param_old(ch_0, ch_45, ch_90, ch_135)
    
    #AoLP = calc_AoLP(S_1, S_2)
    #pseudo_RGB = cv2.applyColorMap(AoLP, cv2.COLORMAP_JET)
    
    #DoLP = calc_DoLP(S_0, S_1, S_2)
    #pseudo_RGB = cv2.applyColorMap(DoLP, cv2.COLORMAP_JET)

    reflect = give_anblik_image(ch_0, ch_45, ch_90, ch_135)
    #result_magic_1 = give_magic_1(ch_0, ch_45, ch_90, ch_135, reflect)
    #result_magic_2 = give_magic_2(ch_0, ch_45, ch_90, ch_135, reflect)

    pseudo_RGB = reflect
    


    
    end_frame = end_frame.replace('.npy', '')

    end_frame = ("./Stocks_param/" + file_list[i])
    #file_name_pseudo_RGB = end_frame + 'Stocks_param' + '.jpg'
    #cv2.imwrite(file_name_pseudo_RGB, pseudo_RGB)
    
    '''
    file_name_ch_0 = end_frame + 'ch_0' + '.png'
    cv2.imwrite(file_name_ch_0, ch_0)

    file_name_ch_45 = end_frame + 'ch_45' + '.png'
    cv2.imwrite(file_name_ch_45, ch_45)

    file_name_ch_90 = end_frame + 'ch_90' + '.png'
    cv2.imwrite(file_name_ch_90, ch_90)

    file_name_ch_135 = end_frame + 'ch_135' + '.png'
    cv2.imwrite(file_name_ch_135, ch_135)
    '''

    
    '''
    pseudo_RGB = np.zeros([1028, 1232, 3])
    pseudo_RGB[:,:, 0] = ch_0
    pseudo_RGB[:,:, 1] = ch_90
    pseudo_RGB[:,:, 2] = ch_135
    '''

    file_name_S_0 = end_frame + 'S_0' + '.png'
    S_0 = S_0 * (255 / np.max(S_0))
    cv2.imwrite(file_name_S_0, S_0)

    file_name_S_1 = end_frame + 'S_1' + '.png'
    S_1 = S_1 + np.abs(np.min(S_1))
    S_1 = S_1 * (255 / np.max(S_1))
    cv2.imwrite(file_name_S_1, S_1)

    file_name_S_2 = end_frame + 'S_2' + '.png'
    S_2 = S_2 + np.abs(np.min(S_2))
    S_2 = S_2 * (255 / np.max(S_2))
    cv2.imwrite(file_name_S_2, S_2)
    
    
    print('np.min(S_0)', np.min(S_0))
    print('np.max(S_0)', np.max(S_0))
    print('np.min(S_1)', np.min(S_1))
    print('np.max(S_1)', np.max(S_1))
    print('np.min(S_2)', np.min(S_2))
    print('np.max(S_2)', np.max(S_2))    
    print(i)



