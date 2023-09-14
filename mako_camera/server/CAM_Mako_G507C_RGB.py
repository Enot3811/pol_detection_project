from typing import Optional
import cv2
import numpy as np
from pymba import Frame, Vimba
from datetime import datetime, time
import time
import zmq
import pickle5 as pickle
from subprocess import Popen, PIPE
import signal
import os
from pathlib import Path


def recv_mess_cam_rgb() -> "tuple[int, str, int, int, int]":
    data = sock_cam_rgb.recv()
    flag, auto_exposure_val, exposure_val, gain_val, exposure_target_val = pickle.loads(data)
    return flag, auto_exposure_val, exposure_val, gain_val, exposure_target_val


def send_mess_cam_rgb(data):
    data = pickle.dumps(data)
    sock_cam_rgb.send(data) 


def save_frame(frame: Frame, delay: Optional[int] = 1) -> None: # default Optional[int] = 1
    global trig_off
    global count
    global DIR
    global auto_exposure_val
    global exposure_val
    global gain_val
    global exposure_target_val

    Number_Frame = ('frame {}'.format(frame.data.frameID) + str(datetime.now()))
    Number_Frame = ("".join(Number_Frame))
    UID_Frame = str(frame.data.frameID)
    
    # Применение новых параметров камеры
    flag, new_auto_exposure_val, new_exposure_val, new_gain_val, new_exposure_target_val = recv_mess_cam_rgb()
    if new_auto_exposure_val != auto_exposure_val:
        auto_exposure_val = new_auto_exposure_val
        cam_rgb.ExposureAuto = auto_exposure_val
    if new_exposure_val != exposure_val:
        exposure_val = new_exposure_val
        cam_rgb.ExposureTimeAbs = exposure_val
    if new_gain_val != gain_val:
        gain_val = new_gain_val
        cam_rgb.Gain = gain_val
    if new_exposure_target_val != exposure_target_val:
        exposure_target_val = new_exposure_target_val
        cam_rgb.ExposureAutoTarget = exposure_target_val
    print('flag', flag)    
    cam_rgb.GevTimestampControlLatch()
    send_mess_cam_rgb(cam_rgb.GevTimestampValue)    
    print(cam_rgb.GevTimestampValue)
    print('Exposure time:', cam_rgb.ExposureTimeAbs)
    print(cam_rgb.PtpStatus)

    # get a copy of the frame data
    image = frame.buffer_data_numpy()
 
    image_to_show = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
    image_to_show = cv2.resize(image_to_show, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('image', image_to_show)

    cv2.waitKey(1)
    if flag == 1:
       frame_name = f'rgb_{frame.data.frameID}.npy'
       np.save(DIR / frame_name, image)
       frame_time = time.time() - start_time
       #file_name = f'image/{cam_rgb.GevTimestampValue}_rgb_{frame_time}.jpg'
       #cv2.imwrite(file_name, image)

       count += 1
       print('Frame', count, 'saved')
    elif flag == 2:
       cv2.destroyAllWindows()
       print('############### 0 ESC##########')
       trig_off = 1


def close_exit():
    cam_rgb.stop_frame_acquisition()
    cam_rgb.disarm()
    cam_rgb.close()


if __name__ == '__main__':
    # Запуск RGB сервера
    context = zmq.Context()
    sock_cam_rgb = context.socket(zmq.REP)
    sock_cam_rgb.setsockopt (zmq.LINGER, 0)
    nomer_porta = 12345  
    port = 'tcp://127.0.0.1:' + str(nomer_porta)
    try:
        sock_cam_rgb.bind(port)
        print('Сервер RGB запущен, время:', time.time())
    except:
        process = Popen(["lsof", "-i", ":{0}".format(nomer_porta)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        for process in str(stdout.decode("utf-8")).split("\n")[1:]:      
            data = [x for x in process.split(" ") if x != '']
            if (len(data) <= 1):
                continue
            os.kill(int(data[1]), signal.SIGKILL)
        time.sleep(0.3)
        sock_cam_rgb.bind(port)
        print('Сервер RGB запущен после освобождения порта, время:', time.time())

    # Создание директории для сохранения получаемых снимков,
    # подготовка глобальных параметров
    DIR = Path(f'images_rgb_{datetime.now()}')
    DIR.mkdir(parents=True, exist_ok=True)
    trig_off = 0
    count = 0
    auto_exposure_val = 0
    exposure_val = 500
    gain_val = 0
    exposure_target_val = 50

    start_time = time.time()
    with Vimba() as vimba:
        cam_rgb = vimba.camera('DEV_000F315D630A')
        cam_rgb.open()

        # GigE Control Block
        #cam_rgb.BandwidthControlMode = 'StreamBytesPerSecond' # 'StreamBytesPerSecond' , 'SCPD0', 'Both'
        cam_rgb.StreamBytesPerSecond = 48000000 
        cam_rgb.GevSCPSPacketSize = 3000# MIN: 500 ; MAX: 9973 ; default 1500
        cam_rgb.PtpMode = 'Off'# 'Off', 'Slave', 'Master', 'Auto'
        cam_rgb.PtpMode = 'Slave'# 'Off', 'Slave', 'Master', 'Auto'
        print('start ptp status', cam_rgb.PtpStatus)

        # cam_rgb.GevTimestampControlReset()
        cam_rgb.GevTimestampControlLatch()
        print('start ptp Timestamp', cam_rgb.GevTimestampValue)

        cam_rgb.PtpAcquisitionGateTime = cam_rgb.GevTimestampValue + 1000000000


        # Trigger Control Block #https://cdn.alliedvision.com/fileadmin/content/documents/products/cam_rgbs/various/appnote/GigE/Triggering_Concept.pdf


        cam_rgb.TriggerSelector = 'FrameStart' # Выберите здесь триггер, затем используйте элементы управления {TriggerMode, TriggerSoftware, TriggerSource, TriggerActivation, TriggerOverlap, TriggerDelayAbs} для настройки и чтения атрибутов триггера. FrameStart - это триггер, запускающий каждое изображение (когда выполняется сборка). AcquisitionStart - это триггер, запускающий процесс сбора данных. Value: 'FrameStart' , 'AcquisitionStart' , 'AcquisitionEnd' , 'AcquisitionRecord'   ; Default 'FrameStart'
        cam_rgb.TriggerMode = 'On' # когда триггер FrameStart отключен, изображения запускаются с фиксированной скоростью, указанной в AcquisitionFrameRateAbs. Values: 'On', 'Off' ; Default 'On'
        cam_rgb.TriggerOverlap = 'Off' # Разрешенное окно срабатывания триггера относительно предыдущего кадра. Values: 'Off' , 'PreviousFrame'   ; Default 'Off'
        
        cam_rgb.TriggerSource = 'FixedRate' # Источник триггера, когда TriggerMode включен. Это может быть только аппаратный триггер, генератор с фиксированной частотой или только программный триггер.;;; Values: 'Freerun', 'Line1', 'Line2', 'FixedRate', 'Software', 'Action0', 'Action1' ;;; Default 'Freerun'
        #cam_rgb.TriggerDelayAbs = 0 # Delay from hardware trigger activation to trigger effect, in microseconds.
        cam_rgb.TriggerActivation = 'RisingEdge' # Тип активации, для аппаратных триггеров. Это контролирует чувствительность края / уровня и полярности. ;;; 'RisingEdge', 'FallingEdge', 'AnyEdge', 'LevelHigh', 'LevelLow'  ;;; Default 'RisingEdge'


        
        cam_rgb.PixelFormat = 'BayerRG8'


         # Matrix and ROI control
         
         
        try:
            cam_rgb.OffsetX = 0 # начальный столбец области считывания матрицы ROI
            cam_rgb.OffsetY = 0 # начальная строка области считывания матрицы ROI

        except:
            cam_rgb.Height = cam_rgb.HeightMax # Max 4856 # my user 4576 
            cam_rgb.Width = cam_rgb.WidthMax # Max 6476 # my user 4576      
            cam_rgb.OffsetX = 0 # начальный столбец области считывания матрицы ROI
            cam_rgb.OffsetY = 0 # начальная строка области считывания матрицы ROI

        cam_rgb.Height = cam_rgb.HeightMax # Max 4856 # my user 4576 
        cam_rgb.Width = cam_rgb.WidthMax # Max 6476 # my user 4576 

        cam_rgb.DSPSubregionBottom = cam_rgb.Height
        cam_rgb.DSPSubregionLeft = 0
        cam_rgb.DSPSubregionRight = cam_rgb.Width
        cam_rgb.DSPSubregionTop = 0

        cam_rgb.BinningHorizontal = 1 # 1...4
        cam_rgb.BinningVertical = 1 # 1...4




        cam_rgb.BlackLevel = 4 # MIN: 0 , MAX: 127.75, derault 4
        cam_rgb.Gamma = 1 # Применяет значение гаммы к необработанному сигналу датчика. MIN: 0.25 , MAX: 4, default 1
        cam_rgb.Hue = 0 # поворачивает цветовые векторы в плоскости U / V, 1 градус на шаг ; MIN: -40 , MAX: 40, default 0
        cam_rgb.Saturation = 0.8 # Насыщенность усиливает цветовые векторы в плоскости U / V ; MIN: 0 , MAX: 2, default 0.8




        #Auto Function Block
 

        cam_rgb.DefectMaskEnable = 'On' # Маскировка дефектных пикселей ; 'On' , 'Off'
        #cam_rgb.ExposureTimeAbs = 250 # Fersting expouse set value

        cam_rgb.ExposureAuto = 'Off' # Функция автоматического расчета времени выдержки (экспозиции): 'Continuous' , 'Once', 'Off'
        cam_rgb.ExposureTimeAbs = 50 # Жесткое задание времени выдержки в микросекундах
        cam_rgb.ExposureAutoTarget = 30
        #cam_rgb.ExposureTimeAbs = cam_rgb.ExposureTimeAbs - 200000 # correcting expouse

        cam_rgb.Gain = 0
        #cam_rgb.GainAuto = 'Continuous' # Gain: 'Continuous' , 'Once', 'Off'


        # White Balance Block
        #cam_rgb.BalanceRatioSelector = 'Red' # Выбор канала для функции автобаланса белого: values 'Red', 'Blue' ; default 'Red'
        #cam_rgb.BalanceWhiteAuto = 'Continuous' # BalanceWhite: 'Continuous' , 'Once', 'Off'
        #cam_rgb.BalanceRatioAbs = 'Red' # Отрегулируйте усиление красного или синего канала (см. BalanceRatioSelector). Коэффициент усиления зеленого канала всегда равен 1.00 МИНИМУМ: 0,01 ; МАКСИМУМ: 3.99

#Когда изображение захватывается цифровой камерой, реакция сенсора на каждый пиксель зависит от освещенности. То есть каждое значение пикселя, зарегистрированное датчиком, связано с цветовой температурой источника света. Когда белый объект освещается лампой теплой цветовой температуры, на записанном изображении он становится красноватым. Точно так же он будет казаться голубоватым при свете лампы с холодной цветовой температурой. Баланс белого (WB) предназначен для обработки изображения таким образом, чтобы визуально оно выглядело одинаково, независимо от источника света. Почти все датчики, используемые в цифровых камерах, чувствительны только к интенсивности света, а не к его компонентам (разным длинам волн). Вот почему для цветного изображения требуются цветные фильтры, чаще всего фильтр Байера RGB. 

# Примечание автоматического баланса белого. Камеры Allied Vision GigE используют теорию серого мира в качестве основы для режимов автоматического баланса белого (однократный или непрерывный). Если датчик подвергается равномерному белому (или серому) освещению, результирующее изображение должно соответствовать цветности света, создавая равные средние сигналы (Ravg = Gavg = Bavg) на всех трех цветовых каналах; алгоритм автоматического баланса белого увеличивает или уменьшает поправочные коэффициенты CorrR и CorrB для достижения этого результата. #Функцию автоматического баланса белого можно использовать в следующих целях: • Одноразовая операция: баланс белого достигается во время потоковой передачи в течение интервала времени, необходимого для достижения целевого значения цветового баланса; • Непрерывная работа: баланс белого постоянно регулируется во время потоковой передачи изображения с хорошим балансом белого в большинстве условий, и это относительно легко реализовать. 
#В приложениях для сшивания изображений автоматический баланс белого может привести к разочаровывающим результатам, поскольку одинаковый цветовой баланс должен сохраняться для всех сшиваемых изображений, в то время как автоматический баланс белого применяется только к отдельным изображениям.

# Примечание ручной баланс белого Пользователь имеет прямой доступ к усилению баланса белого (GCred и GCblue) для красного и синего каналов, в то время как усиление зеленого канала остается равным 1.0: GCred = 1 / CorrR; GCblue = 1 / CorrB (2) Следует проявлять осторожность при усилении баланса белого ниже 1.0 - соответствующие каналы (красный или синий) могут насыщаться, преждевременно вызывая ошибки цветности в светлых участках. Коэффициенты усиления баланса белого по умолчанию не калибруются, и их диапазон не нормализуется. Следовательно, изображения, снятые несколькими камерами одного типа, могут незначительно отличаться из-за небольших различий от сенсора к сенсору и от камеры к камере.

        #cam_rgb.MulticastEnable = 'true' # true
        #MulticastIPAddress = '169.254.1.1'





#Источник: https://cdn.alliedvision.com/fileadmin/content/documents/products/cam_rgbs/various/appnote/GigE/White_Balance_for_GigE_cam_rgbs.pdf

        cam_rgb.AcquisitionFrameRateAbs = 1 # FPS ; Max 3.358 for full resolution ;  AFFECTED FEATURE(S):ExposureTimeAbs, AcquisitionFrameRateLimit

        print ('cam_rgb ids',vimba.camera_ids())
        print ('Interface ids',vimba.interface_ids())
        print ('Vimba version', vimba.version())
        print ('cam_rgb info', cam_rgb.info)
        print ('Device Temperature', cam_rgb.DeviceTemperature)
        print ('Exposure Auto selector - ', cam_rgb.ExposureAuto)
        print ('Exposure Time, microseconds =', cam_rgb.ExposureTimeAbs)
        print ('GainAuto selector - ', cam_rgb.GainAuto)
        print ('Gain value =', cam_rgb.Gain)

        print ('OffsetX (начальный столбец области считывания матрицы ROI) =', cam_rgb.OffsetX)
        print ('OffsetY (начальная строка области считывания матрицы ROI)=', cam_rgb.OffsetY)

        print ('HeightMax =', cam_rgb.HeightMax)
        print ('WidthMax =', cam_rgb.WidthMax)

        print ('Высота =', cam_rgb.Height)
        print ('Ширина =', cam_rgb.Width)
        print ('GigE Bandwidth Control Mode selector -', cam_rgb.BandwidthControlMode)
        print ('GigE Stream Bytes Per Second =', cam_rgb.StreamBytesPerSecond)
        print ('GigE GevSCPS Packet Size=', cam_rgb.GevSCPSPacketSize)
        print ('FPS =', cam_rgb.AcquisitionFrameRateAbs)
        print ('Maximum teoretical FPS =', cam_rgb.AcquisitionFrameRateLimit)
        print ('DSPSubregionBottom =', cam_rgb.DSPSubregionBottom)
        print ('DSPSubregionRight =', cam_rgb.DSPSubregionRight)







        # arm the cam_rgb and provide a function to be called upon frame ready
        cam_rgb.arm('Continuous', save_frame, 2) #Continuous
##################
        cam_rgb.AcquisitionMode = 'Continuous' # Режим сбора данных определяет поведение камеры при запуске сбора данных. «Continuous» режим: камера будет получать изображения, пока не сработает остановка получения. Режим «SingleFrame»: камера получает одно изображение. Режим «MultiFrame»: камера получит количество изображений, указанное в AcquisitionFrameCount. Режим «Recoder»: камера будет работать непрерывно, и при запуске события регистратора отправлять на хост изображения до и после запуска.
        #cam_rgb.AcquisitionFrameCount = 3 # Это количество изображений, которые необходимо получить в режимах получения MultiFrame и Recorder. МИНИМУМ: 1 МАКСИМУМ: 65535
        #cam_rgb.RecorderPreEventCount = 0 # Это количество изображений перед событием, которые необходимо получить в режиме сбора данных регистратора. Это должно быть меньше или равно AcquisitionFrameCount. МИНИМУМ: 0 МАКСИМУМ: 65535


#######################################

        cam_rgb.start_frame_acquisition()
        while True:
            if trig_off == 1:
                close_exit()
            else:
                pass
        
        

  

        





