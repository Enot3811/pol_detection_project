from typing import Optional
import cv2
import numpy as np
from pymba import Frame, Vimba, VimbaException
from datetime import datetime, time
import time
import sys
import zmq
import pickle5 as pickle
from subprocess import Popen, PIPE
import signal
import os
from pathlib import Path


def recv_mess_cam_pol() -> "tuple[int, str, int, int, int]":
    data = sock_cam_pol.recv()
    flag, auto_exposure_val, exposure_val, gain_val, exposure_target_val = pickle.loads(data)
    return flag, auto_exposure_val, exposure_val, gain_val, exposure_target_val
    
def send_mess_cam_pol(data: int):
    """Отправить сообщение на сервер.
    
    Сообщение - временная метка камеры в тиках.
    """
    data = pickle.dumps(data)
    sock_cam_pol.send(data)
    

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
    flag, new_auto_exposure_val, new_exposure_val, new_gain_val, new_exposure_target_val = recv_mess_cam_pol()
    if new_auto_exposure_val != auto_exposure_val:
        auto_exposure_val = new_auto_exposure_val
        cam_pol.ExposureAuto = auto_exposure_val
    if new_exposure_val != exposure_val:
        exposure_val = new_exposure_val
        cam_pol.ExposureTimeAbs = exposure_val
    if new_gain_val != gain_val:
        gain_val = new_gain_val
        cam_pol.Gain = gain_val
    if new_exposure_target_val != exposure_target_val:
        exposure_target_val = new_exposure_target_val
        cam_pol.ExposureAutoTarget = exposure_target_val

    print('flag', flag)
    cam_pol.GevTimestampControlLatch()
    timestamp = cam_pol.GevTimestampValue
    send_mess_cam_pol(timestamp)
    print(cam_pol.GevTimestampValue)
    print('Exposure time:', cam_pol.ExposureTimeAbs)
    print(cam_pol.PtpStatus)


    # get a copy of the frame data
    image = frame.buffer_data_numpy()
    print(frame.pixel_format)
 
    image_to_show = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('image', image_to_show)
    
    cv2.waitKey(1)
    if flag == 1:
        if not DIR.exists():
            DIR.mkdir(parents=True, exist_ok=True)
        frame_name = f'pol_{frame.data.frameID}.npy'
        np.save(DIR / frame_name, image)
        #np.save(name_end_frame_POL, image)
        frame_time = time.time() - start_time
        #file_name = f'image/{cam_pol.GevTimestampValue}_pol_{frame_time}.jpg'
        #cv2.imwrite(file_name, image)

        count += 1
        print('Frame', count, 'saved')
    elif flag == 2:
        cv2.destroyAllWindows()
        print('############### 0 ESC##########')
        trig_off = 1




def close_exit():
    cam_pol.stop_frame_acquisition()
    cam_pol.disarm()
    cam_pol.close()


if __name__ == '__main__':
    # Запуск POL сервера
    context = zmq.Context()
    sock_cam_pol = context.socket(zmq.REP)
    sock_cam_pol.setsockopt (zmq.LINGER, 0)
    nomer_porta = 1234  
    port = 'tcp://127.0.0.1:' + str(nomer_porta)
    try:
        sock_cam_pol.bind(port)
        print('Сервер POL запущен, время:', time.time())
    except:
        process = Popen(["lsof", "-i", ":{0}".format(nomer_porta)], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        for process in str(stdout.decode("utf-8")).split("\n")[1:]:      
            data = [x for x in process.split(" ") if x != '']
            if (len(data) <= 1):
                continue
            os.kill(int(data[1]), signal.SIGKILL)
        time.sleep(0.3)
        sock_cam_pol.bind(port)
        print('Сервер POL запущен после освобождения порта, время:', time.time())

    # Создание директории для сохранения получаемых снимков,
    # подготовка глобальных параметров
    DIR = Path(__file__).parents[2] / 'data' / 'camera' / 'new' / f'images_pol_{datetime.now()}'
    trig_off = 0
    count = 0
    auto_exposure_val = 0
    exposure_val = 100000
    gain_val = 0
    exposure_target_val = 50

    start_time = time.time()
    with Vimba() as vimba:
        cam_pol = vimba.camera('DEV_000F315D16C4')
        cam_pol.open()

        # GigE Control Block
        #cam_pol.BandwidthControlMode = 'StreamBytesPerSecond' # 'StreamBytesPerSecond' , 'SCPD0', 'Both'
        cam_pol.StreamBytesPerSecond = 48000000 # MIN: 1000000 ; MAX: 124000000 ; default 115000000
        # cam_pol.GevSCPSPacketSize = 3000# MIN: 500 ; MAX: 9973 ; default 1500
        cam_pol.PtpMode = 'Off'# 'Off', 'Slave', 'Master', 'Auto'
        cam_pol.PtpMode = 'Master'# 'Off', 'Slave', 'Master', 'Auto'
        print('start ptp status', cam_pol.PtpStatus)

        # cam_rgb.GevTimestampControlReset()
        cam_pol.GevTimestampControlLatch()
        print('start ptp Timestamp', cam_pol.GevTimestampValue)
        
        cam_pol.PtpAcquisitionGateTime = cam_pol.GevTimestampValue + 1000000000 + 80000000
        #cam_pol.PtpAcquisitionGateTime = 10000000000

        
       
        # Trigger Control Block #https://cdn.alliedvision.com/fileadmin/content/documents/products/cam_pols/various/appnote/GigE/Triggering_Concept.pdf


        cam_pol.TriggerSelector = 'FrameStart' # Выберите здесь триггер, затем используйте элементы управления {TriggerMode, TriggerSoftware, TriggerSource, TriggerActivation, TriggerOverlap, TriggerDelayAbs} для настройки и чтения атрибутов триггера. FrameStart - это триггер, запускающий каждое изображение (когда выполняется сборка). AcquisitionStart - это триггер, запускающий процесс сбора данных. Value: 'FrameStart' , 'AcquisitionStart' , 'AcquisitionEnd' , 'AcquisitionRecord'   ; Default 'FrameStart'
        cam_pol.TriggerMode = 'On' # когда триггер FrameStart отключен, изображения запускаются с фиксированной скоростью, указанной в AcquisitionFrameRateAbs. Values: 'On', 'Off' ; Default 'On'
        cam_pol.TriggerOverlap = 'Off' # Разрешенное окно срабатывания триггера относительно предыдущего кадра. Values: 'Off' , 'PreviousFrame'   ; Default 'Off'
        
        cam_pol.TriggerSource = 'FixedRate' # Источник триггера, когда TriggerMode включен. Это может быть только аппаратный триггер, генератор с фиксированной частотой или только программный триггер.;;; Values: 'Freerun', 'Line1', 'Line2', 'FixedRate', 'Software', 'Action0', 'Action1' ;;; Default 'Freerun'
        #cam_pol.TriggerDelayAbs = 0 # Delay from hardware trigger activation to trigger effect, in microseconds.
        cam_pol.TriggerActivation = 'RisingEdge' # Тип активации, для аппаратных триггеров. Это контролирует чувствительность края / уровня и полярности. ;;; 'RisingEdge', 'FallingEdge', 'AnyEdge', 'LevelHigh', 'LevelLow'  ;;; Default 'RisingEdge'


        
        cam_pol.PixelFormat = 'Mono8'


         # Matrix and ROI control
         
         
        try:
            cam_pol.OffsetX = 0 # начальный столбец области считывания матрицы ROI
            cam_pol.OffsetY = 0 # начальная строка области считывания матрицы ROI

        except:
            cam_pol.Height = cam_pol.HeightMax # Max 4856 # my user 4576 
            cam_pol.Width = cam_pol.WidthMax # Max 6476 # my user 4576      
            cam_pol.OffsetX = 0 # начальный столбец области считывания матрицы ROI
            cam_pol.OffsetY = 0 # начальная строка области считывания матрицы ROI
            
        cam_pol.Height = cam_pol.HeightMax # Max 4856 # my user 4576 
        cam_pol.Width = cam_pol.WidthMax # Max 6476 # my user 4576 

        cam_pol.DSPSubregionBottom = cam_pol.Height
        cam_pol.DSPSubregionLeft = 0
        cam_pol.DSPSubregionRight = cam_pol.Width
        cam_pol.DSPSubregionTop = 0

        cam_pol.BinningHorizontal = 1 # 1...4
        cam_pol.BinningVertical = 1 # 1...4




        cam_pol.BlackLevel = 4 # MIN: 0 , MAX: 127.75, derault 4
        cam_pol.Gamma = 1 # Применяет значение гаммы к необработанному сигналу датчика. MIN: 0.25 , MAX: 4, default 1
        cam_pol.Hue = 0 # поворачивает цветовые векторы в плоскости U / V, 1 градус на шаг ; MIN: -40 , MAX: 40, default 0
        cam_pol.Saturation = 0.8 # Насыщенность усиливает цветовые векторы в плоскости U / V ; MIN: 0 , MAX: 2, default 0.8




        #Auto Function Block
 

        cam_pol.DefectMaskEnable = 'On' # Маскировка дефектных пикселей ; 'On' , 'Off'
        # cam_pol.ExposureTimeAbs = 250 # Fersting expouse set value

        cam_pol.ExposureAuto = 'Continuous' # Функция автоматического расчета времени выдержки (экспозиции): 'Continuous' , 'Once', 'Off'
        cam_pol.ExposureAutoTarget = 30
        # cam_pol.ExposureTimeAbs = 2000 # Жесткое задание времени выдержки в микросекундах

        cam_pol.Gain = 0
        #cam_pol.GainAuto = 'Continuous' # Gain: 'Continuous' , 'Once', 'Off'


        # White Balance Block
        #cam_pol.BalanceRatioSelector = 'Red' # Выбор канала для функции автобаланса белого: values 'Red', 'Blue' ; default 'Red'
        #cam_pol.BalanceWhiteAuto = 'Continuous' # BalanceWhite: 'Continuous' , 'Once', 'Off'
        #cam_pol.BalanceRatioAbs = 'Red' # Отрегулируйте усиление красного или синего канала (см. BalanceRatioSelector). Коэффициент усиления зеленого канала всегда равен 1.00 МИНИМУМ: 0,01 ; МАКСИМУМ: 3.99

#Когда изображение захватывается цифровой камерой, реакция сенсора на каждый пиксель зависит от освещенности. То есть каждое значение пикселя, зарегистрированное датчиком, связано с цветовой температурой источника света. Когда белый объект освещается лампой теплой цветовой температуры, на записанном изображении он становится красноватым. Точно так же он будет казаться голубоватым при свете лампы с холодной цветовой температурой. Баланс белого (WB) предназначен для обработки изображения таким образом, чтобы визуально оно выглядело одинаково, независимо от источника света. Почти все датчики, используемые в цифровых камерах, чувствительны только к интенсивности света, а не к его компонентам (разным длинам волн). Вот почему для цветного изображения требуются цветные фильтры, чаще всего фильтр Байера RGB. 

# Примечание автоматического баланса белого. Камеры Allied Vision GigE используют теорию серого мира в качестве основы для режимов автоматического баланса белого (однократный или непрерывный). Если датчик подвергается равномерному белому (или серому) освещению, результирующее изображение должно соответствовать цветности света, создавая равные средние сигналы (Ravg = Gavg = Bavg) на всех трех цветовых каналах; алгоритм автоматического баланса белого увеличивает или уменьшает поправочные коэффициенты CorrR и CorrB для достижения этого результата. #Функцию автоматического баланса белого можно использовать в следующих целях: • Одноразовая операция: баланс белого достигается во время потоковой передачи в течение интервала времени, необходимого для достижения целевого значения цветового баланса; • Непрерывная работа: баланс белого постоянно регулируется во время потоковой передачи изображения с хорошим балансом белого в большинстве условий, и это относительно легко реализовать. 
#В приложениях для сшивания изображений автоматический баланс белого может привести к разочаровывающим результатам, поскольку одинаковый цветовой баланс должен сохраняться для всех сшиваемых изображений, в то время как автоматический баланс белого применяется только к отдельным изображениям.

# Примечание ручной баланс белого Пользователь имеет прямой доступ к усилению баланса белого (GCred и GCblue) для красного и синего каналов, в то время как усиление зеленого канала остается равным 1.0: GCred = 1 / CorrR; GCblue = 1 / CorrB (2) Следует проявлять осторожность при усилении баланса белого ниже 1.0 - соответствующие каналы (красный или синий) могут насыщаться, преждевременно вызывая ошибки цветности в светлых участках. Коэффициенты усиления баланса белого по умолчанию не калибруются, и их диапазон не нормализуется. Следовательно, изображения, снятые несколькими камерами одного типа, могут незначительно отличаться из-за небольших различий от сенсора к сенсору и от камеры к камере.

        #cam_pol.MulticastEnable = 'true' # true
        #MulticastIPAddress = '169.254.1.1'





#Источник: https://cdn.alliedvision.com/fileadmin/content/documents/products/cam_pols/various/appnote/GigE/White_Balance_for_GigE_cam_pols.pdf

        cam_pol.AcquisitionFrameRateAbs = 1 # FPS ; Max 3.358 for full resolution ;  AFFECTED FEATURE(S):ExposureTimeAbs, AcquisitionFrameRateLimit 


        print ('cam_pol ids',vimba.camera_ids())
        print ('Interface ids',vimba.interface_ids())
        print ('Vimba version', vimba.version())
        print ('cam_pol info', cam_pol.info)
        print ('Device Temperature', cam_pol.DeviceTemperature)
        print ('Exposure Auto selector - ', cam_pol.ExposureAuto)
        print ('Exposure Time, microseconds =', cam_pol.ExposureTimeAbs)
        print ('GainAuto selector - ', cam_pol.GainAuto)
        print ('Gain value =', cam_pol.Gain)

        print ('OffsetX (начальный столбец области считывания матрицы ROI) =', cam_pol.OffsetX)
        print ('OffsetY (начальная строка области считывания матрицы ROI)=', cam_pol.OffsetY)

        print ('HeightMax =', cam_pol.HeightMax)
        print ('WidthMax =', cam_pol.WidthMax)

        print ('Высота =', cam_pol.Height)
        print ('Ширина =', cam_pol.Width)
        print ('GigE Bandwidth Control Mode selector -', cam_pol.BandwidthControlMode)
        print ('GigE Stream Bytes Per Second =', cam_pol.StreamBytesPerSecond)
        print ('GigE GevSCPS Packet Size=', cam_pol.GevSCPSPacketSize)
        print ('FPS =', cam_pol.AcquisitionFrameRateAbs)
        print ('Maximum teoretical FPS =', cam_pol.AcquisitionFrameRateLimit)
        print ('DSPSubregionBottom =', cam_pol.DSPSubregionBottom)
        print ('DSPSubregionRight =', cam_pol.DSPSubregionRight)







        # arm the cam_pol and provide a function to be called upon frame ready
        cam_pol.arm('Continuous', save_frame, 2) #Continuous

##################
        cam_pol.AcquisitionMode = 'Continuous' # Режим сбора данных определяет поведение камеры при запуске сбора данных. «Continuous» режим: камера будет получать изображения, пока не сработает остановка получения. Режим «SingleFrame»: камера получает одно изображение. Режим «MultiFrame»: камера получит количество изображений, указанное в AcquisitionFrameCount. Режим «Recoder»: камера будет работать непрерывно, и при запуске события регистратора отправлять на хост изображения до и после запуска.
        #cam_pol.AcquisitionFrameCount = 3 # Это количество изображений, которые необходимо получить в режимах получения MultiFrame и Recorder. МИНИМУМ: 1 МАКСИМУМ: 65535
        #cam_pol.RecorderPreEventCount = 0 # Это количество изображений перед событием, которые необходимо получить в режиме сбора данных регистратора. Это должно быть меньше или равно AcquisitionFrameCount. МИНИМУМ: 0 МАКСИМУМ: 65535


#######################################

        cam_pol.start_frame_acquisition()
        while True:
            if trig_off == 1:
                close_exit()
            else:
                pass
