import os
import gc
import math
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import scipy.stats as stats 
import time

import xlsxwriter

import read_data
from madgwick import AHRS
import processing

import writefile

from dataclasses import dataclass
from dataclasses import field


import matplotlib.pyplot as plt


from MadgwickAHRS import MadgwickAHRS 
from quaternion import Quaternion
import Datastructure

import pandas as pd
import plotData

# Sampling Freq
dt_ms = 2

# View3D.showIMUOrientation2()

@dataclass
class euler():   
    roll : float
    pitch : float
    yaw : float


@dataclass
class quat():
    w : float = 1
    x : float = 0
    y : float = 0
    z : float = 0

    def get_euler(self)->euler:

        t0 = +2.0 * (self.w * self.x + self.y * self.z)
        t1 = +1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (self.w * self.y - self.z * self.x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (self.w * self.z + self.x * self.y)
        t4 = +1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw_z = math.atan2(t3, t4)  
 
        _euler = euler(roll_x * (180 / math.pi) ,pitch_y * (180 / math.pi),yaw_z * (180 / math.pi))
        return _euler
    


@dataclass
class IMU_Sensor():
    name: str
    time:list[int] = field(default_factory=list)
    AX: list[float] = field(default_factory=list)
    AY: list[float] = field(default_factory=list)
    AZ: list[float] = field(default_factory=list)
    GX: list[float] = field(default_factory=list)
    GY: list[float] = field(default_factory=list)
    GZ: list[float] = field(default_factory=list)
    MX: list[float] = field(default_factory=list)
    MY: list[float] = field(default_factory=list)
    MZ: list[float] = field(default_factory=list)
    AX_down: list[float] = field(default_factory=list)
    AY_down: list[float] = field(default_factory=list)
    AZ_down: list[float] = field(default_factory=list)
    GX_down: list[float] = field(default_factory=list)
    GY_down: list[float] = field(default_factory=list)
    GZ_down: list[float] = field(default_factory=list)
    MX_down: list[float] = field(default_factory=list)
    MY_down: list[float] = field(default_factory=list)
    MZ_down: list[float] = field(default_factory=list)

    Free_GAX: list[float] = field(default_factory=list)
    Free_GAY: list[float] = field(default_factory=list)
    Free_GAZ: list[float] = field(default_factory=list)

    time_down:list[int] = field(default_factory=list)
    Q: list[quat] = field(init=False, default_factory=list)
    Q_down: list[quat] = field(init=False, default_factory=list)
    Euler: list[euler] = field(init=False, default_factory=list)



@dataclass
class myTask():
    name : str
    
    Data : list[IMU_Sensor] = field(default_factory=list)

    


@dataclass
class player():
    name: str
    age: int
    mass: float
    height: int
    gender: str
    ID : str
    pass_n : str
    data_path: list[str] = field(default_factory=list)
    Time_data: list[float] = field(init=False , default_factory=list)
    Tasks: list[myTask] = field(init=False, default_factory=list)
 

@dataclass
class Features():

    subject : str =""
    pre_post : str =""
    TaskName : str ="" 
    SensorName : str =""
    SourceNAme : str =""
    SourceFrame : str=""
    Component : str=""
    Fearture : str=""
    FeartureType : str=""
    Period : str=""
    Value : float=0

    



@dataclass
class JUMP():
    type : str
    mass : float = 1
    sub_height : float = 0

    
    JH : float = 0
    JH_FT : float = 0
    FT : float = 0

    TimeLanding : float =0    
    TimeMin2MaxTakeOFF : float =0
    Min2MaxLanding : float = 0
    Time2PeakLanding : float =0 
    SkewnessLandingAcc :float =0
    KurtosisLandingAcc : float =0
    SkewnessLandingGyro :float =0
    KurtosisLandingGyro : float =0

    MaxACCVerticalTakeoff_G :float =0
    MaxACCVerticalLanding_G :float =0
    MaxACCHorizontalTakeoff_G :float =0
    MaxACCHorizontalLanding_G :float =0

    MaxACCVerticalTakeoff_L :float =0
    MaxACCVerticalLanding_L :float =0
    MaxACCHorizontalTakeoff_L :float =0
    MaxACCHorizontalLanding_L :float =0

    MaxGYROSagitalTakeoff_G :float =0
    MaxGYROSagitalLanding_G :float =0
    MaxGYROFrontalTakeoff_G :float =0
    MaxGYROFrontalLanding_G :float =0

    MaxGYROSagitalTakeoff_L :float =0
    MaxGYROSagitalLanding_L :float =0
    MaxGYROFrontalTakeoff_L :float =0
    MaxGYROFrontalLanding_L :float =0

    TibVarPeakAccV : float = 0
    TibVarMin : float = 0
    TibVarMax : float = 0
    TibVarRange : float = 0

    TibFlexMax : float = 0
    TibFlexMin : float = 0
    TibFlexRange : float = 0
    M1_ACCV_Max : float = 0
    M2_ACCV_Max : float = 0
    KAM_ACCV_Max : float = 0


    LP90 : float = 0


    PeakVerticalForceTakeoff : float = 0
    PeakVerticalForceLanding : float = 0
    PeakMediolateralForceTakeoff : float = 0
    PeakMediolateralForceLanding : float = 0

    take_off_vel : int = 0
    landing_index : int =0

    time:list[int] = field(default_factory=list)
    Velocity: list[float] = field(default_factory=list)


    MainData = Datastructure.signalSource()



    
    Main_Acc_Z: list[float] = field(default_factory=list)
    Main_Acc_X: list[float] = field(default_factory=list)


    ACCX: list[float] = field(default_factory=list)
    ACCY: list[float] = field(default_factory=list)
    ACCZ: list[float] = field(default_factory=list)


    GYROX: list[float] = field(default_factory=list)
    GYROY: list[float] = field(default_factory=list)
    GYROZ: list[float] = field(default_factory=list)

    MAGX: list[float] = field(default_factory=list)
    MAGY: list[float] = field(default_factory=list)
    MAGZ: list[float] = field(default_factory=list)

    ACCX_B: list[float] = field(default_factory=list)
    ACCY_B: list[float] = field(default_factory=list)
    ACCZ_B: list[float] = field(default_factory=list)

    GYROX_B: list[float] = field(default_factory=list)
    GYROY_B: list[float] = field(default_factory=list)
    GYROZ_B: list[float] = field(default_factory=list)


    ACCX_G: list[float] = field(default_factory=list)
    ACCY_G: list[float] = field(default_factory=list)
    ACCZ_G: list[float] = field(default_factory=list)

    ACCX_G2: list[float] = field(default_factory=list)
    ACCY_G2: list[float] = field(default_factory=list)
    ACCZ_G2: list[float] = field(default_factory=list)


    GYROX_G: list[float] = field(default_factory=list)
    GYROY_G: list[float] = field(default_factory=list)
    GYROZ_G: list[float] = field(default_factory=list)

    GYROX_G2: list[float] = field(default_factory=list)
    GYROY_G2: list[float] = field(default_factory=list)
    GYROZ_G2: list[float] = field(default_factory=list)

    Q: list[quat] = field(init=False, default_factory=list)


    Force_X: list[float] = field(default_factory=list)
    Force_Y: list[float] = field(default_factory=list)
    Force_Z: list[float] = field(default_factory=list)

    Vertical_ACC: list[float] = field(default_factory=list)
    Horizontal_ACC: list[float] = field(default_factory=list)

    roll: list[float] = field(default_factory=list)
    pitch: list[float] = field(default_factory=list)
    yaw: list[float] = field(default_factory=list)


    roll2: list[float] = field(default_factory=list)
    pitch2: list[float] = field(default_factory=list)
    yaw2: list[float] = field(default_factory=list)


    Moment_Sagital: list[float] = field(default_factory=list)

    Moment_KAM_1: list[float] = field(default_factory=list)
    Moment_KAM_2: list[float] = field(default_factory=list)


    Moment2_KAM : list[float] = field(default= list)
    Moment2_Flex : list[float] = field(default= list)


    StaticRoll : list[float] = field(default= list)
    StaticPitch : list[float] = field(default= list)
    StaticYaw : list[float] = field(default= list)



    A : float = 0
    C : float = 0
    D : float = 0
    F : float = 0
    I : float = 0
    G : float = 0





    ALLFeatures: list[Features] = field(default_factory=list)
    

    def addEmptyFeatures(self , sub_code ,pre_post, TaskName, SName): 

                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='JH', FeartureType="Param",Period='JMP',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='JHFT', FeartureType= 'Param',Period='JMP',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='FT', FeartureType= 'Param',Period='JMP',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='Time2Stability',FeartureType="Param", Period='LAN',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='TimeMin2MaxTakeoff',FeartureType="Param" , Period='TAK',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='Time2PeakLanding', FeartureType="Param" ,Period='LAN',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='QUA',SourceFrame='G',Component='X',Fearture='FlexRange',FeartureType="Range",Period='LAN',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='QUA',SourceFrame='G',Component='Z',Fearture='AbAdRange',FeartureType="Range",Period='LAN',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='M1_ACCV_Max',FeartureType="Peak",Period='LAN',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='M2_ACCV_Max',FeartureType="Peak",Period='LAN',Value=np.nan))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='KAM_ACCV_Max',FeartureType="Peak",Period='LAN',Value=np.nan))

    def setMainData(self,StartEvent,take_off_vel_index,landing_index,unWeighting_phase_index,Barking_phase_index):
        

        try:

            _signals =  vars(self.MainData)       

            # ACC - GYRO - QUAT
            SourceName = list(_signals.keys())

            Sourcesignal = Datastructure.signalSource()
            for i in range(len(SourceName)):
                # SF - GF
                _frame =  getattr(self.MainData , SourceName[i])
                _frameVar = vars(_frame)
                FrameName = list(_frameVar.keys())

                Framesignal = Datastructure.CoordinationFrame()  
                for i2 in range(len(FrameName)): 
                    # X - Y - Z
                    _axis =  getattr(_frame , FrameName[i2])
                    _axisVar = vars(_axis)
                    AxisName = list(_axisVar.keys()) 
                    axissignal = Datastructure.Axis()               
                    for i3 in range(len(AxisName)):
                        #  Preiod => TK(takeoff) , LA(landing), JM(Jump),CON, ECC
                        _period =  getattr(_axis , AxisName[i3])
                        _periodvar = vars(_period)
                        PeriodName = list(_periodvar.keys())   

                        if (SourceName[i]=='Acc'):
                            if (FrameName[i2]=='SF'):
                                if (AxisName[i3]=='X'):
                                    signal = self.ACCX_B
                                elif (AxisName[i3]=='Y'):
                                    signal = self.ACCY_B
                                elif (AxisName[i3]=='Z'):
                                    signal = self.ACCZ_B
                            elif (FrameName[i2]=='GF'):
                                if (AxisName[i3]=='X'):
                                    signal = self.ACCX_G
                                elif (AxisName[i3]=='Y'):
                                    signal = self.ACCY_G
                                elif (AxisName[i3]=='Z'):
                                    signal = self.ACCZ_G

                        if (SourceName[i]=='Gyro'):
                            if (FrameName[i2]=='SF'):
                                if (AxisName[i3]=='X'):
                                    signal = self.GYROX_B
                                elif (AxisName[i3]=='Y'):
                                    signal = self.GYROY_B
                                elif (AxisName[i3]=='Z'):
                                    signal = self.GYROZ_B
                            elif (FrameName[i2]=='GF'):
                                if (AxisName[i3]=='X'):
                                    signal = self.GYROX_G
                                elif (AxisName[i3]=='Y'):
                                    signal = self.GYROY_G
                                elif (AxisName[i3]=='Z'):
                                    signal = self.GYROZ_G


                        if (SourceName[i]=='Quat'):
                            if (FrameName[i2]=='SF'):
                                if (AxisName[i3]=='X'):
                                    signal = []
                                elif (AxisName[i3]=='Y'):
                                    signal = []
                                elif (AxisName[i3]=='Z'):
                                    signal = []
                            elif (FrameName[i2]=='GF'):
                                if (AxisName[i3]=='X'):
                                    signal = self.roll
                                elif (AxisName[i3]=='Y'):
                                    signal = self.pitch
                                elif (AxisName[i3]=='Z'):
                                    signal = self.yaw                     

                        periodsignal = Datastructure.Periods()                  
                        TK = signal[StartEvent:take_off_vel_index]
                        setattr(periodsignal,PeriodName[0],TK)
                        LA = signal[landing_index:]
                        setattr(periodsignal,PeriodName[1],LA)
                        JM = signal
                        setattr(periodsignal,PeriodName[2],JM)
                        CON = signal[Barking_phase_index:take_off_vel_index]
                        setattr(periodsignal,PeriodName[3],CON)
                        ECC = signal[unWeighting_phase_index:Barking_phase_index]                       
                        setattr(periodsignal,PeriodName[3],ECC)



                        


                        setattr(axissignal,AxisName[i3],periodsignal)
                            
                    setattr(Framesignal,FrameName[i2],axissignal)

                setattr(Sourcesignal,SourceName[i],Framesignal)

            self.MainData = Sourcesignal


        except Exception as e:
            print(e)
          




                                              


    def featur_extraction(self, sub_code ,pre_post, TaskName,SName):

        

            _signals =  vars(self.MainData)       

            # ACC - GYRO - QUAT
            SourceName = list(_signals.keys())
            for i in range(len(SourceName)):
                # SF - GF
                _frame =  getattr(self.MainData , SourceName[i])
                _frameVar = vars(_frame)
                FrameName = list(_frameVar.keys())
                for i2 in range(len(FrameName)): 
                    # X - Y - Z
                    _axis =  getattr(_frame , FrameName[i2])
                    _axisVar = vars(_axis)
                    AxisName = list(_axisVar.keys())                
                    for i3 in range(len(AxisName)):
                        #  Preiod => TK(takeoff) , LA(landing), JM(Jump),CON, ECC
                        _period =  getattr(_axis , AxisName[i3])
                        _periodVar = vars(_period)
                        PeriodName = list(_periodVar.keys())      
                        for i4 in range(len(PeriodName)):
                            # Features max , min , P10 , P90 , mean , med , skw , kur 
                            
                            signal = getattr(_period,PeriodName[i4])
                         
                            if(len(signal)>0) :

                                signal = signal[~np.isnan(signal)]
                                maxVal = np.max(signal)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Max",Period=PeriodName[i4],Value=maxVal))

                                minVal = np.min(signal)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Min",Period=PeriodName[i4],Value=minVal))

                                P10 = np.percentile(signal,10)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="P10",Period=PeriodName[i4],Value=P10))

                                P90 = np.percentile(signal,90)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="P90",Period=PeriodName[i4],Value=P90))


                                mean = np.average(signal)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Mean",Period=PeriodName[i4],Value=mean))

                                
                                median = np.median(signal)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Median",Period=PeriodName[i4],Value=median))

                                skw = stats.skew(signal)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="SKW",Period=PeriodName[i4],Value=skw))

                                kur = stats.kurtosis(signal)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="KUR",Period=PeriodName[i4],Value=kur))

                                
                                Range = abs(maxVal - minVal)
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Range",Period=PeriodName[i4],Value=Range))
                                

                            else:

                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Max",Period=PeriodName[i4],Value=0))
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Min",Period=PeriodName[i4],Value=0))
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="P10",Period=PeriodName[i4],Value=0))
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="P90",Period=PeriodName[i4],Value=0))
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Mean",Period=PeriodName[i4],Value=0))
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="Median",Period=PeriodName[i4],Value=0))
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="SKW",Period=PeriodName[i4],Value=0))
                                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post,TaskName= TaskName,SensorName=SName,SourceNAme=SourceName[i],SourceFrame=FrameName[i2],Component=AxisName[i3],Fearture='',FeartureType="KUR",Period=PeriodName[i4],Value=0))




               






         
    def start_Event(self):

        avg = np.average(self.ACCZ_G[0:100])
        std = np.std(self.ACCZ_G[0:100])

        inx = [inx for inx,d in enumerate(self.ACCZ_G) if abs(d) >= avg+ 2*std ]

        return inx[0]



    def end_Event(self):

        avg = np.average(self.ACCZ_G[0:100])
        std = np.std(self.ACCZ_G[0:100])

        inx = [inx for inx,d in enumerate(self.ACCZ_G[self.landing_index:]) if abs(d) <= avg+ 5*std ]
        if (len(inx)>0):
            return inx[0]
        else :
            return len(self.ACCZ_G)

        

   

    def run(self,SName, TaskName,sub_code , pre_post):
                    
        revers = False

        if "RSJ" in TaskName:
            revers = True
        # get all events
        try:
            [take_off_vel,take_off_vel_index,landing_index,unWeighting_phase_index,Barking_phase_index] = processing.getAllEvents(self.Velocity)

            if SName == "Pelvis":              
                StartEvent = self.start_Event()
                self.A = StartEvent
                self.C = unWeighting_phase_index
                self.D = Barking_phase_index
                self.F = take_off_vel_index
                self.I = landing_index
                self.G = self.end_Event()

                EVENTs[0] = StartEvent
                EVENTs[1] = unWeighting_phase_index
                EVENTs[2] = Barking_phase_index
                EVENTs[3] = take_off_vel_index
                EVENTs[4] = landing_index
                EVENTs[5] = self.end_Event()


            else :

                StartEvent = EVENTs[0]
                self.A = StartEvent
                self.C = EVENTs[1]
                self.D = EVENTs[2]
                self.F = EVENTs[3]
                self.I = EVENTs[4]
                self.G = EVENTs[5]



            
            # self.Force_Z[take_off_vel_index : landing_index] = 0
            # self.Force_X[take_off_vel_index : landing_index] = 0
            # self.Force_Y[take_off_vel_index : landing_index] = 0
            
            # self.ACCZ_B[take_off_vel_index : landing_index] = 0
            # self.ACCX_B[take_off_vel_index : landing_index] = 0
            # self.ACCY_B[take_off_vel_index : landing_index] = 0

            # self.GYROZ_B[take_off_vel_index : landing_index] = 0
            # self.GYROX_B[take_off_vel_index : landing_index] = 0
            # self.GYROY_B[take_off_vel_index : landing_index] = 0

            # self.ACCZ_G[take_off_vel_index : landing_index] = 0
            # self.ACCX_G[take_off_vel_index : landing_index] = 0
            # self.ACCY_G[take_off_vel_index : landing_index] = 0

            # self.GYROZ_G[take_off_vel_index : landing_index] = 0
            # self.GYROX_G[take_off_vel_index : landing_index] = 0
            # self.GYROY_G[take_off_vel_index : landing_index] = 0           


            # self.roll[take_off_vel_index : landing_index] = 0
            # self.yaw[take_off_vel_index : landing_index] = 0
            # self.pitch[take_off_vel_index : landing_index] = 0


            self.FT = (self.time[landing_index] - self.time[take_off_vel_index]) #Second
            self.take_off_vel_index = take_off_vel_index
            self.take_off_vel = take_off_vel
            self.landing_index = landing_index


            if (revers):

                self.ACCX_B =  -1 * self.ACCX_B
                self.ACCX_G =  -1 * self.ACCX_G
                self.ACCX_G2 =  -1 * self.ACCX_G2


                self.GYROY_B = -1 * self.GYROY_B
                self.GYROZ_B = -1 * self.GYROZ_B

                self.GYROY_G = -1 * self.GYROY_G
                self.GYROZ_G = -1 * self.GYROZ_G

                self.GYROY_G2 = -1 * self.GYROY_G2
                self.GYROZ_G2 = -1 * self.GYROZ_G2
                
                
                self.yaw = -1 * self.yaw                          
                self.pitch = -1 * self.pitch



 

            # Compute Angle based on GYRO 

            self.roll2 = processing.integral2(self.GYROX_B,self.time)
            self.pitch2 = processing.integral2(self.GYROY_B,self.time)
            self.yaw2 = processing.integral2(self.GYROZ_B,self.time)

            self.roll2[landing_index:] -=  self.roll2[landing_index-5]
            self.pitch2[landing_index:] -=  self.pitch2[landing_index-5]
            self.yaw2[landing_index:] -=  self.yaw2[landing_index-5]



            # self.setMainData(StartEvent,take_off_vel_index,landing_index,unWeighting_phase_index,Barking_phase_index)

            # self.featur_extraction(sub_code,pre_post,TaskName,SName)



            #height of jump
            self.JH =  processing.getHJ(take_off_vel)        
            self.JH_FT = processing.getHJ_FT(self.FT)

            if True: #self.JH > 0.1:

                #RSImod
                self.RSImod = processing.get_RSImod(self.JH,self.FT)

                #Force           
                self.Force_X[take_off_vel_index : landing_index] = 0
                self.Force_Y[take_off_vel_index : landing_index] = 0
                self.Force_Z[take_off_vel_index : landing_index] = 0

                self.PeakVerticalForceTakeoff = np.max(self.Force_Z[0:take_off_vel_index])/self.mass
                self.PeakVerticalForceLanding = np.max(self.Force_Z[landing_index:])/self.mass

                self.PeakMediolateralForceTakeoff = np.max(self.Force_X[0:take_off_vel_index])/self.mass
                self.PeakMediolateralForceLanding = np.max(self.Force_X[landing_index:])/self.mass
                

                # ---------------------------------------------------------------------------------------

                self.MaxACCVerticalTakeoff_L = np.max(self.ACCZ_B[0:take_off_vel_index])/self.mass
                self.MaxACCVerticalLanding_L = np.max(self.ACCZ_B[landing_index:])/self.mass
                self.MaxACCVerticalTakeoff_G = np.max(self.ACCZ_G[0:take_off_vel_index])/self.mass
                self.MaxACCVerticalLanding_G = np.max(self.ACCZ_G[landing_index:])/self.mass

                self.MaxACCHorizontalTakeoff_L = np.max(self.ACCX_B[0:take_off_vel_index])/self.mass
                self.MaxACCHorizontalLanding_L = np.max(self.ACCX_B[landing_index:])/self.mass
                self.MaxACCHorizontalTakeoff_G = np.max(self.ACCX_G[0:take_off_vel_index])/self.mass
                self.MaxACCHorizontalLanding_G = np.max(self.ACCX_G[landing_index:])/self.mass

                self.MaxGYROSagitalTakeoff_L = np.max(self.GYROX_B[0:take_off_vel_index])/self.mass
                self.MaxGYROSagitalLanding_L = np.max(self.GYROX_B[landing_index:])/self.mass
                self.MaxGYROSagitalTakeoff_G = np.max(self.GYROX_G[0:take_off_vel_index])/self.mass
                self.MaxGYROSagitalLanding_G = np.max(self.GYROX_G[landing_index:])/self.mass

                self.MaxGYROFrontalTakeoff_L = np.max(self.GYROZ_B[0:take_off_vel_index])/self.mass
                self.MaxGYROFrontalLanding_L = np.max(self.GYROZ_B[landing_index:])/self.mass
                self.MaxGYROFrontalTakeoff_G = np.max(self.GYROZ_G[0:take_off_vel_index])/self.mass
                self.MaxGYROFrontalLanding_G = np.max(self.GYROZ_G[landing_index:])/self.mass
                

                #Power
                # self.Power = processing.get_Power(self.Force,self.Velocity)
                # self.Peak_P_takeoff = np.max(self.Power[0:take_off_vel_index])/self.mass
                # self.Peak_P_landing = np.min(self.Force[landing_index:])/self.mass

            

                self.TimeMin2MaxTakeOFF = self.time[np.argmax(self.ACCZ_G[self.A:take_off_vel_index])] - self.time[np.argmin(self.ACCZ_G[self.A:take_off_vel_index])]
                self.Time2PeakLanding = self.time[landing_index + np.argmax(self.ACCZ_G[landing_index:])] - self.time[landing_index]

                # self.Min2MaxLanding = np.max(self.ACCZ_G[landing_index,:]) - np.min(self.ACCZ_G[landing_index,:])


                self.SkewnessLandingAcc = stats.skew(self.ACCZ_G[landing_index:] , axis=0,bias=True)
                self.KurtosisLandingAcc = stats.kurtosis(self.ACCZ_G[landing_index:], axis=0, bias=True)

                self.SkewnessLandingGyro = stats.skew(self.GYROZ_B[landing_index:] , axis=0,bias=True)
                self.KurtosisLandingGyro = stats.kurtosis(self.GYROZ_B[landing_index:], axis=0, bias=True)

                Steady_std = np.std(self.ACCZ_G[landing_index:])
                Steady_avg = np.average(self.ACCZ_G[landing_index:])

                self.LP90 = processing.Percentile(self.ACCZ_G[landing_index:],90)
                
                # SteadyLanding = [item+landing_index for item in range(len(self.ACCZ_B[landing_index:])) print(self.ACCZ_B[landing + item])] #if  self.ACCZ_B[landing_index+item] < Steady_avg+Steady_std and self.ACCZ_B[landing_index+item] > Steady_avg - Steady_std ]
                Steady_duration = 0
                for ii in range(len(self.ACCZ_G[landing_index:])):                             
                    if  self.ACCZ_G[landing_index+ii] < Steady_avg+Steady_std and self.ACCZ_G[landing_index+ii] > Steady_avg - Steady_std:
                        Steady_duration+=1
                        if Steady_duration > 5:
                            self.TimeLanding = self.time[landing_index+ii] - self.time[landing_index] 
                            break
                    else:
                        Steady_duration = 0                   
                


                if ("Shank" in  SName): 

                    self.TibFlexMax = np.max(self.roll[landing_index:])
                    self.TibFlexMin = np.min(self.roll[landing_index:])
                    self.TibFlexRange = abs(self.TibFlexMax - self.TibFlexMin)

                    self.TibVarMax = np.max(self.pitch[landing_index:])
                    self.TibVarMin = np.min(self.pitch[landing_index:])
                    self.TibVarRange = abs(self.TibVarMax - self.TibVarMin)


                    ################################ KAM

                    TibiaLen = processing.getTibiaLen(self.sub_height)
                    arm = [math.sin(math.radians(val))*TibiaLen for val in self.roll]  
                    Flex_moment = arm * self.ACCZ_G            
                    self.Moment_Sagital = Flex_moment
                    
                    KAM = processing.Compute_KAM(self.roll,self.yaw,TibiaLen,self.Main_Acc_Z , self.ACCX_G /9.81)
                    KAM2 = processing.Compute_KAM2(TibiaLen, self.Q , [self.ACCX_G , self.ACCZ_G , self.ACCZ_G])
                    KAM2 = np.array(KAM2)
                    self.Moment2_Flex = KAM2[:,0] 
                    self.Moment2_KAM = KAM2[:,1]

                    

                    # KAM['M1'] *= 9.81
                    # KAM['M2'] *= 9.81

                    self.Moment_KAM_1 = KAM['M1'] 
                    self.Moment_KAM_2 = KAM['M2'] 


                    self.M1_ACCV_Max = np.max(np.array(KAM['M1'])[landing_index:])
                    self.M2_ACCV_Max = np.max(np.array(KAM['M2'])[landing_index:])
                    self.KAM_ACCV_Max = self.M1_ACCV_Max + self.M2_ACCV_Max

                  
            

                # X = np.arange(0,len(self.roll[landing_index:]))
                # plt.plot(X,self.roll[landing_index:] , X , self.yaw[landing_index:])
                # plt.legend(['roll', 'yaw'])
                # plt.show()
                # input('OK')


                # Features Generation            

                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='JH', FeartureType="Param",Period='JMP',Value=self.JH))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='JHFT', FeartureType= 'Param',Period='JMP',Value=self.JH_FT))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='FT', FeartureType= 'Param',Period='JMP',Value=self.FT))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='Time2Stability',FeartureType="Param", Period='LAN',Value=self.TimeLanding))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='TimeMin2MaxTakeoff',FeartureType="Param" , Period='TAK',Value=self.TimeMin2MaxTakeOFF))
                self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='Time2PeakLanding', FeartureType="Param" ,Period='LAN',Value=self.Time2PeakLanding))
                # self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName, SensorName='#'+SName,SourceNAme='AAC',SourceFrame='G',Component='Z',Fearture='Min2MaxLanding', FeartureType="Param" ,Period='LAN',Value=self.Min2MaxLanding))

                
                if ("Shank" in  SName): 
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='QUA',SourceFrame='G',Component='X',Fearture='FlexRange',FeartureType="Range",Period='LAN',Value=self.TibFlexRange))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='QUA',SourceFrame='G',Component='Z',Fearture='AbAdRange',FeartureType="Range",Period='LAN',Value=self.TibVarRange))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='M1_ACCV_Max',FeartureType="Peak",Period='LAN',Value=self.M1_ACCV_Max))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='M2_ACCV_Max',FeartureType="Peak",Period='LAN',Value=self.M2_ACCV_Max))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='KAM_ACCV_Max',FeartureType="Peak",Period='LAN',Value=self.KAM_ACCV_Max))
                else:
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='QUA',SourceFrame='G',Component='X',Fearture='FlexRange',FeartureType="Range",Period='LAN',Value=np.nan))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='QUA',SourceFrame='G',Component='Z',Fearture='AbAdRange',FeartureType="Range",Period='LAN',Value=np.nan))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='M1_ACCV_Max',FeartureType="Peak",Period='LAN',Value=np.nan))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='M2_ACCV_Max',FeartureType="Peak",Period='LAN',Value=np.nan))
                    self.ALLFeatures.append(Features(subject=sub_code,pre_post=pre_post, TaskName= TaskName,SensorName='#'+SName,SourceNAme='TRQ',SourceFrame='G',Component='Z',Fearture='KAM_ACCV_Max',FeartureType="Peak",Period='LAN',Value=np.nan))








            else:
                self.addEmptyFeatures(sub_code , pre_post, TaskName , SName)
        except:
            print(TaskName ,sub_code ,pre_post)
            self.addEmptyFeatures(sub_code , pre_post, TaskName , SName)


    



@dataclass
class player_jumps():
    name: str
    age: int
    mass: int
    gender: str
    ID : str
    pass_n : str
    jumps : list[JUMP] = field(init=False, default_factory=list)



 


def Fusion(acc,gyro,mag,p):

    _ahrs = AHRS(SamplePeriod= p)
    
    q = []

    for i in range(0,500):
        _ahrs.Update_IMU(acc[0],gyro[0],mag[0])


    for inx in range(len(acc)):        
        _ahrs.Update_IMU(acc[inx],gyro[inx],mag[inx])
        _q  = _ahrs.get_Quat()
        q.append(quat(w = _q.W, x = _q.X, y = _q.Y, z = _q.Z))        

    return q

def Fusion2(acc,gyro,p):

    _ahrs = AHRS(SamplePeriod= p)
    
    q = []

    for i in range(0,500):
        _ahrs.Update_IMU2([acc[0][0],acc[0][1],acc[0][2]] ,[gyro[0][0],gyro[0][1],gyro[0][2]])


    for inx in range(len(acc)):        
        _ahrs.Update_IMU2([acc[inx][0],acc[inx][1],acc[inx][2]] ,[gyro[inx][0],gyro[inx][1],gyro[inx][2]])
        _q  = _ahrs.get_Quat()
        q.append(quat(w = _q.W, x = _q.X, y = _q.Y, z = _q.Z))        

    return q



def findStartPointMadgwicks(gyro , w , thershould):

    tmp  = np.array(gyro)
    x = np.abs(tmp[:,0])
    y = np.abs(tmp[:,1])
    z = np.abs(tmp[:,2])
   
    sum = x + y + z

    for i in range(len(sum)):                    
        _std = np.average(sum[i:i+100])

        if _std < thershould:
            return i

    return 0



def SignalModifier(acc,gyro,startPoint):

    tmp  = np.array(acc)
    x = tmp[:,0]
    y = tmp[:,1]
    z = tmp[:,2]

    # x[0:startPoint] = 0
    # y[0:startPoint] = 0
    # z[0:startPoint] = 0

    newAcc = [x ,y , z]

    tmp  = np.array(gyro)
    x = tmp[:,0]
    y = tmp[:,1]
    z = tmp[:,2]

    x[0:startPoint] = 0
    y[0:startPoint] = 0
    z[0:startPoint] = 0

    newGyro = [x ,y , z]

    return [newAcc , newGyro] 


    
   




def computeOrientation(acc,gyro,p,beta , startIndex=0 , itr=1000):
    q = []
    _ahrs =  MadgwickAHRS(p,quaternion=Quaternion(1,0,0,0), beta=beta)
    
    avg_gyX = 0
    avg_gyY = 0
    avg_gyZ = 0

    avg_accX = 0
    avg_accY = 0
    avg_accZ = 0

    for i in range(startIndex,startIndex+50):
        avg_gyX += gyro[i][0]
        avg_gyY += gyro[i][1]
        avg_gyZ += gyro[i][2]

        avg_accX += acc[i][0]
        avg_accY += acc[i][1]
        avg_accZ += acc[i][2]


    avg_accX /= 50
    avg_accY /= 50
    avg_accZ /= 50

    avg_gyX /= 50
    avg_gyY /= 50
    avg_gyZ /= 50   


    for i in range(0,itr):
        _ahrs.update_imu([0,0,0],[avg_accX,avg_accY,avg_accZ])    


    for inx in range(startIndex): 
        q.append(quat(w = 1, x = 0, y = 0, z = 0))       

    for inx in range(startIndex,len(acc)):      
        _ahrs.update_imu([gyro[inx][0]* math.pi/180    ,gyro[inx][1] * math.pi/180,gyro[inx][2] * math.pi/180],[acc[inx][0],acc[inx][1],acc[inx][2]])
        _q  = _ahrs.quaternion.q
        q.append(quat(w = _q[0], x = _q[1], y = _q[2], z = _q[3]))        
        # q.append(quat(w = 1, x = 0, y = 0, z = 0))        


    return q



def computeOrientation_mag(acc,gyro,mag,p , beta):
    q = []
    _ahrs =  MadgwickAHRS(p,quaternion=Quaternion(1,0,0,0), beta=beta)
    

    for i in range(0,500):
        _ahrs.update([gyro[0][0]* math.pi/180    ,gyro[0][1] * math.pi/180,gyro[0][2] * math.pi/180],[acc[0][0],acc[0][1],acc[0][2]] , [mag[0][0] , mag[0][1] , mag[0][2]])



    for inx in range(len(acc)):
        _ahrs.update([gyro[inx][0]* math.pi/180    ,gyro[inx][1] * math.pi/180,gyro[inx][2] * math.pi/180],[acc[inx][0],acc[inx][1],acc[inx][2]],[mag[inx][0] , mag[inx][1] , mag[inx][2]])
        _q  = _ahrs.quaternion.q
        q.append(quat(w = _q[0], x = _q[1], y = _q[2], z = _q[3]))        

    return q




def Apply_DownSampling(_S):
    
    freq =  int(round(len(_S.time) / 500 * 100))

    print(freq)

    if (freq!=0):
        _S.time = processing.DownSample(_S.time,freq)
        _S.AX = processing.DownSample(_S.AX,freq)
        _S.AY = processing.DownSample(_S.AY,freq)
        _S.AZ = processing.DownSample(_S.AZ,freq)

        _S.GX = processing.DownSample(_S.GX,freq)
        _S.GY = processing.DownSample(_S.GY,freq)
        _S.GZ = processing.DownSample(_S.GZ,freq)

        _S.MX = processing.DownSample(_S.MX,freq)
        _S.MY = processing.DownSample(_S.MY,freq)
        _S.MZ = processing.DownSample(_S.MZ,freq)




# read mat files
def read_mat(path,events,e_i , DownSample):

    sample_period = 0.002

    list_task = []

    rawData = loadmat(path)   
  

    imus = rawData['I'].dtype.names

    # Tasks
    _static = myTask('Static')
    _supine = myTask('Supine')

    _CMJ = [] 
    for t in events['CMJ'][e_i]:
        _CMJ.append(myTask('CMJ'))

    _RSJ = [] 
    for t in events['RSJ'][e_i]:
        _RSJ.append(myTask('RSJ'))

    _LSJ = [] 
    for t in events['LSJ'][e_i]:
        _LSJ.append(myTask('LSJ'))

    _RDJ = [] 
    for t in events['RDJ'][e_i]:
        _RDJ.append(myTask('RDJ'))

    _LDJ = [] 
    for t in events['LDJ'][e_i]:
        _LDJ.append(myTask('LDJ'))

    _RCUT = [] 
    for t in events['RCUT'][e_i]:
        _RCUT.append(myTask('RCUT'))

    _LCUT = [] 
    for t in events['LCUT'][e_i]:
        _LCUT.append(myTask('LCUT'))

    for s in imus:
        imu = rawData['I'][s][0][0]
        acc = imu['acc'][0,0]
        gyro = imu['gyr'][0,0]
        mag = imu['mag'][0,0]

        # 500 Hz
        acc = np.array(acc)
        gyro = np.array(gyro)
        mag = np.array(mag)
        time = np.arange(0,(len(acc)*sample_period),sample_period)

      
       
        pos_index1 = events['Static'][e_i][0]
        pos_index2 = events['Static'][e_i][1]
        _S = IMU_Sensor(name=s)        

        _S.time = time[pos_index1:pos_index2]
        _S.AX = acc[pos_index1:pos_index2,0]
        _S.AY = acc[pos_index1:pos_index2,1]
        _S.AZ = acc[pos_index1:pos_index2,2]

        _S.GX = gyro[pos_index1:pos_index2,0]
        _S.GY = gyro[pos_index1:pos_index2,1]
        _S.GZ = gyro[pos_index1:pos_index2,2]

        _S.MX = mag[pos_index1:pos_index2,0]
        _S.MY = mag[pos_index1:pos_index2,1]
        _S.MZ = mag[pos_index1:pos_index2,2]      

        if DownSample:
            Apply_DownSampling(_S)
        _static.Data.append(_S)    

        
        pos_index1 = events['Supine'][e_i][0]
        pos_index2 = events['Supine'][e_i][1]
        _S = IMU_Sensor(name=s)
        _S.time = time[pos_index1:pos_index2]
        _S.AX = acc[pos_index1:pos_index2,0]
        _S.AY = acc[pos_index1:pos_index2,1]
        _S.AZ = acc[pos_index1:pos_index2,2]

        _S.GX = gyro[pos_index1:pos_index2,0]
        _S.GY = gyro[pos_index1:pos_index2,1]
        _S.GZ = gyro[pos_index1:pos_index2,2]

        _S.MX = mag[pos_index1:pos_index2,0]
        _S.MY = mag[pos_index1:pos_index2,1]
        _S.MZ = mag[pos_index1:pos_index2,2]
   
        if DownSample:
            Apply_DownSampling(_S)
        _supine.Data.append(_S)       


        # CMJ
        
        for i,t in enumerate(events['CMJ'][e_i]):           
            pos_index1 = t[0]
            pos_index2 = t[1]
            _S = IMU_Sensor(name=s)
            _S.time = time[pos_index1:pos_index2]
           

            _S.AX = acc[pos_index1:pos_index2,0]
            _S.AY = acc[pos_index1:pos_index2,1]
            _S.AZ = acc[pos_index1:pos_index2,2]

            _S.GX = gyro[pos_index1:pos_index2,0]
            _S.GY = gyro[pos_index1:pos_index2,1]
            _S.GZ = gyro[pos_index1:pos_index2,2]

            _S.MX = mag[pos_index1:pos_index2,0]
            _S.MY = mag[pos_index1:pos_index2,1]
            _S.MZ = mag[pos_index1:pos_index2,2]

            if DownSample:
                Apply_DownSampling(_S)
            _CMJ[i].Data.append(_S) 
                  

        # RSJ
        
        for i,t in enumerate(events['RSJ'][e_i]):           
            pos_index1 = t[0]
            pos_index2 = t[1]
            _S = IMU_Sensor(name=s)
            _S.time = time[pos_index1:pos_index2]
            _S.AX = acc[pos_index1:pos_index2,0]
            _S.AY = acc[pos_index1:pos_index2,1]
            _S.AZ = acc[pos_index1:pos_index2,2]

            _S.GX = gyro[pos_index1:pos_index2,0]
            _S.GY = gyro[pos_index1:pos_index2,1]
            _S.GZ = gyro[pos_index1:pos_index2,2]

            _S.MX = mag[pos_index1:pos_index2,0]
            _S.MY = mag[pos_index1:pos_index2,1]
            _S.MZ = mag[pos_index1:pos_index2,2]

            if DownSample:
                Apply_DownSampling(_S)
            _RSJ[i].Data.append(_S)

        

        # LSJ
       
        for i,t in enumerate(events['LSJ'][e_i]):
            
            pos_index1 = t[0]
            pos_index2 = t[1]
            _S = IMU_Sensor(name=s)
            _S.time = time[pos_index1:pos_index2]
            _S.AX = acc[pos_index1:pos_index2,0]
            _S.AY = acc[pos_index1:pos_index2,1]
            _S.AZ = acc[pos_index1:pos_index2,2]

            _S.GX = gyro[pos_index1:pos_index2,0]
            _S.GY = gyro[pos_index1:pos_index2,1]
            _S.GZ = gyro[pos_index1:pos_index2,2]

            _S.MX = mag[pos_index1:pos_index2,0]
            _S.MY = mag[pos_index1:pos_index2,1]
            _S.MZ = mag[pos_index1:pos_index2,2]

            if DownSample:
                Apply_DownSampling(_S)

            _LSJ[i].Data.append(_S)
          

        # RDJ
       
        for i,t in enumerate(events['RDJ'][e_i]):

            pos_index1 = t[0]
            pos_index2 = t[1]
            _S = IMU_Sensor(name=s)
            _S.time = time[pos_index1:pos_index2]
            _S.AX = acc[pos_index1:pos_index2,0]
            _S.AY = acc[pos_index1:pos_index2,1]
            _S.AZ = acc[pos_index1:pos_index2,2]

            _S.GX = gyro[pos_index1:pos_index2,0]
            _S.GY = gyro[pos_index1:pos_index2,1]
            _S.GZ = gyro[pos_index1:pos_index2,2]

            _S.MX = mag[pos_index1:pos_index2,0]
            _S.MY = mag[pos_index1:pos_index2,1]
            _S.MZ = mag[pos_index1:pos_index2,2]

            if DownSample:
                Apply_DownSampling(_S)

            _RDJ[i].Data.append(_S)
          


        # LDJ
        
        for i,t in enumerate(events['LDJ'][e_i]):
            
            pos_index1 = t[0]
            pos_index2 = t[1]
            _S = IMU_Sensor(name=s)
            _S.time = time[pos_index1:pos_index2]
            _S.AX = acc[pos_index1:pos_index2,0]
            _S.AY = acc[pos_index1:pos_index2,1]
            _S.AZ = acc[pos_index1:pos_index2,2]

            _S.GX = gyro[pos_index1:pos_index2,0]
            _S.GY = gyro[pos_index1:pos_index2,1]
            _S.GZ = gyro[pos_index1:pos_index2,2]

            _S.MX = mag[pos_index1:pos_index2,0]
            _S.MY = mag[pos_index1:pos_index2,1]
            _S.MZ = mag[pos_index1:pos_index2,2]

            if DownSample:
                Apply_DownSampling(_S)

            _LDJ[i].Data.append(_S)
            

        # RCUT
       
        for i,t in enumerate(events['RCUT'][e_i]):
           
            pos_index1 = t[0]
            pos_index2 = t[1]
            _S = IMU_Sensor(name=s)
            _S.time = time[pos_index1:pos_index2]
            _S.AX = acc[pos_index1:pos_index2,0]
            _S.AY = acc[pos_index1:pos_index2,1]
            _S.AZ = acc[pos_index1:pos_index2,2]

            _S.GX = gyro[pos_index1:pos_index2,0]
            _S.GY = gyro[pos_index1:pos_index2,1]
            _S.GZ = gyro[pos_index1:pos_index2,2]

            _S.MX = mag[pos_index1:pos_index2,0]
            _S.MY = mag[pos_index1:pos_index2,1]
            _S.MZ = mag[pos_index1:pos_index2,2]

            if DownSample:
                Apply_DownSampling(_S)

            _RCUT[i].Data.append(_S)
           

        # LCUT
        
        for i,t in enumerate(events['LCUT'][e_i]):

            pos_index1 = t[0]
            pos_index2 = t[1]
            _S = IMU_Sensor(name=s)
            _S.time = time[pos_index1:pos_index2]
            _S.AX = acc[pos_index1:pos_index2,0]
            _S.AY = acc[pos_index1:pos_index2,1]
            _S.AZ = acc[pos_index1:pos_index2,2]

            _S.GX = gyro[pos_index1:pos_index2,0]
            _S.GY = gyro[pos_index1:pos_index2,1]
            _S.GZ = gyro[pos_index1:pos_index2,2]

            _S.MX = mag[pos_index1:pos_index2,0]
            _S.MY = mag[pos_index1:pos_index2,1]
            _S.MZ = mag[pos_index1:pos_index2,2]

            if DownSample:
                Apply_DownSampling(_S)

            _LCUT[i].Data.append(_S)
           
    
    list_task.append(_static)
    list_task.append(_supine)
    list_task += _CMJ
    list_task += _LSJ
    list_task += _RSJ
    list_task += _RDJ
    list_task += _LDJ
    list_task += _RCUT
    list_task += _LCUT


    return list_task

        




def read_xlsx_mass(_path):
    df = pd.read_excel(_path)
    output ={}
    names = df['Name']
    for inx,n in enumerate(names):
        output[n] = {'W' : df['W'][inx] , 'H' : df['H'][inx] }


    return output




# def main():
#     # Main

#     IMU_Number = 4  #,3,2] #Pelvic , Rshank , Lshank

#     all_subjects = []

#     # Read Events Frame from XLSX File
#     Events = read_data.read_xlsx_IMU_Frames(os.path.join(os.getcwd(),'movafaghian/Frames.xlsx'))


#     # Read mat files
#     all_files = os.listdir(os.path.join(os.getcwd(),'movafaghian/IMU_I'))


#     # output
#     file_output = open("output1.csv", "w")
#     param = ["File" , "Jump" , "HJ" , "FT", "Peak Force Vertical - TakeOff" , "Peak Force Vertical - Landing" , "Peak Force Mediolateral - TakeOff", "Peak Force Mediolateral - Landing" ]
#     file_output.write(",".join(param)+'\n')

#     workbook = xlsxwriter.Workbook('IMU_Extraxted_Data.xlsx')
#     info = read_xlsx_mass('ATH_W_H.xlsx')

#     for f in all_files:
#         sub_code = f.split('_')[0]

#         subject_info = info[sub_code]
#         _np = player(f,20,subject_info['W'],subject_info['H'],'M',9999,0)   

#         ev_index = Events['Name'].index(f.split('.')[0])
#         _np.Tasks = read_mat( os.path.join(os.getcwd(),'movafaghian/IMU_I/',f),Events,ev_index)

        
        

#         worksheet = workbook.add_worksheet(_np.name)        
#         # Trunk - Sensor 2 Body 
#         ACC_standing = np.array([_np.Tasks[0].Data[IMU_Number].AX ,_np.Tasks[0].Data[IMU_Number].AY, _np.Tasks[0].Data[IMU_Number].AZ ])
#         ACC_laying = np.array([_np.Tasks[1].Data[IMU_Number].AX ,_np.Tasks[1].Data[IMU_Number].AY, _np.Tasks[1].Data[IMU_Number].AZ ])
#         trunk_results = processing.Compute_Sensor2Body_rotation(ACC_standing,ACC_laying)

#         # Apply to other sensors
#         # ax1 = plt.subplot(231)    
#         # ax1.plot(_np.Tasks[0].Data[5].time,_np.Tasks[0].Data[5].AX)
#         # ax1.plot(_np.Tasks[0].Data[5].time,trunk_results['test'][:,0])

#         # ax2 = plt.subplot(232)    
#         # ax2.plot(_np.Tasks[0].Data[5].time,_np.Tasks[0].Data[5].AY)
#         # ax2.plot(_np.Tasks[0].Data[5].time,trunk_results['test'][:,1])

#         # ax3 = plt.subplot(233)    
#         # ax3.plot(_np.Tasks[0].Data[5].time,_np.Tasks[0].Data[5].AZ)
#         # ax3.plot(_np.Tasks[0].Data[5].time,trunk_results['test'][:,2])

#         # ax1 = plt.subplot(234)    
#         # ax1.plot(_np.Tasks[1].Data[5].time,_np.Tasks[1].Data[5].AX)
#         # ax1.plot(_np.Tasks[1].Data[5].time,trunk_results['eval'][:,0])

#         # ax2 = plt.subplot(235)    
#         # ax2.plot(_np.Tasks[1].Data[5].time,_np.Tasks[1].Data[5].AY)
#         # ax2.plot(_np.Tasks[1].Data[5].time,trunk_results['eval'][:,1])

#         # ax3 = plt.subplot(236)    
#         # ax3.plot(_np.Tasks[1].Data[5].time,_np.Tasks[1].Data[5].AZ)
#         # ax3.plot(_np.Tasks[1].Data[5].time,trunk_results['eval'][:,2])

#         # plt.show()

#         # Trunk - Apply rotation to ACC & Gyro data
        


#         # Compute Euler of Shank in Stand pos
#         # Right Shank
#         ACC_standing = np.array([_np.Tasks[0].Data[3].AX ,_np.Tasks[0].Data[3].AY, _np.Tasks[0].Data[3].AZ ])
#         ACC_laying = np.array([_np.Tasks[1].Data[3].AX ,_np.Tasks[1].Data[3].AY, _np.Tasks[1].Data[3].AZ ])
#         Rotation_results = processing.Compute_Sensor2Body_rotation(ACC_standing,ACC_laying)

#         ACC_temp = np.array([_np.Tasks[0].Data[3].AX ,_np.Tasks[0].Data[3].AY, _np.Tasks[0].Data[3].AZ ])
#         GYRO_temp = np.array([_np.Tasks[0].Data[3].GX ,_np.Tasks[0].Data[3].GY, _np.Tasks[0].Data[3].GZ ])
#         corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_temp)
#         corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_temp)
#         # lp_corrected_ACC =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_ACC))
#         # lp_corrected_GYRO =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_GYRO))
#         lp_corrected_ACC =  corrected_ACC
#         lp_corrected_GYRO =  corrected_GYRO

#         # Compute Orientation
#         _q = computeOrientation(lp_corrected_ACC,lp_corrected_GYRO,dt_ms/1000)

#         # Q2E
#         Euler = np.array([i.get_euler() for i in _q ])                                               
#         RS_roll = np.average([i.roll for i in Euler])
#         RS_pitch = np.average([i.pitch for i in Euler])
#         RS_yaw = np.average([i.yaw for i in Euler])


#         # Left Shank
#         ACC_standing = np.array([_np.Tasks[0].Data[2].AX ,_np.Tasks[0].Data[2].AY, _np.Tasks[0].Data[2].AZ ])
#         ACC_laying = np.array([_np.Tasks[1].Data[2].AX ,_np.Tasks[1].Data[2].AY, _np.Tasks[1].Data[2].AZ ])
#         Rotation_results = processing.Compute_Sensor2Body_rotation(ACC_standing,ACC_laying)

#         ACC_temp = np.array([_np.Tasks[0].Data[2].AX ,_np.Tasks[0].Data[2].AY, _np.Tasks[0].Data[2].AZ ])
#         GYRO_temp = np.array([_np.Tasks[0].Data[2].GX ,_np.Tasks[0].Data[2].GY, _np.Tasks[0].Data[2].GZ ])
#         corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_temp)
#         corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_temp)
#         # lp_corrected_ACC =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_ACC))
#         # lp_corrected_GYRO =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_GYRO))
#         lp_corrected_ACC =  corrected_ACC
#         lp_corrected_GYRO =  corrected_GYRO

#         # Compute Orientation
#         _q = computeOrientation(lp_corrected_ACC,lp_corrected_GYRO,dt_ms/1000)

#         # Q2E
#         Euler = np.array([i.get_euler() for i in _q ])                                               
#         LS_roll = np.average([i.roll for i in Euler])
#         LS_pitch = np.average([i.pitch for i in Euler])
#         LS_yaw = np.average([i.yaw for i in Euler])





#         # Tasks
#         type_tmp = ''
#         c = 0
#         for inx in range(2,len(_np.Tasks)):
            
#             if _np.Tasks[inx].name=='CMJ' or _np.Tasks[inx].name=='LSJ' or _np.Tasks[inx].name=='RSJ' or _np.Tasks[inx].name=='LDJ' or _np.Tasks[inx].name=='RDJ':


#                 if type_tmp != _np.Tasks[inx].name:
#                     type_tmp =  _np.Tasks[inx].name
#                     c = 1
#                 else:
#                     c += 1
            


#                 ACC_temp = np.array([_np.Tasks[inx].Data[IMU_Number].AX ,_np.Tasks[inx].Data[IMU_Number].AY, _np.Tasks[inx].Data[IMU_Number].AZ ])
#                 GYRO_temp = np.array([_np.Tasks[inx].Data[IMU_Number].GX ,_np.Tasks[inx].Data[IMU_Number].GY, _np.Tasks[inx].Data[IMU_Number].GZ ])
#                 corrected_ACC = processing.Apply_Sensor2Body(trunk_results['R1'],trunk_results['R2'],ACC_temp)
#                 corrected_GYRO = processing.Apply_Sensor2Body(trunk_results['R1'],trunk_results['R2'],GYRO_temp)
#                 # lp_corrected_ACC =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_ACC))
#                 # lp_corrected_GYRO =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_GYRO))
#                 lp_corrected_ACC =  corrected_ACC
#                 lp_corrected_GYRO =  corrected_GYRO

#                 # Compute Orientation
#                 _q = computeOrientation(lp_corrected_ACC,lp_corrected_GYRO,dt_ms/1000)
#                 _np.Tasks[inx].Data[IMU_Number].Q = _q

#                 # View3D.showIMUOrientation([_np.Tasks[inx].Data[IMU_Number]])



#                 _np.Tasks[inx].Data[IMU_Number].Free_GAX = (np.array(lp_corrected_ACC)[:,0]) * 9.81
#                 _np.Tasks[inx].Data[IMU_Number].Free_GAY = (np.array(lp_corrected_ACC)[:,1]-1) * 9.81
#                 _np.Tasks[inx].Data[IMU_Number].Free_GAZ = (np.array(lp_corrected_ACC)[:,2]) * 9.81

#                 _np.Tasks[inx].Data[IMU_Number].AX = (np.array(lp_corrected_ACC)[:,0])
#                 _np.Tasks[inx].Data[IMU_Number].AY = (np.array(lp_corrected_ACC)[:,1])
#                 _np.Tasks[inx].Data[IMU_Number].AZ = (np.array(lp_corrected_ACC)[:,2])

#                 vertical_V = processing.integral2(_np.Tasks[inx].Data[IMU_Number].Free_GAY ,_np.Tasks[inx].Data[IMU_Number].time)
#                 try:
#                     vertical_V_C = processing.Velocity_correction(vertical_V, _np.Tasks[inx].Data[IMU_Number].Free_GAY,1000/dt_ms)
#                 except:
#                     print('...')
            

#                 # plt.plot(np.arange(len(_np.Tasks[inx].Data[IMU_Number].time)), _np.Tasks[inx].Data[IMU_Number].Free_GAY)            
#                 # plt.show()

#                 # plt.plot(np.arange(len(_np.Tasks[inx].Data[IMU_Number].time)), vertical_V)            
#                 # plt.show()

#                 # plt.plot(np.arange(len(_np.Tasks[inx].Data[IMU_Number].time)), vertical_V_C)            
#                 # plt.show()

#                 # input("......Vel")


#                 # Create Jump            
#                 _J = JUMP(_np.Tasks[inx].name)
#                 _J.Velocity = vertical_V_C
#                 _J.ACCZ_B = processing.low_pass_filter(dt_ms, _np.Tasks[inx].Data[IMU_Number].AY)
#                 _J.ACCX_B = processing.low_pass_filter(dt_ms,_np.Tasks[inx].Data[IMU_Number].AX) * -1
#                 _J.ACCY_B = processing.low_pass_filter(dt_ms,_np.Tasks[inx].Data[IMU_Number].AZ)
#                 _J.ACCZ =  ACC_temp[2]
#                 _J.ACCX =  ACC_temp[0]
#                 _J.ACCY =  ACC_temp[1]
                


#                 _J.time = _np.Tasks[inx].Data[IMU_Number].time
#                 _J.Force_X = processing.low_pass_filter(dt_ms,_np.Tasks[inx].Data[IMU_Number].Free_GAX * _np.mass)
#                 _J.Force_Z = processing.low_pass_filter(dt_ms,_np.Tasks[inx].Data[IMU_Number].Free_GAY * _np.mass)
#                 _J.Force_Y = processing.low_pass_filter(dt_ms,_np.Tasks[inx].Data[IMU_Number].Free_GAZ * _np.mass)


            

            

#                 try:                               
#                     _J.run()


#                     _J.Force_Z[_J.take_off_vel_index : _J.landing_index] = 0
#                     _J.Force_X[_J.take_off_vel_index : _J.landing_index] = 0
#                     _J.Force_Y[_J.take_off_vel_index : _J.landing_index] = 0

                    
#                     _J.ACCZ_B[_J.take_off_vel_index : _J.landing_index] = 0
#                     _J.ACCX_B[_J.take_off_vel_index : _J.landing_index] = 0
#                     _J.ACCY_B[_J.take_off_vel_index : _J.landing_index] = 0


#                     #################################################################################
#                     # Shank Data Process
#                     Shank_IMU_Number = -1
#                     if "R" in _np.Tasks[inx].name:
#                         Shank_IMU_Number = 3
#                     elif "L" in _np.Tasks[inx].name:
#                         Shank_IMU_Number = 2
#                     elif "CMJ" in _np.Tasks[inx].name:
#                         Shank_IMU_Number = 100


#                     roll = []
#                     pitch = []
#                     yaw = []

#                     if Shank_IMU_Number == 100:
#                         ACC_standing = np.array([_np.Tasks[0].Data[3].AX ,_np.Tasks[0].Data[3].AY, _np.Tasks[0].Data[3].AZ ])
#                         ACC_laying = np.array([_np.Tasks[1].Data[3].AX ,_np.Tasks[1].Data[3].AY, _np.Tasks[1].Data[3].AZ ])
#                         Rotation_results = processing.Compute_Sensor2Body_rotation(ACC_standing,ACC_laying)


#                         ACC_temp = np.array([_np.Tasks[inx].Data[3].AX ,_np.Tasks[inx].Data[3].AY, _np.Tasks[inx].Data[3].AZ ])
#                         GYRO_temp = np.array([_np.Tasks[inx].Data[3].GX ,_np.Tasks[inx].Data[3].GY, _np.Tasks[inx].Data[3].GZ ])
#                         corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_temp)
#                         corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_temp)
#                         # lp_corrected_ACC =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_ACC))
#                         # lp_corrected_GYRO =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_GYRO))
#                         lp_corrected_ACC =  corrected_ACC
#                         lp_corrected_GYRO =  corrected_GYRO
                
#                         # Compute Orientation
#                         _q = computeOrientation(lp_corrected_ACC,lp_corrected_GYRO,dt_ms/1000)

#                         # Q2E
#                         Euler = np.array([i.get_euler() for i in _q ])                                               
#                         roll = [i.roll for i in Euler]
#                         pitch = [i.pitch for i in Euler]
#                         yaw = [i.yaw for i in Euler]


#                         ACC_standing = np.array([_np.Tasks[0].Data[2].AX ,_np.Tasks[0].Data[2].AY, _np.Tasks[0].Data[2].AZ ])
#                         ACC_laying = np.array([_np.Tasks[1].Data[2].AX ,_np.Tasks[1].Data[2].AY, _np.Tasks[1].Data[2].AZ ])
#                         Rotation_results = processing.Compute_Sensor2Body_rotation(ACC_standing,ACC_laying)

#                         ACC_temp = np.array([_np.Tasks[inx].Data[2].AX ,_np.Tasks[inx].Data[2].AY, _np.Tasks[inx].Data[2].AZ ])
#                         GYRO_temp = np.array([_np.Tasks[inx].Data[2].GX ,_np.Tasks[inx].Data[2].GY, _np.Tasks[inx].Data[2].GZ ])
#                         corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_temp)
#                         corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_temp)
#                         # lp_corrected_ACC =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_ACC))
#                         # lp_corrected_GYRO =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_GYRO))
#                         lp_corrected_ACC = corrected_ACC
#                         lp_corrected_GYRO = corrected_GYRO
    

#                         # Compute Orientation
#                         _q = computeOrientation(lp_corrected_ACC,lp_corrected_GYRO,dt_ms/1000)

#                         # Q2E
#                         Euler = np.array([i.get_euler() for i in _q ]) 

#                         roll = [sum(i)/2 for i in zip(roll, [i.roll for i in Euler])]
#                         pitch = [sum(i)/2 for i in zip(pitch, [i.pitch for i in Euler])]
#                         yaw = [sum(i)/2 for i in zip(yaw, [i.yaw for i in Euler])]

#                         # roll_offset= np.average(roll[0:50])
#                         # roll -= (LS_roll+RS_roll)/2
#                         # pitch -= (LS_pitch+RS_pitch)/2
#                         # yaw -= (LS_yaw+RS_yaw)/2

#                         roll -= np.average(roll[0:50])
#                         yaw -= np.average(yaw[0:50])

#                         roll[_J.take_off_vel_index : _J.landing_index] = 0
#                         yaw[_J.take_off_vel_index : _J.landing_index] = 0
#                         pitch[_J.take_off_vel_index : _J.landing_index] = 0
                        

#                         _J.Horizontal_ACC = _J.ACCX_B
#                         _J.Vertical_ACC = _J.ACCZ_B

#                         _J.Horizontal_ACC[_J.take_off_vel_index : _J.landing_index] = 0
#                         _J.Vertical_ACC[_J.take_off_vel_index : _J.landing_index] = 0

                        
#                         print("Display -->CMJ")
#                         # Xp = np.arange(0,len(_q))
#                         # plt.plot(Xp,roll,Xp,pitch,Xp,yaw)
                        
#                         # plt.show()

#                         # # Lever Arm                    
#                         # print("Stop!! -->Lever arm")
                        
#                         TibiaLen = processing.getTibiaLen(_np.height)
#                         arm = [math.sin(math.radians(val))*TibiaLen for val in roll]  
#                         Flex_moment = arm * _J.Force_Z
#                         Flex_moment /= _np.mass
#                         _J.Moment_Sagital = Flex_moment


                        
#                         KAM = processing.Compute_KAM(roll,yaw,TibiaLen,_J.Vertical_ACC , _J.Horizontal_ACC)

#                         KAM['M1'] *= 9.81
#                         KAM['M2'] *= 9.81

#                         _J.Moment_KAM_1 = KAM['M1']
#                         _J.Moment_KAM_2 = KAM['M2']
                        

#                         Xp = np.arange(0,len(_J.time))
#                         plotData.plot_tibia_angle_leverArm_FlexMoment(Xp,roll,arm,_J.ACCZ_B,Flex_moment,sub_code,sub_code+str("_Moment1_CMJ")+str(c))
#                         plotData.plot_tibia_angle_leverArm_KAM(Xp,roll,yaw,KAM['arm1'],KAM['arm2'],_J.ACCZ_B, _J.Force_X,KAM['M1'], KAM['M2'],sub_code, sub_code+str("_Moment1_")+_np.Tasks[inx].name+str(c))

    
#                     elif Shank_IMU_Number != -1:
#                         ACC_standing = np.array([_np.Tasks[0].Data[Shank_IMU_Number].AX ,_np.Tasks[0].Data[Shank_IMU_Number].AY, _np.Tasks[0].Data[Shank_IMU_Number].AZ ])
#                         ACC_laying = np.array([_np.Tasks[1].Data[Shank_IMU_Number].AX ,_np.Tasks[1].Data[Shank_IMU_Number].AY, _np.Tasks[1].Data[Shank_IMU_Number].AZ ])
#                         Rotation_results = processing.Compute_Sensor2Body_rotation(ACC_standing,ACC_laying)

#                         ACC_temp = np.array([_np.Tasks[inx].Data[Shank_IMU_Number].AX ,_np.Tasks[inx].Data[Shank_IMU_Number].AY, _np.Tasks[inx].Data[Shank_IMU_Number].AZ ])
#                         GYRO_temp = np.array([_np.Tasks[inx].Data[Shank_IMU_Number].GX ,_np.Tasks[inx].Data[Shank_IMU_Number].GY, _np.Tasks[inx].Data[Shank_IMU_Number].GZ ])
#                         corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_temp)
#                         corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_temp)
#                         # lp_corrected_ACC =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_ACC))
#                         # lp_corrected_GYRO =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_GYRO))
#                         lp_corrected_ACC =  corrected_ACC
#                         lp_corrected_GYRO =  corrected_GYRO


#                         _J.Horizontal_ACC = np.array(lp_corrected_ACC)[:,0]
#                         _J.Vertical_ACC = np.array(lp_corrected_ACC)[:,1]

                        
#                         _J.Horizontal_ACC[_J.take_off_vel_index : _J.landing_index] = 0
#                         _J.Vertical_ACC[_J.take_off_vel_index : _J.landing_index] = 0



#                         # Compute Orientation
#                         _q = computeOrientation(lp_corrected_ACC,lp_corrected_GYRO,dt_ms/1000)

#                         # Q2E
#                         Euler = np.array([i.get_euler() for i in _q ])                                               
#                         roll = [i.roll for i in Euler]
#                         pitch = [i.pitch for i in Euler]
#                         yaw = [i.yaw for i in Euler]


#                         Xp = np.arange(0,len(_q))
#                         w = [q.w for q in _q]
#                         x = [q.x for q in _q]
#                         y = [q.y for q in _q]
#                         z = [q.z for q in _q]

#                         #roll_offset= np.average(roll[0:50])
#                         # if Shank_IMU_Number==2:
#                         #     roll -= LS_roll
#                         #     pitch -= LS_pitch
#                         #     yaw -= LS_yaw
#                         # else :
#                         #     roll -= RS_roll
#                         #     pitch -= RS_pitch
#                         #     yaw -= RS_yaw

#                         roll -= np.average(roll[0:50])
#                         yaw -= np.average(yaw[0:50])

#                         roll[_J.take_off_vel_index : _J.landing_index] = 0
#                         yaw[_J.take_off_vel_index : _J.landing_index] = 0
                        


#                         # Lever Arm    
#                         TibiaLen = processing.getTibiaLen(_np.height)               
#                         print("Display!! -->SL " + str(Shank_IMU_Number))                   
#                         arm = [math.sin(math.radians(val))*TibiaLen for val in roll] 
#                         Flex_moment = arm * _J.Force_Z 
#                         # Normalize
#                         Flex_moment /= _np.mass
#                         _J.Moment_Sagital = Flex_moment

#                         KAM = processing.Compute_KAM(roll,yaw,TibiaLen,_J.Vertical_ACC , _J.Horizontal_ACC)

#                         if Shank_IMU_Number==2: # Left IMU
#                             KAM['M1'] *= -1

#                         KAM['M1'] *= 9.81
#                         KAM['M2'] *= 9.81

#                         _J.Moment_KAM_1 = KAM['M1']
#                         _J.Moment_KAM_2 = KAM['M2']


#                         Xp = _J.time                  
#                         plotData.plot_tibia_angle_leverArm_FlexMoment(Xp,roll,arm,_J.ACCZ_B,Flex_moment,sub_code,sub_code+"_FlexMoment_"+_np.Tasks[inx].name+str(c))
#                         plotData.plot_tibia_angle_leverArm_KAM(Xp,roll,yaw,KAM['arm1'],KAM['arm2'],_J.ACCZ_B, _J.Force_X,KAM['M1'], KAM['M2'],sub_code, sub_code+_np.Tasks[inx].name+str(c))

                    

#                         # Add Shank data 
#                         _np.Tasks[inx].Data[Shank_IMU_Number].Q = _q

#                         _np.Tasks[inx].Data[Shank_IMU_Number].Free_GAX = (np.array(lp_corrected_ACC)[:,0]) * 9.81
#                         _np.Tasks[inx].Data[Shank_IMU_Number].Free_GAY = (np.array(lp_corrected_ACC)[:,1]-1) * 9.81
#                         _np.Tasks[inx].Data[Shank_IMU_Number].Free_GAZ = (np.array(lp_corrected_ACC)[:,2]) * 9.81

#                         _np.Tasks[inx].Data[Shank_IMU_Number].AX = (np.array(lp_corrected_ACC)[:,0])
#                         _np.Tasks[inx].Data[Shank_IMU_Number].AY = (np.array(lp_corrected_ACC)[:,1])
#                         _np.Tasks[inx].Data[Shank_IMU_Number].AZ = (np.array(lp_corrected_ACC)[:,2])


#                 ########################################################################


#                     col = (inx-2)*16               



#                     worksheet.write(0, col, 'Time')
#                     worksheet.write(0, col+1, _np.Tasks[inx].name+str(c)+'_ACCX')
#                     worksheet.write(0, col+2, _np.Tasks[inx].name+str(c)+'_ACCY')
#                     worksheet.write(0, col+3, _np.Tasks[inx].name+str(c)+'_ACCZ')
#                     worksheet.write(0, col+4, _np.Tasks[inx].name+str(c)+'_ACCX_B')
#                     worksheet.write(0, col+5, _np.Tasks[inx].name+str(c)+'_ACCY_B')
#                     worksheet.write(0, col+6, _np.Tasks[inx].name+str(c)+'_ACCZ_B')
#                     worksheet.write(0, col+7, _np.Tasks[inx].name+str(c)+'_Velocity')
#                     worksheet.write(0, col+8, _np.Tasks[inx].name+str(c)+'_Force Vertical')
#                     worksheet.write(0, col+9, _np.Tasks[inx].name+str(c)+'_Force Mediolateral')
#                     worksheet.write(0, col+10, _np.Tasks[inx].name+str(c)+'_Shank_Roll')
#                     worksheet.write(0, col+11, _np.Tasks[inx].name+str(c)+'_Shank_Pitch')
#                     worksheet.write(0, col+12, _np.Tasks[inx].name+str(c)+'_Shank_Yaw')
#                     worksheet.write(0, col+13, _np.Tasks[inx].name+str(c)+'_Knee_Moment_Flex')
#                     worksheet.write(0, col+14, _np.Tasks[inx].name+str(c)+'_KAM_M1')
#                     worksheet.write(0, col+15, _np.Tasks[inx].name+str(c)+'_KAM_M2')


                    
#                     for frame in range(1,len(_J.time)):
#                         worksheet.write(frame, col, _J.time[frame])
#                         worksheet.write(frame, col+1, _J.ACCX[frame])
#                         worksheet.write(frame, col+2, _J.ACCY[frame])
#                         worksheet.write(frame, col+3, _J.ACCZ[frame])
#                         worksheet.write(frame, col+4, _J.ACCX_B[frame])
#                         worksheet.write(frame, col+5, _J.ACCY_B[frame])
#                         worksheet.write(frame, col+6, _J.ACCZ_B[frame])
#                         worksheet.write(frame, col+7, _J.Velocity[frame])
#                         worksheet.write(frame, col+8, _J.ACCZ_B[frame])
#                         worksheet.write(frame, col+9, _J.ACCX_B[frame])
#                         worksheet.write(frame, col+10, roll[frame])
#                         worksheet.write(frame, col+11, pitch[frame])
#                         worksheet.write(frame, col+12, yaw[frame])
#                         worksheet.write(frame, col+13, _J.Moment_Sagital[frame])
#                         worksheet.write(frame, col+14, _J.Moment_KAM_1[frame])
#                         worksheet.write(frame, col+15, _J.Moment_KAM_2[frame])




#                 except Exception as e:
#                     print(e)
                


#                 # write to file
#                 param = [_np.name , _np.Tasks[inx].name , str(_J.JH) , str(_J.FT), str(_J.PeakVerticalForceTakeoff) , str(_J.PeakVerticalForceLanding), str(_J.PeakMediolateralForceTakeoff), str(_J.PeakMediolateralForceLanding) ]
#                 file_output.write(",".join(param)+'\n')

#     workbook.close()





def AddStaticSupine2RawFile(sub_i,name ,_type, Trunk ,Pelvis , Rshank , Lshank):

        # add Static Acc & Mag to Raw File
        add2StrucRaw(sub_i,name,_type,1,"Trunk","Acc","X","SF",0,0,0,0,0,Trunk.AX,1)
        add2StrucRaw(sub_i,name,_type,1,"Trunk","Acc","Y","SF",0,0,0,0,0,Trunk.AY,1)
        add2StrucRaw(sub_i,name,_type,1,"Trunk","Acc","Z","SF",0,0,0,0,0,Trunk.AZ,1)

        add2StrucRaw(sub_i,name,_type,1,"Trunk","GYR","X","SF",0,0,0,0,0,Trunk.GX,1)
        add2StrucRaw(sub_i,name,_type,1,"Trunk","GYR","Y","SF",0,0,0,0,0,Trunk.GY,1)
        add2StrucRaw(sub_i,name,_type,1,"Trunk","GYR","Z","SF",0,0,0,0,0,Trunk.GZ,1)

        add2StrucRaw(sub_i,name,_type,1,"Trunk","Mag","X","SF",0,0,0,0,0,Trunk.MX,1)
        add2StrucRaw(sub_i,name,_type,1,"Trunk","Mag","Y","SF",0,0,0,0,0,Trunk.MY,1)
        add2StrucRaw(sub_i,name,_type,1,"Trunk","Mag","Z","SF",0,0,0,0,0,Trunk.MZ,1)

        #--------------------------
        add2StrucRaw(sub_i,name,_type,1,"Pelvis","Acc","X","SF",0,0,0,0,0,Pelvis.AX,1)
        add2StrucRaw(sub_i,name,_type,1,"Pelvis","Acc","Y","SF",0,0,0,0,0,Pelvis.AY,1)
        add2StrucRaw(sub_i,name,_type,1,"Pelvis","Acc","Z","SF",0,0,0,0,0,Pelvis.AZ,1)

        add2StrucRaw(sub_i,name,_type,1,"Pelvis","GYR","X","SF",0,0,0,0,0,Pelvis.GX,1)
        add2StrucRaw(sub_i,name,_type,1,"Pelvis","GYR","Y","SF",0,0,0,0,0,Pelvis.GY,1)
        add2StrucRaw(sub_i,name,_type,1,"Pelvis","GYR","Z","SF",0,0,0,0,0,Pelvis.GZ,1)

        add2StrucRaw(sub_i,name,_type,1,"Pelvis","Mag","X","SF",0,0,0,0,0,Pelvis.MX,1)
        add2StrucRaw(sub_i,name,_type,1,"Pelvis","Mag","Y","SF",0,0,0,0,0,Pelvis.MY,1)
        add2StrucRaw(sub_i,name,_type,1,"Pelvis","Mag","Z","SF",0,0,0,0,0,Pelvis.MZ,1)

        #--------------------------
        add2StrucRaw(sub_i,name,_type,1,"Rshank","Acc","X","SF",0,0,0,0,0,Rshank.AX,1)
        add2StrucRaw(sub_i,name,_type,1,"Rshank","Acc","Y","SF",0,0,0,0,0,Rshank.AY,1)
        add2StrucRaw(sub_i,name,_type,1,"Rshank","Acc","Z","SF",0,0,0,0,0,Rshank.AZ,1)

        add2StrucRaw(sub_i,name,_type,1,"Rshank","GYR","X","SF",0,0,0,0,0,Rshank.GX,1)
        add2StrucRaw(sub_i,name,_type,1,"Rshank","GYR","Y","SF",0,0,0,0,0,Rshank.GY,1)
        add2StrucRaw(sub_i,name,_type,1,"Rshank","GYR","Z","SF",0,0,0,0,0,Rshank.GZ,1)

        add2StrucRaw(sub_i,name,_type,1,"Rshank","Mag","X","SF",0,0,0,0,0,Rshank.MX,1)
        add2StrucRaw(sub_i,name,_type,1,"Rshank","Mag","Y","SF",0,0,0,0,0,Rshank.MY,1)
        add2StrucRaw(sub_i,name,_type,1,"Rshank","Mag","Z","SF",0,0,0,0,0,Rshank.MZ,1)

        #--------------------------

        _type = list(_type)
        _type[0] = 'L'
        _type = ''.join(_type)


        add2StrucRaw(sub_i,name,_type,1,"Lshank","Acc","X","SF",0,0,0,0,0,Lshank.AX,1)
        add2StrucRaw(sub_i,name,_type,1,"Lshank","Acc","Y","SF",0,0,0,0,0,Lshank.AY,1)
        add2StrucRaw(sub_i,name,_type,1,"Lshank","Acc","Z","SF",0,0,0,0,0,Lshank.AZ,1)

        add2StrucRaw(sub_i,name,_type,1,"Lshank","GYR","X","SF",0,0,0,0,0,Lshank.GX,1)
        add2StrucRaw(sub_i,name,_type,1,"Lshank","GYR","Y","SF",0,0,0,0,0,Lshank.GY,1)
        add2StrucRaw(sub_i,name,_type,1,"Lshank","GYR","Z","SF",0,0,0,0,0,Lshank.GZ,1)       

        
        add2StrucRaw(sub_i,name,_type,1,"Lshank","Mag","X","SF",0,0,0,0,0,Lshank.MX,1)
        add2StrucRaw(sub_i,name,_type,1,"Lshank","Mag","Y","SF",0,0,0,0,0,Lshank.MY,1)
        add2StrucRaw(sub_i,name,_type,1,"Lshank","Mag","Z","SF",0,0,0,0,0,Lshank.MZ,1)




def GetFeatures(IMUs , cutoff , file_path , beta , FC_type):

    

    clear_Struc()

    # Read Events Frame from XLSX File
    Events = read_data.read_xlsx_IMU_Frames(os.path.join(os.getcwd(),'movafaghian/Frames.xlsx'))


    # Read mat files
    all_files = os.listdir(os.path.join(os.getcwd(), file_path )) #'movafaghian/IMU_I'))

    all_files.sort()

    info = read_xlsx_mass('ATH_W_H.xlsx')
    outputs = []
    workbook = xlsxwriter.Workbook('IMU_Extraxted_Data.xlsx')

    # output
    file_output = open("output1.csv", "w")
    param = ["File" , "Jump" , "HJ" , "FT", "Peak Force Vertical - TakeOff" , "Peak Force Vertical - Landing" , "Peak Force Mediolateral - TakeOff", "Peak Force Mediolateral - Landing" ]
    file_output.write(",".join(param)+'\n')




    for subjectNum,f in  enumerate(all_files):


        print("#################   " + str(subjectNum))

        [sub_code,pre_post] = f.split('_')
        pre_post = pre_post.split('.')[0] 
          

        subject_info = info[sub_code]
        _np = player(f,20,subject_info['W'],subject_info['H'],'M',9999,0)   

        ev_index = Events['Name'].index(f.split('.')[0])
        _np.Tasks = read_mat( os.path.join(os.getcwd(),file_path,f),Events,ev_index , False)


        print("#########################################################")
        print("subject File :" +  _np.name)
        print("Tasks :" +  str(len(_np.Tasks)))
        for i in range(len(_np.Tasks)):           
            print("Tasks :" + _np.Tasks[i].name + " / " + str(len(_np.Tasks[i].Data)))

        print("#########################################################")



        # Tasks
        type_tmp = ''
        c = 0
        worksheet = workbook.add_worksheet(_np.name)   
        col=0        


        Trunk_StaticEuler = []
        Pelvis_StaticEuler = []
        Shank_StaticEuler = []


        sub_i =  _np.name.split('_')[0][-2:]
        # AddStaticSupine2RawFile(sub_i,_np.name,'R'+ _np.Tasks[0].name,_np.Tasks[0].Data[5] , _np.Tasks[0].Data[4] , _np.Tasks[0].Data[3] , _np.Tasks[0].Data[2])
        # AddStaticSupine2RawFile(sub_i,_np.name,'R'+ _np.Tasks[1].name,_np.Tasks[1].Data[5] , _np.Tasks[1].Data[4] , _np.Tasks[1].Data[3] , _np.Tasks[1].Data[2])



        for inx in range(2,len(_np.Tasks)):

            EVENTs = [0,0,0,0,0,0]
            

            if _np.Tasks[inx].name=='CMJ1' or _np.Tasks[inx].name=='LSJ' or _np.Tasks[inx].name=='RSJ' or _np.Tasks[inx].name=='LDJ1' or _np.Tasks[inx].name=='RDJ1':

                
                Main_Acc_X=[]
                Main_Acc_Z=[]



                if type_tmp != _np.Tasks[inx].name:
                    type_tmp =  _np.Tasks[inx].name
                    c = 1
                else:
                    c += 1

                # print(f)
                # print(_np.Tasks[inx].name + str(c))            

               
                for IMU_Number in IMUs:  

                    if IMU_Number==4:
                        SensorName = "Pelvis"
                    if IMU_Number==5:
                        SensorName = "Trunk"
                    if IMU_Number==3:
                        SensorName = "RShank"
                    if IMU_Number==2:
                        SensorName = "LShank"  


                    if _np.Tasks[0].Data[IMU_Number].AY[0] < 0:
                        _np.Tasks[0].Data[IMU_Number].AY *=  -1
                        _np.Tasks[0].Data[IMU_Number].AX *=  -1

                        _np.Tasks[0].Data[IMU_Number].GY *=  -1
                        _np.Tasks[0].Data[IMU_Number].GX *=  -1

                        _np.Tasks[1].Data[IMU_Number].AY *=  -1
                        _np.Tasks[1].Data[IMU_Number].AX *=  -1

                        _np.Tasks[1].Data[IMU_Number].GY *=  -1
                        _np.Tasks[1].Data[IMU_Number].GX *=  -1


                    if _np.Tasks[inx].Data[IMU_Number].AY[0] < 0:
                        _np.Tasks[inx].Data[IMU_Number].AY *=  -1
                        _np.Tasks[inx].Data[IMU_Number].AX *=  -1

                        _np.Tasks[inx].Data[IMU_Number].GY *=  -1
                        _np.Tasks[inx].Data[IMU_Number].GX *=  -1



                    

                    ACC_standing = np.array([_np.Tasks[0].Data[IMU_Number].AX ,_np.Tasks[0].Data[IMU_Number].AY, _np.Tasks[0].Data[IMU_Number].AZ ])
                    GYRO_standing = np.array([_np.Tasks[0].Data[IMU_Number].GX ,_np.Tasks[0].Data[IMU_Number].GY, _np.Tasks[0].Data[IMU_Number].GZ ])
                    ACC_laying = np.array([_np.Tasks[1].Data[IMU_Number].AX ,_np.Tasks[1].Data[IMU_Number].AY, _np.Tasks[1].Data[IMU_Number].AZ ])
                    GYRO_laying = np.array([_np.Tasks[1].Data[IMU_Number].GX ,_np.Tasks[1].Data[IMU_Number].GY, _np.Tasks[1].Data[IMU_Number].GZ ])
                    Rotation_results = processing.Compute_Sensor2Body_rotation(ACC_standing,ACC_laying)

                    Standing_corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_standing)
                    Standing_corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_standing)

                    Standing_corrected_ACC = processing.RotateX90(Standing_corrected_ACC)
                    Standing_corrected_GYRO = processing.RotateX90(Standing_corrected_GYRO)

                    laying_corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_laying)
                    laying_corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_laying)

                    laying_corrected_ACC = processing.RotateX90(laying_corrected_ACC)
                    laying_corrected_GYRO = processing.RotateX90(laying_corrected_GYRO)

              
                    ACC_temp = np.array([_np.Tasks[inx].Data[IMU_Number].AX ,_np.Tasks[inx].Data[IMU_Number].AY, _np.Tasks[inx].Data[IMU_Number].AZ ])
                    GYRO_temp = np.array([_np.Tasks[inx].Data[IMU_Number].GX ,_np.Tasks[inx].Data[IMU_Number].GY, _np.Tasks[inx].Data[IMU_Number].GZ ])
                    MAG_temp = np.array([_np.Tasks[inx].Data[IMU_Number].MX ,_np.Tasks[inx].Data[IMU_Number].MY, _np.Tasks[inx].Data[IMU_Number].MZ ])

                   
                    _qStatic = computeOrientation(Standing_corrected_ACC,Standing_corrected_GYRO,dt_ms/1000 , beta , 0, 10000)

                    # V = [0,1,0]
                    # R = np.array([[1,0,0] , [0,0,-1] , [0,1,0]])
                    a = processing.RotationMatrix2Euler(Rotation_results['R1'])
                    # V1 = processing.RotateM(R , V)
                    # V2 = processing.RotateM(np.transpose(R) , V1)


                    Ra = processing.RotationMatrix2Euler(np.transpose(Rotation_results['R1']))


                    b = processing.RotationMatrix2Euler(Rotation_results['R2'])


                    corrected_ACC = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],ACC_temp)
                    corrected_GYRO = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],GYRO_temp)
                    corrected_MAG = processing.Apply_Sensor2Body(Rotation_results['R1'],Rotation_results['R2'],MAG_temp)


                    if FC_type=='FCM':                    
                        if ("Shank" in SensorName): 
                            R1_prim = Rotation_results['R1']
                            R1_prim[0,0]=1
                            R1_prim[0,1]=0
                            R1_prim[0,2]=0
                            R1_prim[1,0]=0
                            R1_prim[2,0]=0                            
                            corrected_ACC = processing.RotateM(np.transpose(R1_prim),corrected_ACC)
                            corrected_GYRO = processing.RotateM(np.transpose(R1_prim),corrected_GYRO)

                 
                    lp_corrected_ACC =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_ACC),cutoff)
                    lp_corrected_GYRO =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_GYRO),cutoff)
                    lp_corrected_MAG =  processing.low_pass_filter_3axis(dt_ms,np.array(corrected_MAG),cutoff)


                    
                    # if SensorName=='Pelvis':
                    #     plotData.plotXYZ2(lp_corrected_ACC)
                    #     input('TES;;;;')


                    lp_corrected_ACC = processing.RotateX90(lp_corrected_ACC)
                    lp_corrected_GYRO = processing.RotateX90(lp_corrected_GYRO)   
                    lp_corrected_MAG = processing.RotateX90(lp_corrected_MAG)                


                    t1 =  time.time()



                    startPoint = findStartPointMadgwicks(lp_corrected_GYRO, 100, 2)    
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^")
                    print(startPoint) 

                    if (startPoint > len(lp_corrected_GYRO)/5):
                        startPoint = 0

                    [lp_corrected_ACC , lp_corrected_GYRO] =  SignalModifier(lp_corrected_ACC,lp_corrected_GYRO , startPoint)

                    lp_corrected_ACC = np.transpose(lp_corrected_ACC)
                    lp_corrected_GYRO = np.transpose(lp_corrected_GYRO)

                    # Compute Orientation   
                                     
                    _q = computeOrientation(lp_corrected_ACC,lp_corrected_GYRO,dt_ms/1000 , beta , startIndex=startPoint , itr= 10000)
                    _np.Tasks[inx].Data[IMU_Number].Q = _q

                    # View3D.showIMUOrientation([_np.Tasks[inx].Data[IMU_Number]])

                    # Q2E
                    Euler = np.array([i.get_euler() for i in _q ])                                               
                    roll = [i.roll for i in Euler]
                    pitch = [i.pitch for i in Euler]
                    yaw = [i.yaw for i in Euler]

                    t2 = time.time()
                    print("timing ::", t2-t1 , " Secondes" )


                    # Body 2 Global
                    Global_ACC = np.array(processing.Compute_Body2G_rotation(lp_corrected_ACC,_q))
                    # Global_ACC2 = np.array(processing.Compute_Body2Body_rotation(_q,avgStandingGlobalFrameArry, lp_corrected_ACC))
                    # Global_ACC[:,2] *= -1

                    Global_GYRO = np.array(processing.Compute_Body2G_rotation(lp_corrected_GYRO,_q))
                    # Global_GYRO2 = np.array(processing.Compute_Body2Body_rotation(_q,avgStandingGlobalFrameArry, lp_corrected_GYRO))


                    _np.Tasks[inx].Data[IMU_Number].Free_GAX = (np.array(lp_corrected_ACC)[:,0]) * 9.81
                    _np.Tasks[inx].Data[IMU_Number].Free_GAY = (np.array(lp_corrected_ACC)[:,1]-1) * 9.81
                    _np.Tasks[inx].Data[IMU_Number].Free_GAZ = (np.array(lp_corrected_ACC)[:,2]) * 9.81

                    _np.Tasks[inx].Data[IMU_Number].AX = (np.array(lp_corrected_ACC)[:,0])
                    _np.Tasks[inx].Data[IMU_Number].AY = (np.array(lp_corrected_ACC)[:,1])
                    _np.Tasks[inx].Data[IMU_Number].AZ = (np.array(lp_corrected_ACC)[:,2])                   

                    vertical_V = processing.integral2((Global_ACC[:,2]-1) * 9.81 ,_np.Tasks[inx].Data[IMU_Number].time)
                    try:
                        vertical_V_C = processing.Velocity_correction(vertical_V, (Global_ACC[:,2]-1) * 9.81 ,1000/dt_ms)
                    except:
                        print('...')
                        vertical_V_C = vertical_V               
                    


                    # Create Jump            
                    _J = JUMP(_np.Tasks[inx].name,mass=_np.mass , sub_height=_np.height)

                    _J.Velocity = vertical_V_C
                    _J.ACCZ_B = processing.low_pass_filter(cutoff,dt_ms, np.array(_np.Tasks[inx].Data[IMU_Number].AZ))
                    _J.ACCX_B = processing.low_pass_filter(cutoff,dt_ms, np.array(_np.Tasks[inx].Data[IMU_Number].AX))
                    _J.ACCY_B = processing.low_pass_filter(cutoff,dt_ms, np.array(_np.Tasks[inx].Data[IMU_Number].AY))

                    _J.GYROZ_B = processing.low_pass_filter(cutoff,dt_ms,np.array(lp_corrected_GYRO)[:,2])
                    _J.GYROX_B = processing.low_pass_filter(cutoff,dt_ms,np.array(lp_corrected_GYRO)[:,0])
                    _J.GYROY_B = processing.low_pass_filter(cutoff,dt_ms,np.array(lp_corrected_GYRO)[:,1])


                    _J.ACCZ_G = processing.low_pass_filter(cutoff,dt_ms, np.array(Global_ACC)[:,2])
                    # _J.ACCZ_G -= 1
                    _J.ACCX_G = processing.low_pass_filter(cutoff,dt_ms, np.array(Global_ACC)[:,0])
                    _J.ACCY_G = processing.low_pass_filter(cutoff,dt_ms, np.array(Global_ACC)[:,1])

                    # _J.ACCZ_G2 = processing.low_pass_filter(cutoff,dt_ms, np.array(Global_ACC2)[:,2])
                    # _J.ACCX_G2 = processing.low_pass_filter(cutoff,dt_ms, np.array(Global_ACC2)[:,0])
                    # _J.ACCY_G2 = processing.low_pass_filter(cutoff,dt_ms, np.array(Global_ACC2)[:,1])

                    _J.GYROZ_G = processing.low_pass_filter(cutoff,dt_ms,np.array(Global_GYRO)[:,2])
                    _J.GYROX_G = processing.low_pass_filter(cutoff,dt_ms,np.array(Global_GYRO)[:,0])
                    _J.GYROY_G = processing.low_pass_filter(cutoff,dt_ms,np.array(Global_GYRO)[:,1])

                    # _J.GYROZ_G2 = processing.low_pass_filter(cutoff,dt_ms,np.array(Global_GYRO2)[:,2])
                    # _J.GYROX_G2 = processing.low_pass_filter(cutoff,dt_ms,np.array(Global_GYRO2)[:,0])
                    # _J.GYROY_G2 = processing.low_pass_filter(cutoff,dt_ms,np.array(Global_GYRO2)[:,1])


                    _J.Q = _q


                    _J.ACCZ =  ACC_temp[2]
                    _J.ACCX =  ACC_temp[0]
                    _J.ACCY =  ACC_temp[1]                    

                    _J.GYROZ =  GYRO_temp[2]
                    _J.GYROX =  GYRO_temp[0]
                    _J.GYROY =  GYRO_temp[1]  


                    _J.MAGX = MAG_temp[0]
                    _J.MAGY = MAG_temp[1]
                    _J.MAGZ = MAG_temp[2]

         

                    _J.time = _np.Tasks[inx].Data[IMU_Number].time
                    _J.Force_X = processing.low_pass_filter(cutoff,dt_ms, (_J.ACCX_G)*9.81 * _np.mass)
                    _J.Force_Z = processing.low_pass_filter(cutoff,dt_ms, (_J.ACCZ_G)* 9.81 * _np.mass)
                    _J.Force_Y = processing.low_pass_filter(cutoff,dt_ms, (_J.ACCY_G)*9.81 * _np.mass)    

                    _J.roll = np.array(roll)
                    _J.pitch = np.array(pitch)
                    _J.yaw =  np.array(yaw)
                  




                    ############################ WithOut Correction Static and Supin                      
                    _Staticq = computeOrientation(np.transpose(ACC_standing),np.transpose(GYRO_standing),dt_ms/1000 , 0.1, 0,10000 )                    
                    # Q2E
                    _StaticEuler = np.array([i.get_euler() for i in _Staticq ])                                               
                    _StaticRoll = [i.roll for i in _StaticEuler]
                    _StaticPitch = [i.pitch for i in _StaticEuler]
                    _StaticYaw = [i.yaw for i in _StaticEuler]

                    if 'Pelvis' in SensorName:
                        Pelvis_StaticEuler = [_StaticRoll,_StaticPitch , _StaticYaw]
                    elif 'Trunk' in SensorName:
                        Trunk_StaticEuler = [_StaticRoll,_StaticPitch , _StaticYaw]
                    elif 'Shank' in SensorName:
                        Shank_StaticEuler = [_StaticRoll,_StaticPitch , _StaticYaw]


                    # plotData.plot(ACC_standing)
                    # plotData.plot(GYRO_standing)
                    # plotData.plot([_StaticRoll , _StaticPitch , _StaticYaw])
                    # w = np.array([item.w for item in _Staticq])
                    # x = np.array([item.x for item in _Staticq])
                    # y = np.array([item.y for item in _Staticq])
                    # z = np.array([item.z for item in _Staticq])
                    # # plotData.plot([w , x , y , z])

                    # # input("Stop!!")
                    
                    #######################################################################################################                   

                 
                    data_id = sub_code+'_'+pre_post+'_'+str(c)+'_'+type_tmp+'_'+SensorName
                

                    if (SensorName == "LShank" and type_tmp != "RSJ"):
                            print(SensorName)
                            _J.Main_Acc_X = Main_Acc_X
                            _J.Main_Acc_Z = Main_Acc_Z
                            _J.run("Shank",type_tmp+str(c),sub_code,pre_post)                       
                            outputs.append(_J.ALLFeatures)

                    if (SensorName == "RShank" and type_tmp != "LSJ"):
                        print(SensorName)
                        _J.Main_Acc_X = Main_Acc_X
                        _J.Main_Acc_Z = Main_Acc_Z
                        _J.run("Shank",type_tmp+str(c),sub_code,pre_post)
                        outputs.append(_J.ALLFeatures)  
                    
                    if SensorName == "Pelvis" or SensorName == "Trunk":
                        print(SensorName)
                        _J.run(SensorName,type_tmp+str(c),sub_code,pre_post)

                        if SensorName == "Trunk":                                              
                            Main_Acc_X = _J.ACCX_B
                            Main_Acc_Z = _J.ACCZ_G

                        outputs.append(_J.ALLFeatures)                                        
                    

                    #Write to XLSX file

                    if (SensorName == "RShank" and type_tmp != "RSJ") or (SensorName == "LShank" and type_tmp != "LSJ"):
                        print('')
                    else:         
                    
                    
                        sub_i =  _np.name.split('_')[0][-2:]

                        quality = 1
                        if (data_id in OmitList):
                            quality = 0



                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"Acc","X","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCX,quality)
                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"Acc","Y","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCY,quality)
                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"Acc","Z","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCZ,quality)

                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"GYR","X","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROX,quality)
                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"GYR","Y","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROY,quality)
                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"GYR","Z","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROZ,quality)

                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"Mag","X","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.MAGX,quality)
                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"Mag","Y","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.MAGY,quality)
                        # add2StrucRaw(sub_i,_np.name,type_tmp,c,SensorName,"Mag","Z","SF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.MAGZ,quality)

                        

                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Acc","X","AF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCX_B,quality)  
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Acc","Y","AF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCY_B,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Acc","Z","AF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCZ_B,quality)                       

                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Acc","X","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCX_G,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Acc","Y","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCY_G,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Acc","Z","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.ACCZ_G,quality)      

                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"GYR","X","AF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROX_B,quality)  
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"GYR","Y","AF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROY_B,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"GYR","Z","AF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROZ_B,quality)                       

                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"GYR","X","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROX_G,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"GYR","Y","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROY_G,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"GYR","Z","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.GYROZ_G,quality)    

                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Eul","X","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.roll,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Eul","Y","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.pitch,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Eul","Z","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.yaw,quality)         

                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Eul2","X","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.roll2,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Eul2","Y","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.pitch2,quality)                       
                        add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"Eul2","Z","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.yaw2,quality) 

                        if len(_J.Moment_KAM_1):
                            add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"M1","X","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.Moment_KAM_1,quality)
                            add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"M2","X","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.Moment_KAM_2,quality)  
                            add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"KAM","X","GF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.Moment2_KAM,quality)  




                        # if SensorName == "RShank" or SensorName == "LShank" :
                        #     try:
                        #         add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"KFM","X","LF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.Moment2_Flex,quality)                       
                        #         add2Struc(sub_i,_np.name,type_tmp,c,SensorName,"KAM","Y","LF",_J.A,_J.C,_J.D,_J.F,_J.I,_J.Moment2_KAM,quality)                       
                        #     except:
                        #         print('...')





                    # worksheet.write(0, col, 'Time')
                    # worksheet.write(0, col+1, SensorName+ _np.Tasks[inx].name+str(c)+'_ACCX_B')
                    # worksheet.write(0, col+2, SensorName+ _np.Tasks[inx].name+str(c)+'_ACCY_B')
                    # worksheet.write(0, col+3, SensorName+ _np.Tasks[inx].name+str(c)+'_ACCZ_B')
                    # worksheet.write(0, col+4, SensorName+ _np.Tasks[inx].name+str(c)+'_ACCX_G')
                    # worksheet.write(0, col+5, SensorName+ _np.Tasks[inx].name+str(c)+'_ACCY_G')
                    # worksheet.write(0, col+6, SensorName+ _np.Tasks[inx].name+str(c)+'_ACCZ_G')
                    # worksheet.write(0, col+7, SensorName+ _np.Tasks[inx].name+str(c)+'_GYRX_B')
                    # worksheet.write(0, col+8, SensorName+ _np.Tasks[inx].name+str(c)+'_GYRY_B')
                    # worksheet.write(0, col+9, SensorName+ _np.Tasks[inx].name+str(c)+'_GYRZ_B')
                    # worksheet.write(0, col+10, SensorName+ _np.Tasks[inx].name+str(c)+'_GYRX_G')
                    # worksheet.write(0, col+11, SensorName+ _np.Tasks[inx].name+str(c)+'_GYRY_G')
                    # worksheet.write(0, col+12, SensorName+ _np.Tasks[inx].name+str(c)+'_GYRZ_G')

                    # worksheet.write(0, col+13, SensorName+ _np.Tasks[inx].name+str(c)+'_Roll_B')
                    # worksheet.write(0, col+14, SensorName+ _np.Tasks[inx].name+str(c)+'_Pitch_B')
                    # worksheet.write(0, col+15, SensorName+ _np.Tasks[inx].name+str(c)+'_Yaw_B')

                    # worksheet.write(0, col+16, SensorName+ _np.Tasks[inx].name+str(c)+'_M1')
                    # worksheet.write(0, col+17, SensorName+ _np.Tasks[inx].name+str(c)+'_M2')
                    # worksheet.write(0, col+18, SensorName+ _np.Tasks[inx].name+str(c)+'_M')
                    # worksheet.write(0, col+19, SensorName+ _np.Tasks[inx].name+str(c)+'_KFM')
                    # worksheet.write(0, col+20, SensorName+ _np.Tasks[inx].name+str(c)+'_KAM')

                    # worksheet.write(0, col+21, SensorName+ _np.Tasks[inx].name+str(c)+'_Q0')
                    # worksheet.write(0, col+22, SensorName+ _np.Tasks[inx].name+str(c)+'_Q1')
                    # worksheet.write(0, col+23, SensorName+ _np.Tasks[inx].name+str(c)+'_Q2')
                    # worksheet.write(0, col+24, SensorName+ _np.Tasks[inx].name+str(c)+'_Q3')



        
                    # for frame in range(1,len(_J.time)):
                    #     worksheet.write(frame, col, _J.time[frame])
                    #     worksheet.write(frame, col+1, _J.ACCX_B[frame])
                    #     worksheet.write(frame, col+2, _J.ACCY_B[frame])
                    #     worksheet.write(frame, col+3, _J.ACCZ_B[frame])
                    #     worksheet.write(frame, col+4, _J.ACCX_G[frame])
                    #     worksheet.write(frame, col+5, _J.ACCY_G[frame])                        
                    #     worksheet.write(frame, col+6, _J.ACCZ_G[frame])

                    #     worksheet.write(frame, col+7, _J.GYROX_B[frame])
                    #     worksheet.write(frame, col+8, _J.GYROY_B[frame])
                    #     worksheet.write(frame, col+9, _J.GYROZ_B[frame])
                    #     worksheet.write(frame, col+10, _J.GYROX_G[frame])
                    #     worksheet.write(frame, col+11, _J.GYROY_G[frame])
                    #     worksheet.write(frame, col+12, _J.GYROZ_G[frame])


                    #     worksheet.write(frame, col+13, _J.roll[frame])
                    #     worksheet.write(frame, col+14, _J.pitch[frame])
                    #     worksheet.write(frame, col+15, _J.yaw[frame])

                    #     if (len(_J.Moment_KAM_1)>frame):
                    #         worksheet.write(frame, col+16, _J.Moment_KAM_1[frame])
                    #         worksheet.write(frame, col+17, _J.Moment_KAM_2[frame])
                    #         worksheet.write(frame, col+18, _J.Moment_KAM_2[frame] + _J.Moment_KAM_1[frame])
                    #         worksheet.write(frame, col+19, _J.Moment2_Flex[frame])
                    #         worksheet.write(frame, col+20, _J.Moment2_KAM[frame])
                    #     else:
                    #         worksheet.write(frame, col+16, 0)
                    #         worksheet.write(frame, col+17, 0)
                    #         worksheet.write(frame, col+18, 0)
                    #         worksheet.write(frame, col+19, 0)
                    #         worksheet.write(frame, col+20, 0)

                    #     worksheet.write(frame, col+21, _J.Q[frame].w)
                    #     worksheet.write(frame, col+22, _J.Q[frame].x)
                    #     worksheet.write(frame, col+23, _J.Q[frame].y)
                    #     worksheet.write(frame, col+24, _J.Q[frame].z)

                    # col += 25



                    # if SensorName=="Pelvis":                         
                        
                    #     axes[c-1,0].plot(_J.time,_J.ACCX_B, label = "X" , linewidth=1)
                    #     axes[c-1,0].plot(_J.time,_J.ACCY_B, label = "Y", linewidth=1)
                    #     axes[c-1,0].plot(_J.time,_J.ACCZ_B, label = "Z", linewidth=1)
                    #     axes[0,0].set_title('Acc Body Frame')
                    #     axes[c-1,0].set_ylabel('Trial '+str(c))
                        
                    #     axes[c-1,1].plot(_J.time,_J.ACCX_G,label = "X", linewidth=1)
                    #     axes[c-1,1].plot(_J.time,_J.ACCY_G,label = "Y", linewidth=1)
                    #     axes[c-1,1].plot(_J.time,_J.ACCZ_G,label = "Z", linewidth=1)
                    #     axes[0,1].set_title('Acc Global Frame')
                        

                        
                    #     axes[c-1,2].plot(_J.time,_J.GYROX_B,label = "X", linewidth=1)
                    #     axes[c-1,2].plot(_J.time,_J.GYROY_B,label = "Y", linewidth=1)
                    #     axes[c-1,2].plot(_J.time,_J.GYROZ_B,label = "Z", linewidth=1)
                    #     axes[0,2].set_title('GYR Body Frame')
                        
                        
                    #     axes[c-1,3].plot(_J.time,_J.GYROX_G,label = "X", linewidth=1)
                    #     axes[c-1,3].plot(_J.time,_J.GYROY_G,label = "Y", linewidth=1)
                    #     axes[c-1,3].plot(_J.time,_J.GYROZ_G,label = "Z", linewidth=1)
                    #     axes[0,3].set_title('GYR Global Frame')
                        
                        
                    #     axes[c-1,4].plot(_J.time,_J.roll,label = "X", linewidth=1)
                    #     axes[c-1,4].plot(_J.time,_J.pitch ,label = "Y", linewidth=1)
                    #     axes[c-1,4].plot(_J.time,_J.yaw,label = "Z", linewidth=1)
                    #     axes[0,4].set_title('Euler')
            
                    #     if c==3:
                    #         fig.suptitle(SensorName + " " +type_tmp)
                    #         fig.savefig("plot/"+_np.name+"_Data plot "+SensorName + " " +type_tmp +".png")
                    #         fig.legend()
                    #         plt.close()
                    #         fig, axes = plt.subplots(3, 5, figsize = (15, 8), tight_layout=True)


                    # if SensorName=="Trunk":                         
                        
                        # Trunk_axes[c-1,0].plot(_J.time,_J.ACCX_B, label = "X" , linewidth=1)
                        # Trunk_axes[c-1,0].plot(_J.time,_J.ACCY_B, label = "Y", linewidth=1)
                        # Trunk_axes[c-1,0].plot(_J.time,_J.ACCZ_B, label = "Z", linewidth=1)
                        # Trunk_axes[0,0].set_title('Acc Body Frame')
                        # Trunk_axes[c-1,0].set_ylabel('Trial '+str(c))
                        
                        # Trunk_axes[c-1,1].plot(_J.time,_J.ACCX_G,label = "X", linewidth=1)
                        # Trunk_axes[c-1,1].plot(_J.time,_J.ACCY_G,label = "Y", linewidth=1)
                        # Trunk_axes[c-1,1].plot(_J.time,_J.ACCZ_G,label = "Z", linewidth=1)
                        # Trunk_axes[0,1].set_title('Acc Global Frame')
                        

                        
                        # Trunk_axes[c-1,2].plot(_J.time,_J.GYROX_B,label = "X", linewidth=1)
                        # Trunk_axes[c-1,2].plot(_J.time,_J.GYROY_B,label = "Y", linewidth=1)
                        # Trunk_axes[c-1,2].plot(_J.time,_J.GYROZ_B,label = "Z", linewidth=1)
                        # Trunk_axes[0,2].set_title('GYR Body Frame')
                        
                        
                        # Trunk_axes[c-1,3].plot(_J.time,_J.GYROX_G,label = "X", linewidth=1)
                        # Trunk_axes[c-1,3].plot(_J.time,_J.GYROY_G,label = "Y", linewidth=1)
                        # Trunk_axes[c-1,3].plot(_J.time,_J.GYROZ_G,label = "Z", linewidth=1)
                        # Trunk_axes[0,3].set_title('GYR Global Frame')
                        
                        
                        # Trunk_axes[c-1,4].plot(_J.time,_J.roll,label = "X", linewidth=1)
                        # Trunk_axes[c-1,4].plot(_J.time,_J.pitch ,label = "Y", linewidth=1)
                        # Trunk_axes[c-1,4].plot(_J.time,_J.yaw,label = "Z", linewidth=1)
                        # Trunk_axes[0,4].set_title('Euler')
            
                        # if c==3:
                        #     Trunk_fig.suptitle(SensorName + " " +type_tmp)
                        #     Trunk_fig.savefig("plot/"+_np.name+"_Data plot "+SensorName + " " +type_tmp +".png")
                        #     Trunk_fig.legend()
                        #     plt.close()
                        #     Trunk_fig, Trunk_axes = plt.subplots(3, 5, figsize = (15, 8), tight_layout=True)


                    # if SensorName=="LShank" or SensorName=="RShank":                         
                        
                    #     Shank_axes[c-1,0].plot(_J.time,_J.ACCX_B, label = "X" , linewidth=1)
                    #     Shank_axes[c-1,0].plot(_J.time,_J.ACCY_B, label = "Y", linewidth=1)
                    #     Shank_axes[c-1,0].plot(_J.time,_J.ACCZ_B, label = "Z", linewidth=1)
                    #     Shank_axes[0,0].set_title('Acc Body Frame')
                    #     Shank_axes[c-1,0].set_ylabel('Trial '+str(c))
                        
                    #     Shank_axes[c-1,1].plot(_J.time,_J.ACCX_G,label = "X", linewidth=1)
                    #     Shank_axes[c-1,1].plot(_J.time,_J.ACCY_G,label = "Y", linewidth=1)
                    #     Shank_axes[c-1,1].plot(_J.time,_J.ACCZ_G,label = "Z", linewidth=1)
                    #     Shank_axes[0,1].set_title('Acc Global Frame')
                        

                        
                    #     Shank_axes[c-1,2].plot(_J.time,_J.GYROX_B,label = "X", linewidth=1)
                    #     Shank_axes[c-1,2].plot(_J.time,_J.GYROY_B,label = "Y", linewidth=1)
                    #     Shank_axes[c-1,2].plot(_J.time,_J.GYROZ_B,label = "Z", linewidth=1)
                    #     Shank_axes[0,2].set_title('GYR Body Frame')
                        
                        
                    #     Shank_axes[c-1,3].plot(_J.time,_J.GYROX_G,label = "X", linewidth=1)
                    #     Shank_axes[c-1,3].plot(_J.time,_J.GYROY_G,label = "Y", linewidth=1)
                    #     Shank_axes[c-1,3].plot(_J.time,_J.GYROZ_G,label = "Z", linewidth=1)
                    #     Shank_axes[0,3].set_title('GYR Global Frame')
                        
                        
                    #     Shank_axes[c-1,4].plot(_J.time,_J.roll,label = "X", linewidth=1)
                    #     Shank_axes[c-1,4].plot(_J.time,_J.pitch ,label = "Y", linewidth=1)
                    #     Shank_axes[c-1,4].plot(_J.time,_J.yaw,label = "Z", linewidth=1)
                    #     Shank_axes[0,4].set_title('Euler')
            
                    #     if c==3:
                    #         Shank_fig.suptitle(SensorName + " " +type_tmp)
                    #         Shank_fig.savefig("plot/"+_np.name+"_Data plot "+SensorName + " " +type_tmp +".png")
                    #         Shank_fig.legend()
                    #         plt.close()
                    #         Shank_fig, Trunk_axes = plt.subplots(3, 5, figsize = (15, 8), tight_layout=True)


            # plt.close()




        # add Static Euler Without correectin
        add2Struc(sub_i,_np.name,'RStatic',1,"Trunk","Eul","X","SF",0,0,0,0,0,Trunk_StaticEuler[0],1)
        add2Struc(sub_i,_np.name,'RStatic',1,"Trunk","Eul","Y","SF",0,0,0,0,0,Trunk_StaticEuler[1],1)
        add2Struc(sub_i,_np.name,'RStatic',1,"Trunk","Eul","Z","SF",0,0,0,0,0,Trunk_StaticEuler[2],1)

        add2Struc(sub_i,_np.name,'RStatic',1,"Pelvis","Eul","X","SF",0,0,0,0,0,Pelvis_StaticEuler[0],1)
        add2Struc(sub_i,_np.name,'RStatic',1,"Pelvis","Eul","Y","SF",0,0,0,0,0,Pelvis_StaticEuler[1],1)
        add2Struc(sub_i,_np.name,'RStatic',1,"Pelvis","Eul","Z","SF",0,0,0,0,0,Pelvis_StaticEuler[2],1)

        add2Struc(sub_i,_np.name,'RStatic',1,"Shank","Eul","X","SF",0,0,0,0,0,Shank_StaticEuler[0],1)
        add2Struc(sub_i,_np.name,'RStatic',1,"Shank","Eul","Y","SF",0,0,0,0,0,Shank_StaticEuler[1],1)
        add2Struc(sub_i,_np.name,'RStatic',1,"Shank","Eul","Z","SF",0,0,0,0,0,Shank_StaticEuler[2],1)











    workbook.close()

    matDf = pd.DataFrame(matStructure)
    mdic = {"DataStruc" : matDf.to_numpy()}
    file_name = 'matlab_DataStructure_' + FC_type + '_LP'+ str(cutoff) + '_Mag'+str(beta) + '.mat'
    savemat(file_name,mdic)




    # matDf = pd.DataFrame(RaWmatStructure)
    # mdic = {"DataStruc" : matDf.to_numpy()}

    # savemat("matlab_DataStructure_Raw.mat",mdic)


    # input("Data Structre!!!!!")

    final_output = []

    # for i in range(len(outputs)):
    #     if len(outputs[i]) != len(outputs[0]):
    #        print(len(outputs[i]))
    #     #    print(outputs[i])
    #     #    input("STOP" + str(i))
    #     else : 
    #         final_output.append(outputs[i])


    return np.array(final_output)



OmitList =  ["ATH01_Pre_3_RSJ_RShank",
             "ATH24_Pre_1_RSJ_RShank",
             "ATH24_Pre_2_RSJ_Trunk",
             "ATH24_Pre_2_RSJ_RShank",
             "ATH24_Pre_3_RSJ_RShank",
             "ATH03_Post_1_LSJ_Pelvis",
             "ATH03_Post_1_LSJ_LShank",
             "ATH14_Post_3_RSJ_Pelvis",
             "ATH14_Post_1_RSJ_RShank",
             "ATH14_Post_3_RSJ_Trunk",
             ]






#DataStructure
matStructure = {
    "Num" : [],
    "Stat" : [],
    "Side" : [],
    "Task" : [],
    "Trial" : [],
    "Sen" : [],
    "Sig" : [],
    "DOF" : [],
    "Frame" : [],
    "A" : [],
    "C" : [],
    "D" : [],
    "F" : [],
    "I" : [],
    "Data":[],
    "Quality" : []
}


#DataStructure
RaWmatStructure = {
    "Num" : [],
    "Stat" : [],
    "Side" : [],
    "Task" : [],
    "Trial" : [],
    "Sen" : [],
    "Sig" : [],
    "DOF" : [],
    "A" : [],
    "C" : [],
    "D" : [],
    "F" : [],
    "I" : [],
    "Frame" : [],
    "Data":[],
    "Quality" : []
}


EVENTs = [0,0,0,0,0,0]


def clear_Struc():
    #DataStructure

    matStructure['Num']= []
    matStructure['Stat'] = []
    matStructure['Side'] = []
    matStructure['Task'] = []
    matStructure['Trial'] = []
    matStructure['Sen'] = []
    matStructure['Sig'] = []
    matStructure['DOF'] = []
    matStructure['Frame'] = []
    matStructure['A'] = []
    matStructure['C'] = []
    matStructure['D'] = []
    matStructure['F'] = []
    matStructure['I'] = []
    matStructure['Data'] = []
    matStructure['Quality'] = []



    EVENTs[0] = 0
    EVENTs[1] = 0
    EVENTs[2] = 0
    EVENTs[3] = 0
    EVENTs[4] = 0

    




def add2Struc(subjectNum,name,type_tmp,c,SensorName,Sig,DOF,Frame,A,C,D,F,I,Data,quality):

    if 'R' in type_tmp or 'L' in type_tmp:
        side = type_tmp[0]
        type_tmp = type_tmp[1:]


    if 'R' in SensorName or 'L' in SensorName:
        SensorName = SensorName[1:]

    matStructure['Num'].append(int(subjectNum))
    if 'Pre' in name:
        matStructure['Stat'].append("Pre")
    else :
        matStructure['Stat'].append("Post")

    matStructure['Side'].append(side)

    matStructure['Task'].append(type_tmp)
    matStructure['Trial'].append(c)
    matStructure['Sen'].append(SensorName)
    matStructure['Sig'].append(Sig)
    matStructure['DOF'].append(DOF)
    matStructure['Frame'].append(Frame)
    matStructure['A'].append(A)
    matStructure['C'].append(C)
    matStructure['D'].append(D)
    matStructure['F'].append(F)
    matStructure['I'].append(I)
    matStructure['Data'].append(Data)
    matStructure['Quality'].append(quality)





def add2StrucRaw(subjectNum,name,type_tmp,c,SensorName,Sig,DOF,Frame,A,C,D,F,I,Data,quality):

    if 'R' in type_tmp or 'L' in type_tmp:
        side = type_tmp[0]
        type_tmp = type_tmp[1:]


    if 'R' in SensorName or 'L' in SensorName:
        SensorName = SensorName[1:]

    RaWmatStructure['Num'].append(int(subjectNum))
    if 'Pre' in name:
        RaWmatStructure['Stat'].append("Pre")
    else :
        RaWmatStructure['Stat'].append("Post")

    RaWmatStructure['Side'].append(side)

    RaWmatStructure['Task'].append(type_tmp)
    RaWmatStructure['Trial'].append(c)
    RaWmatStructure['Sen'].append(SensorName)
    RaWmatStructure['Sig'].append(Sig)
    RaWmatStructure['DOF'].append(DOF)
    RaWmatStructure['Frame'].append(Frame)
    RaWmatStructure['A'].append(A)
    RaWmatStructure['C'].append(C)
    RaWmatStructure['D'].append(D)
    RaWmatStructure['F'].append(F)
    RaWmatStructure['I'].append(I)
    RaWmatStructure['Data'].append(Data)
    RaWmatStructure['Quality'].append(quality)


# # TEST Quaternion
# AccTest=[]
# GyroTest=[]

# # 0.0439209209118454	0.0144013850701204	0.984497216069645

# for i in range(1000):

#     AccTest.append([0.0552798410718251,	0.214545870547593,0.962903639937636])
#     GyroTest.append([2.30176391646432,	5.30861999647153,-4.77797895288918])

#     # AccTest.append([0.00586463435423756,0.201200685105144,0.978916998607301])
#     # GyroTest.append([-1.52131751500013,1.03951637712464,0.701306308668641])
    





# beta = 0.1
# _q = computeOrientation(AccTest,GyroTest,dt_ms/1000 , beta , 10000)
# # Q2E
# Euler = np.array([i.get_euler() for i in _q ])                                               
# roll = [i.roll for i in Euler]
# pitch = [i.pitch for i in Euler]
# yaw = [i.yaw for i in Euler]


# # q = [quat( 0.8535534, -0.3535534, 0.3535534, -0.1464466)]
# a = processing.Compute_Body2G_rotation([[0.0439209209118454,	0.0144013850701204,	0.984497216069645]],[_q[0]])


# a= 0