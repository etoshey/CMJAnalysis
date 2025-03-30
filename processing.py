import numpy as np
from scipy import signal
import pyquaternion
import math
import copy

from dataclasses import dataclass
from dataclasses import field

from scipy.spatial.transform import Rotation as R

from sklearn.metrics import mean_squared_error

from scipy import signal

from scipy import stats










def integral(y,x,steady)->[]:
    # len(y)==len(x)
    output = [0]
    fb = 0 
    for i in range(1,len(y)):
        if steady[i] == 1 and steady[i-1] == 0: #rising
            fb = i            
        
        if steady[i] == 1:
            output.append(np.trapz(y[fb:i],x[fb:i]))
        else :
            output.append(0) 


    # Drift Correction
   
    fb = 0
    for i in range(1,len(output)):
        if steady[i] == 1 and steady[i-1] == 0: #rising
            fb = i
        elif steady[i] == 0 and steady[i-1] == 1: #falling
            if i>fb:
                dif_slop = (0 - output[i-1]) / (i-fb)

                for j in range(i-fb):
                    output[fb+j] +=  (dif_slop * j)



    return output



def DownSample(data,dim):

    return signal.resample(data,dim)

def integral2(y,x)->[]:
    # len(y)==len(x)
    output = [0]
    fb = 0 
    for i in range(1,len(y)):     
        output.append(np.trapz(y[0:i],x[0:i]))

    return np.array(output)




def low_pass_filter(cutoff,dt_ms,data)->[]:
    #apply low-pass filter
    fs = 1000 / dt_ms
    fc = cutoff  # Cut-off frequency of the filter
    w = fc / ( fs/ 2) # Normalize the frequency
    b, a = signal.butter(2, w, 'low')
    filtered_data = signal.filtfilt(b, a,data)
    return filtered_data


def low_pass_filter_3axis(dt_ms,data, cutoff)->[]:
    #apply low-pass filter
    fs = 1000 / dt_ms
    fc = cutoff  # Cut-off frequency of the filter
    w = fc / ( fs/ 2) # Normalize the frequency
    b, a = signal.butter(2, w, 'low')

    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    filtered_data_x = signal.filtfilt(b, a,x)
    filtered_data_y = signal.filtfilt(b, a,y)
    filtered_data_z = signal.filtfilt(b, a,z)
    filtered_data = []
    for i in range(len(filtered_data_x)):
        filtered_data.append([filtered_data_x[i], filtered_data_y[i], filtered_data_z[i]])



    return np.array(filtered_data)


def steady_detection(signal,YThreshold=10)->[int]:
    # get std from 100 first frame
    
    std = np.std(signal[0:500])
    avg = np.average(signal[0:500])
    abs_signal = np.abs(signal)

    steady_area  = np.where(abs_signal < (avg + 3*std), 0,1)
    Start_flag = False
    Start_index = 0
    End_index= 0
    End_flag = False
    Pulse_duration = 0
    Threshold = 50 #sample

    for i in range(len(steady_area)):
        if steady_area[i] > 0 and Start_flag == False:
            Start_flag = True
            Start_index = i    
        elif steady_area[i] > 0 and Start_flag == True:
            Pulse_duration = 0
            End_flag = False
        elif steady_area[i] == 0 and Start_flag == True and Pulse_duration < Threshold:
            End_flag = True
            Pulse_duration +=1           
        elif Start_flag == True and End_flag == True and Pulse_duration >= Threshold:
            if np.max(signal[Start_index: i - Pulse_duration]) > YThreshold: # 10 m/s2   #(avg + 100*std): 
                steady_area[Start_index: i - Pulse_duration] = 1
            else :
                steady_area[Start_index: i - Pulse_duration] = 0

            Start_flag = False
            End_flag = False
            Pulse_duration = 0
            Start_index = 0


    if Start_flag == True : # if there is no any steady at the End of data
       
        if np.max(signal[Start_index: len(signal)]) > YThreshold: # 10 m/s2   #(avg + 100*std): 
            steady_area[Start_index: len(signal)] = 1
        else :
            steady_area[Start_index: len(signal)] = 0

        # Start_flag = False
        # End_flag = False
        # Pulse_duration = 0
        # Start_index = 0
            
        


    return steady_area


def steady_detection_window(signal,YThreshold=10)->[int]:

    window = 150
    abs_signal = np.abs(signal)
    output = []

    for i in range(len(signal)-window):                  
            w_std = np.std(abs_signal[i:i+window])
            w_avg = np.average(abs_signal[i:i+window])
            w_max = np.average(abs_signal[i:i+window])

            if w_avg < 2 and w_std < 1:
                output.append(0)
            else:
                output.append(1)

    for i in range(window):
        output.append(0)


    output = np.array(output)
    start_index = 0
    ii=0
    while(ii < len(output)): 

        if output[ii] == 1 and output[ii-1]==0 :
            start_index=ii 
        elif output[ii] == 0 and output[ii-1] ==1:
            if np.max(signal[start_index: ii]) > YThreshold: # 10 m/s2   #(avg + 100*std): 
                output[start_index: ii+window] = 1
                ii+= window
            else :
                output[start_index: ii] = 0


        ii+=1

    return output



def force_plate_steady_detection_window(signal,freq,YThreshold=10)->[int]:

    window = round(freq/5)   
    output = []
    rising_index=0

    std = np.std(signal[0:round(freq/2)])
    avg = np.average(signal[0:round(freq/2)])

    for i in range(len(signal)-window):                  
            w_std = np.std(signal[i:i+window])
            w_avg = np.average(signal[i:i+window])            

            
            if (w_avg>(avg+YThreshold) or w_avg<(avg-YThreshold)):
                output.append(100)
                rising_index = i                         
            elif i > rising_index+(freq/2) or rising_index==0 :
                output.append(0)
            else :
                output.append(100)

    for i in range(window):
        output.append(0)


    # output = np.array(output)
    # start_index = 0
    # ii=0
    # while(ii < len(output)): 

    #     if output[ii] == 1 and output[ii-1]==0 :
    #         start_index=ii 
    #     elif output[ii] == 0 and output[ii-1] ==1:
    #         if np.max(signal[start_index: ii]) > YThreshold: # 10 m/s2   #(avg + 100*std): 
    #             output[start_index: ii+window] = 1
    #             ii+= window
    #         else :
    #             output[start_index: ii] = 0


    #     ii+=1

    return output




def steady_detection_vel(signal,YThreshold=1)->[int]:
    # get std from 100 first frame
    

    abs_signal = np.abs(signal)

    steady_area  = np.where(abs_signal < 0.1, 0,1)
    Start_flag = False
    Start_index = 0
    End_index= 0
    End_flag = False
    Pulse_duration = 0
    Threshold = 50 #sample

    for i in range(len(steady_area)):
        if steady_area[i] > 0 and Start_flag == False:
            Start_flag = True
            Start_index = i    
        elif steady_area[i] > 0 and Start_flag == True:
            Pulse_duration = 0
            End_flag = False
        elif steady_area[i] == 0 and Start_flag == True and Pulse_duration < Threshold:
            End_flag = True
            Pulse_duration +=1           
        elif Start_flag == True and End_flag == True and Pulse_duration >= Threshold:
            if np.max(signal[Start_index: i - Pulse_duration]) > YThreshold: # 10 m/s2   #(avg + 100*std): 
                steady_area[Start_index: i - Pulse_duration] = 1
            else :
                steady_area[Start_index: i - Pulse_duration] = 0

            Start_flag = False
            End_flag = False
            Pulse_duration = 0
            Start_index = 0


    if Start_flag == True : # if there is no any steady at the End of data
       
        if np.max(signal[Start_index: len(signal)]) > YThreshold: # 10 m/s2   #(avg + 100*std): 
            steady_area[Start_index: len(signal)] = 1
        else :
            steady_area[Start_index: len(signal)] = 0

        # Start_flag = False
        # End_flag = False
        # Pulse_duration = 0
        # Start_index = 0
            
        


    return steady_area




def getJumpZoon(steady_signal)->[]:

    output = []
    start = 0
    end = 0
    for i in range(1,len(steady_signal)):
        if steady_signal[i] == 1 and steady_signal[i-1]==0: #Rising
            start = i
        elif steady_signal[i] == 0 and steady_signal[i-1]==1 : #Falling
            end = i
            output.append({'start':start , 'end' : end})
            start = 0
            end = 0
    
    if start !=0 : # if there is no any steady at the End of data
        output.append({'start':start , 'end' : len(steady_signal)-1})


    return output





def getLocalMax(signal, search_area):

    JH = []
    for a in search_area:
        JH.append(np.max(signal[a['start']:a['end']]))

    return JH


def Velocity_correction(V,ACC,freq):
   

    # -----------------------------------------------
    w = round(freq/4)
    max_index = [{'val':0 , 'index':0} , {'val':0 , 'index':0}]
    for i in range(0,len(ACC)-w,w):
        
        max = np.max(ACC[i:i+w])
        max_i = i + np.argmax(ACC[i:i+w])
        if max > max_index[0]['val'] and max_i-max_index[1]['index'] > w and max_index[0]['val']  < max_index[1]['val']:            
            max_index[0] = {'val':max , 'index' : max_i}
        elif max > max_index[1]['val'] and max_i-max_index[0]['index'] > w: 
            max_index[1] = {'val':max , 'index' : max_i}
    
        



   
    # -----------------------------------------------
    start = 0
    end = 0
    if max_index[0]['index'] > max_index[1]['index']:
        start = max_index[1]['index']
        end = max_index[0]['index']
    else:
        start = max_index[0]['index']
        end = max_index[1]['index']


    max_v = np.max(V[start:end])
    max_v_i = np.argmax(V[start:end])
    min_v = np.max(V[start:end])
    min_v_i = np.argmin(V[start:end])

    mid_val_i = round((max_v_i + min_v_i)/2)
    offset = V[start+mid_val_i]




    Copy_V = copy.copy(V[start:end])
    Copy_V -= offset
    # step1 find Min
    a = np.empty(start)
    b = np.empty(len(V)-end)
    a.fill(0)
    b.fill(0)

    output = np.concatenate([a,Copy_V,b])

   


    
       
    return output

        
def CompareSignalPlot(s1,s2):

     
    tmp = np.array(s1) 
    tmp = tmp[~np.isnan(tmp)]
    max = np.max(tmp)

    tmp = np.array(s2) 
    tmp = tmp[~np.isnan(tmp)]
    if max < np.max(tmp):
        max = np.max(tmp)
    


    return [s1/max , s2/max]
        
        
def Sample100(d):

    step = round(len(d)/1000)  
    tmp = np.array(d) 
    tmp = tmp[~np.isnan(tmp)]
    max = np.max(tmp)
    output = []
    for i  in range(0,len(d)):
        output.append(d[i])

    return np.array(output)


    
def find_peaks(data):


    w = round(len(data)/100)
    max_index = [{'val':0 , 'index':0} , {'val':0 , 'index':0}]
    for i in range(0,len(data)-w,w):
        
        max = np.max(data[i:i+w])
        max_i = i + np.argmax(data[i:i+w])
        if max > max_index[0]['val'] and max_i-max_index[1]['index'] > w and max_index[0]['val']  < max_index[1]['val']:            
            max_index[0] = {'val':max , 'index' : max_i}
        elif max > max_index[1]['val'] and max_i-max_index[0]['index'] > w: 
            max_index[1] = {'val':max , 'index' : max_i}


    return max_index  


def find_peaks2(data):

    data = data[~np.isnan(data)]
    w = round(len(data)/100)
    max_index = [{'val':0 , 'index':0} , {'val':0 , 'index':0}]
    zero_len=0
    split_index = 0
    for i in range(0,len(data)):

        if data[i]<0.01:
            zero_len +=1

            if zero_len > 10:
                split_index=i
                break
        else:
            zero_len =0

    if split_index>0:
        max_index[0] = { 'val' : np.max(data[0:split_index]) , 'index' : np.argmax(data[0:split_index]) }
        max_index[1] = { 'val' : np.max(data[split_index:len(data)]) , 'index' : split_index + np.argmax(data[split_index:len(data)]) }
    




    return max_index  


def rsme(d,ref):
   MSE =  mean_squared_error(ref,d)
   return math.sqrt(MSE)



def pearsonCorr(x,y):
    return stats.pearsonr(x,y)


def getAllEvents(VEL):
    take_off_vel = np.max(VEL)  
    take_off_vel_index = np.argmax(VEL)
    landing_index = take_off_vel_index + np.argmin(VEL[take_off_vel_index:])
    unWeighting_phase_index = np.argmin(VEL[0:take_off_vel_index])                      
    Barking_phase_index = next((index for index, item in enumerate(VEL[unWeighting_phase_index:take_off_vel_index]) if item > 0),None)

    return [take_off_vel,take_off_vel_index,landing_index,unWeighting_phase_index,unWeighting_phase_index+Barking_phase_index]


def getHJ(max_vel):
    return (max_vel*max_vel) / (2 * 9.81)

def getHJ_FT(FT):
    return FT*FT/8 * 9.81

def get_Force(ACC,mass):
    return ACC * mass     


def get_Power(F,V):        
    return F * V   


def get_RSImod(HJ,FT):        
    return HJ/FT   


def get_normal_q(Q):

    global_quat = pyquaternion.Quaternion(1,0,0,0)
    sensor_quat = pyquaternion.Quaternion(Q.w,Q.x,Q.y,Q.z)
    sensor_quat_inv = sensor_quat.inverse
    #  FromAtoB = Quaternion.Inverse(rotationA) * rotationB
    return sensor_quat_inv * global_quat    


def get_body_q(Q , ref):
    output = []
    for q in Q:      
        output.append(pyquaternion.Quaternion(q.w,q.x,q.y,q.z) * ref)

    return output


def GetFreeGlobalAcc(Q,ACC):
    gravity = pyquaternion.Quaternion(0,0,0,-1)
    Gx = []
    Gy = []
    Gz = []

    for inx, q in enumerate(Q):
        _q = pyquaternion.Quaternion(q.w,q.x,q.y,q.z)
        local_gravity = _q.conjugate * gravity * _q
        free_acc = pyquaternion.Quaternion(0,ACC[0][inx] + local_gravity.x , ACC[1][inx] + local_gravity.y , ACC[2][inx] + local_gravity.z)
        g_acc = _q * free_acc * _q.conjugate
        Gx.append(g_acc.x)
        Gy.append(g_acc.y)
        Gz.append(g_acc.z)
    
    return [Gx , Gy, Gz]



def test():
    v1 = np.array([0,1,0])
    v2 = np.array([0,0,1])

    axis = np.cross(v2,v1)
    Angle = getAngle(v1,v2)
    R1 = MatrixFromAxisAngle(axis,Angle)

    nv = np.matmul(R1,v2)
    print(nv)


def RotateX90(Vector):
    output = []
    for i in range(len(Vector)):
        output.append([Vector[i][0] , Vector[i][2]*-1 , Vector[i][1]])
    return output


def Compute_Body2G_rotation(Vector,Q):
    output = []
    for frame in range(len(Vector)):
        global_quat = pyquaternion.Quaternion(Q[frame].w,Q[frame].x,Q[frame].y,Q[frame].z)
        Vector_quat = pyquaternion.Quaternion(0,Vector[frame][0],Vector[frame][1],Vector[frame][2])
        global_quat_conj = global_quat.conjugate
        #  FromAtoB = Quaternion.Inverse(rotationA) * rotationB
        g =  global_quat * Vector_quat * global_quat_conj
        output.append([g.x , g.y, g.z])

    return output




def Compute_Body2Body_rotation(q1,q2,v):
    
    output = []
    for frame in range(len(v)):
        global_quat1 = pyquaternion.Quaternion(q1[frame].w , q1[frame].x , q1[frame].y , q1[frame].z)
        global_quat2 = pyquaternion.Quaternion(q2[frame].w , q2[frame].x , q2[frame].y , q2[frame].z)

        Vector_quat = pyquaternion.Quaternion(0,v[frame][0],v[frame][1],v[frame][2])
        global_quat_conj1 = global_quat1.conjugate
        global_quat_conj2 = global_quat2.conjugate
        #  FromAtoB = Quaternion.Inverse(rotationA) * rotationB
        g =  global_quat1 * Vector_quat * global_quat_conj1        
        g2 =  global_quat_conj2 * g * global_quat2
        
        output.append([g2.x , g2.y , g2.z])

    return output


def Compute_KAM2(tibiaLen,qTibia, Acc):
    output = []
    for frame in range(len(Acc[0])):
        global_quat = pyquaternion.Quaternion(qTibia[frame].w,qTibia[frame].x,qTibia[frame].y,qTibia[frame].z)
        Vector_quat = pyquaternion.Quaternion(0,0,0,tibiaLen)   # Z-UP
        global_quat_conj = global_quat.conjugate
        #  FromAtoB = Quaternion.Inverse(rotationA) * rotationB
        tibG =  global_quat * Vector_quat * global_quat_conj
        vGRF2KJC = [tibG.x - Acc[0][frame] , tibG.y - Acc[1][frame] , tibG.z - Acc[2][frame]]
        Vcross = np.cross([Acc[0][frame] , Acc[1][frame] , Acc[2][frame]] , vGRF2KJC)
        Vcross_quat = pyquaternion.Quaternion(0,Vcross[0],Vcross[1],Vcross[2])   # Z-UP
        MTibLocal = global_quat_conj * Vcross_quat * global_quat
        output.append([MTibLocal.x , MTibLocal.y , MTibLocal.z])

    return output




def Compute_Sensor2Body_rotation(acc_static, acc_laying):

    exp_acc = np.array([0,1,0])
    avg_acc_standing_x = np.average(acc_static[0,:])
    avg_acc_standing_y = np.average(acc_static[1,:])
    avg_acc_standing_z = np.average(acc_static[2,:])
    current_acc = np.array([avg_acc_standing_x, avg_acc_standing_y, avg_acc_standing_z])
    # Cross Product
    axis1 = np.cross(current_acc,exp_acc)
    axis1 = axis1 / np.linalg.norm(axis1)
    Angle = getAngle(exp_acc,current_acc)
    R1 = MatrixFromAxisAngle(axis1 , Angle)

    nv = np.matmul(R1,current_acc)
    # Apply R1
    acc_laying_R1 = []
    for inx,f in enumerate(acc_laying[0]):
        acc = np.array([[acc_laying[0,inx]],[acc_laying[1,inx]],[acc_laying[2,inx]]])        
        acc_laying_R1.append(np.matmul(R1,acc))
    
    acc_laying_R1 = np.array(acc_laying_R1)

    # test
    acc_static_R1 = []
    for inx,f in enumerate(acc_static[0]):
        acc = np.array([[acc_static[0,inx]],[acc_static[1,inx]],[acc_static[2,inx]]])         
        tmp = np.matmul(R1,acc)       
        acc_static_R1.append([tmp[0][0],tmp[1][0],tmp[2][0]])
    
      
    # step 2
    exp_acc = np.array([0,0,1])
    avg_acc_laying_x = np.average(acc_laying_R1[:,0])
    avg_acc_laying_y = np.average(acc_laying_R1[:,1])
    avg_acc_laying_z = np.average(acc_laying_R1[:,2])
    current_acc = np.array([avg_acc_laying_x, 0 , avg_acc_laying_z])
     # Cross Product
    axis1 = np.cross(current_acc,exp_acc)
    axis1 = axis1 / np.linalg.norm(axis1)
    # axis1 = np.array([0,1,0])
    Angle = getAngle(current_acc,exp_acc) # must be 90 deg
    R2 = MatrixFromAxisAngle(axis1 ,  Angle)

    # Apply R2
    acc_laying_R2 = []
    for inx,f in enumerate(acc_laying_R1):       
        
        acc = np.array([[acc_laying_R1[inx,0][0]],[acc_laying_R1[inx,1][0]],[acc_laying_R1[inx,2][0]]])
        tmp = np.matmul(R2,acc)
        acc_laying_R2.append([tmp[0][0],tmp[1][0],tmp[2][0]])



    return {'R1': R1, 'R2':R2 , 'eval' : np.array(acc_laying_R2) , 'test' : np.array(acc_static_R1) }


def Apply_Sensor2Body(R1,R2,gyro):
    cor_gyro = []
    for inx in range(len(gyro[0])):
        gyro_tmp = np.array([[gyro[0,inx]],[gyro[1,inx]],[gyro[2,inx]]])

        if len(R1)>0:
            temp1 = np.matmul(R1,gyro_tmp)
            temp2 = np.matmul(R2,temp1)
        else:
            temp2 = gyro_tmp
        
        cor_gyro.append([temp2[0][0],temp2[1][0],temp2[2][0]])

    return cor_gyro



def RotateM(R,V):
    
    outPut = []
    for i in range(len(V)):
        outPut.append(np.matmul(R,V[i]))

    return outPut


def getAngle(u,v):
    return  np.arccos(u.dot(v)/(np.linalg.norm(u)*np.linalg.norm(v)))



def MatrixFromAxisAngle(axis, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c

    m00 = c + axis[0]*axis[0]*t
    m11 = c + axis[1]*axis[1]*t
    m22 = c + axis[2]*axis[2]*t

    tmp1 = axis[0] * axis[1] * t
    tmp2 = axis[2] * s
    m10 = tmp1 + tmp2
    m01 = tmp1 - tmp2
    tmp1 = axis[0] * axis[2] * t
    tmp2 = axis[1] * s
    m20 = tmp1 - tmp2
    m02 = tmp1 + tmp2
    tmp1 = axis[1] * axis[2] * t
    tmp2 = axis[0] * s
    m21 = tmp1 + tmp2
    m12 = tmp1 - tmp2

    return np.array([[m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]])



def RotationMatrix2Euler(R):
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x * 180/np.pi , y * 180/np.pi, z * 180 / np.pi])


def Q2E(q):
    output = []
    for item in q:
        r = R.from_quat([item.w , item.x , item.y , item.z])
        output.append(r.as_euler('xyz', degrees=True))

    return output



def getTibiaLen(height):
    # David Winter
    # return height * 0.285
    return 0.285


def getfootH(height):
    # David Winter
    # return height * 0.285
    return 0.039


def Compute_KAM(Tibia_sagital_angle , Tibia_frontal_angle, tibia_len, Vgrf, Hgrf):

    # M1
    arm1 = [math.cos(math.radians(val))*tibia_len for val in Tibia_sagital_angle]
    arm2 = [math.sin(math.radians(val))*tibia_len for val in Tibia_frontal_angle]

    # arm1 = low_pass_filter(20,2,arm1)
    # arm2 = low_pass_filter(20,2,arm2)


    M1 = arm1 * Hgrf
    M2 = arm2 * Vgrf

    return {'arm1' : arm1 , 'arm2':arm2, 'M1':M1 , 'M2':M2 }


def removeZeroes(data1,data2):

    d1 = []
    d2 = []
    for i in range(len(data1)):
        if data1[i]!=0 and data2[i]!=0:
            d1.append(data1[i])
            d2.append(data2[i])


    return [d1,d2]


def removeOUTDataPair(data1, data2):

    d1n = []
    d2n = []

    for inx,item in  enumerate(data1):
        if not math.isnan(item) and not math.isnan(data2[inx]):
           d1n.append(item)
           d2n.append(data2[inx])


    avg = np.average(d1n)
    std = np.std(d2n)
    d1 = []
    d2 = []

    for inx,item in  enumerate(d1n):
        if item  < avg + (1*std) or item > avg - (1*std):
           d1.append(item)
           d2.append(d2n[inx])


    avg = np.average(d2)
    std = np.std(d2)

    d11 =[]
    d22 =[]
    for inx,item in  enumerate(d2):
        if item  < avg + (1*std) or item > avg - (1*std):
            d22.append(item)
            d11.append(d1[inx])
   

    return [d11,d22]


def Percentile(data,p):

    return np.percentile(np.array(data),p)

def ZScoreNormalization(data):

    
    avg = np.average(data)
    std = np.std(data)
    if std==0:
        return (data-avg)
    else:
        return (data-avg)/std
    

def pairdTtest(pre, post):
    return stats.ttest_rel(pre, post) 


def OneWayAnova(pre, post):
    return stats.f_oneway(pre, post) 

def chi2_contingency(pre,post):
    return stats.chi2_contingency([pre,post])


def meanReletiveError(data,ref):

    tmp = 0
    for i in range(len(data)):
        if (ref[i]!=0):
            tmp += abs((data[i] - ref[i])/ref[i])*100

    return(round(tmp/len(data),2))


def meanError(data,ref):

    tmp=0
    for i in range(len(data)):
        tmp += (data[i] - ref[i])       

    return(round(tmp/len(data),2))



def rmse(data,ref):

    tmp = 0
    for i in range(len(data)):
        tmp += pow(data[i] - ref[i],2)       

    return(round(math.sqrt(tmp/len(data)) ,2))