import numpy as np



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


    return output



def process(data,sample_freq,mass):
    output = dict()
    try:
   
        # find start point

        std = np.std(data[0:round(sample_freq/2)])
        avg = np.average(data[0:round(sample_freq/2)])

        max = np.max(data)

        # low_pass filter
        dt = 1000/sample_freq
        time = np.arange(0,len(data),dt) / 1000
        # signal = pros.low_pass_filter(dt,np.array(data))
        signal = np.array(data)


        steady = force_plate_steady_detection_window(signal,sample_freq,10*std)
        
        

        # compute Acc
        ACC = (signal / mass)

        ACC *= 9.81

        #shift 2 zero
        offset = np.average(ACC[0:100])
        ACC = ACC - offset
        
        # Force
        Force = signal

        # OutCome
        jumps = []     
    

        # compute Vel   
        Vel = [0]
        startInx = 0
        for i in range(1,len(ACC)):
            if steady[i] == 100:
                if startInx == 0:
                    startInx = i
                    
                Vel.append(np.trapezoid(ACC[startInx:i],time[startInx:i]))
                
            else:
                
                # extract Jump Features 
                if startInx != 0:
                    tmp = Vel[startInx:i]
                    max_Vel = np.max(tmp)
                    takeoff_index = np.argmax(tmp) 
                    landing_index = takeoff_index+np.argmin(tmp[takeoff_index:-1])
                
                    #Correction
                    mid_val_i  = round((takeoff_index+landing_index) /2)
                    offset = tmp[mid_val_i]

                    tmp = np.array(tmp)
                    tmp[takeoff_index:landing_index] -= offset
                    
                    tmpF = Force[startInx:i]
                    
                    jumps.append({ 'Vel' : tmp , 'Force' : tmpF })                            
                
                
                    Vel[startInx:i] = tmp
                    
                    
                Vel.append(0)
                startInx = 0
                



        
        heightOfJumps = []   
        flightTimes = []
        contactTimes = []
        RSImod = [] 
        takeoffForce = []     

        for jump in jumps:
            
            max_Vel = np.max(jump['Vel'])
            min_Vel = np.min(jump['Vel'])
            max_Vel_inx = np.argmax(jump['Vel'])
            min_Vel_inx = np.argmin(jump['Vel'])
            
            
            _time = np.arange(0,len(jump['Vel']),dt) / 1000
            
            hj = (max_Vel*max_Vel) / (2 * 9.81)
            heightOfJumps.append(hj)
            flightTimes.append(_time[min_Vel_inx] - _time[max_Vel_inx])
            contactTimes.append(_time[max_Vel_inx])
            RSImod.append(hj / _time[max_Vel_inx])    
            
            takeoffForce.append(np.max(jump['Force'])/mass)        
            
            
            



        Dis = [0,0]



        # add average value
        heightOfJumps.append(np.mean(heightOfJumps))
        flightTimes.append(np.mean(flightTimes))
        contactTimes.append(np.mean(contactTimes))
        RSImod.append(np.mean(RSImod))
        takeoffForce.append(np.mean(takeoffForce))
        


        output['FZ'] = signal
        output['steady'] = steady
        output['Acc'] = ACC
        output['Vel'] = Vel
        output['Dis'] = Dis

        output['FT'] = flightTimes
        output['HJ'] = heightOfJumps
        output['HJ-FT'] = 0
        output['HJ_disp'] = 0
        output['Vel_max'] = 0
        output['RSImod'] = RSImod
        output['TKForce'] = takeoffForce
    except Exception as e:
        print(e)
        output['FZ'] = signal
        output['steady'] = steady
        output['Acc'] = []
        output['Vel'] = []
        output['Dis'] = []

        output['FT'] = 0
        output['HJ'] = 0
        output['HJ-FT'] = 0
        output['HJ_disp'] = 0
        output['Vel_max'] = 0
        output['RSImod'] = 0





    return output









