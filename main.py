import sys
import os
import pyc3dtools
import numpy as np
import argparse
import matplotlib.pyplot as plt

import Force_plate_process
from matplotlib.widgets import SpanSelector
from rich.console import Console
from rich.table import Table



Token = "YOURTOKEN"



def redC3D(path, forceplaet=0):
    c3d  = pyc3dtools.readC3D(Token,path)    
    if c3d['Status']=='Failed':
        print(f"Failed to Read File... | {c3d['error']}") 
        sys.exit(0)
        
    else :
        
        print('---------------------------- C3Dtools.Com ----------------------------')
        print(f"Header::Number of Markers = {c3d['Header']['Number_of_Points']}")
        print(f"Header::First Frame = {c3d['Header']['first_frame']}")
        print(f"Header::Last Frame = {c3d['Header']['last_frame']}")
        print(f"Header::Video Sampling Rate = {c3d['Header']['Video_Frame_Rate']}")
        print(f"Header::Analog Channels = {c3d['Header']['Analog_number_channel']}")
        print(f"Header:: Analog Sample Rate = {c3d['Header']['Analog_Frame_Rate']}")
        print(f"Header:: Analog sample per video frame = {c3d['Header']['Analog_sample_per_Frame']}")


        print('----------------------------------------------------------------------')
        print(f"GP::Markers Label = {c3d['Markers Label']}")
        print(f"GP::Analog Label = {c3d['Analog Label']}")
        print('----------------------------------------------------------------------')
        print(f"Markers:: Frame->0 , {c3d['Markers Label'][0]}  = {c3d['Markers'][0][0][:3]}")
        print(f"Markers:: Frame->100 , {c3d['Markers Label'][0]}  = {c3d['Markers'][100][0][:3]}")
        print(f"Markers:: Frame->150 , {c3d['Markers Label'][1]}  = {c3d['Markers'][150][1][:3]}")
        print(f"Markers:: Units = {c3d['Units']}")
        print(f"coordinate System [X_SCREEN, Y_SCREEN] = {c3d['Coordinate system']}")
        print('----------------------------------------------------------------------')
        print(f"Number Of Forceplates = {len(c3d['ForcePlate'])}")
        #print(f"First plate:: FX, FY, FZ :: Frame->100 = ({c3d['ForcePlate'][0]['FX'][100] ,c3d['ForcePlate'][0]['FY'][100],c3d['ForcePlate'][0]['FZ'][100] })") # Analog sample per video frame is equal 20 
        print(f"First plate:: FX, FY, FZ :: Frame->100 :: Analog Sample 10 = {c3d['ForcePlate'][0]['FX'][100][0] ,c3d['ForcePlate'][0]['FY'][100][0],c3d['ForcePlate'][0]['FZ'][100][0] }") # Analog sample per video frame is equal 20 
        print(f"First plate:: Corners = {c3d['ForcePlate'][0]['corners']}")
        print(f"First plate:: Origin = {c3d['ForcePlate'][0]['Origin']}")
        print(f"First plate:: COP :: X,Y,Z :: Frame->50 :: Analog Sample 1 = {c3d['ForcePlate'][0]['COP'][50][0][1],c3d['ForcePlate'][0]['COP'][50][1][1],c3d['ForcePlate'][0]['COP'][50][2][1]}") # Analog sample per video frame is equal 20 
        
        
        FZ = np.array(c3d['ForcePlate'][2]['FZ'])
        FZ = FZ.flatten()
        
        # trim data 
        trimX1 = 0
        trimX2 = len(FZ)

        # Capture mouse selection
        def onselect(xmin, xmax):
            nonlocal trimX1, trimX2
            trimX1 = round(xmin)
            trimX2 = round(xmax)
            print(f"Selected range: Start = {trimX1}, End = {trimX2}")
            plt.close()  
            
        fig, ax = plt.subplots()
        ax.plot(FZ)
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True,  props=dict(alpha=0.5, facecolor='red'))
        plt.show()
        
        
        # Trim FZ
        trimFZ = FZ[int(trimX1):int(trimX2)]
        
        return trimFZ,c3d['Header']['Analog_Frame_Rate']
        


def makeRow(val):
    return f"{val}"
    




def main():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--arg1', type=str, help='First argument')
    parser.add_argument('--arg2', type=int, help='Second argument')
    args = parser.parse_args()

    print(f"Argument 1: {args.arg1}")
    print(f"Argument 2: {args.arg2}")
    
    FZ,samplingRate = redC3D(args.arg1,args.arg2)
    mass = np.mean(FZ[0:100])
    proce_data =  Force_plate_process.process(FZ,samplingRate,mass) #subject mass
    
    console = Console()

    table = Table(title="Jump Analysis Results")

    # Define the columns and their properties
    columns = [
        {"header": "Metric", "justify": "left", "style": "cyan", "no_wrap": True},
    ]
    
    for j in proce_data['HJ']:
        if j == proce_data['HJ'][-1]:
            columns.append({"header": "mean", "justify": "right", "style": "green"})
        else :
            columns.append( {"header": "jump", "justify": "right", "style": "magenta"})

    # Add columns programmatically
    for col in columns:
        table.add_column(col["header"], justify=col["justify"], style=col["style"], no_wrap=col.get("no_wrap", False))
    # Add values to existing rows
    table.add_row("Height of Jumps(m)", *[str(round(val,2)) for val in proce_data['HJ']])
    table.add_row("Flight Times(s)", *[str(round(val,2)) for val in proce_data['FT']])
    table.add_row("RSImod",*[str(round(val,2)) for val in proce_data['RSImod']])
    table.add_row("Take-Off Force",*[str(round(val,2)) for val in proce_data['TKForce']])
    

    console.print(table)
          
    #plot velocity
    plt.plot(proce_data['FZ'])
    plt.plot(proce_data['steady'])
    
    plt.title("Results")
    plt.xlabel("Sample Index")
    plt.ylabel("Force (N)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()