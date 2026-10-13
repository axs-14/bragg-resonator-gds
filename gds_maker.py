# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:41 2026

@author: Aryan
"""

import numpy as np
import gdspy
import sys 
import math
from typing import Tuple, List

def createNotch(start,length=10,width=50,layer=1):
    polygon_pts = [(start[0]-length/2, start[1]-width/2),(start[0]-length/2, start[1]+width/2),\
                   (start[0]+length/2, start[1]+width/2),(start[0]+length/2, start[1]-width/2)]
    polygon = [gdspy.Polygon(polygon_pts,layer = layer)]
    final_position = [start[0]+length/2,start[1]]
    
    return [polygon, final_position]

def createCPWvertDown(start,length,gap=5,width=10,layer=1):
    polygon= []
    polygon_pts1 = [(start[0]-width/2-gap, start[1]),(start[0]-width/2, start[1]),\
                   (start[0]-width/2, start[1]-length),(start[0]-width/2-gap, start[1]-length)]
    polygon_pts2 = [(start[0]+width/2+gap, start[1]),(start[0]+width/2, start[1]),\
                   (start[0]+width/2, start[1]-length),(start[0]+width/2+gap, start[1]-length)]
    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    final_position = [start[0],start[1]-length]
    
    return [polygon, final_position]

def createCPWvertUp(start,length,gap=5,width=10,layer=1):
    polygon= []
    polygon_pts1 = [(start[0]-width/2-gap, start[1]),(start[0]-width/2, start[1]),\
                   (start[0]-width/2, start[1]+length),(start[0]-width/2-gap, start[1]+length)]
    polygon_pts2 = [(start[0]+width/2+gap, start[1]),(start[0]+width/2, start[1]),\
                   (start[0]+width/2, start[1]+length),(start[0]+width/2+gap, start[1]+length)]
    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    final_position = [start[0],start[1]+length]
    
    return [polygon, final_position]

def createCPWhor(start,length,gap=5,width=10,layer=1):
    polygon= []
    polygon_pts1 = [(start[0], start[1]+width/2+gap),(start[0], start[1]+width/2),\
                    (start[0]+length, start[1]+width/2),(start[0]+length, start[1]+width/2+gap)]
    
    polygon_pts2 = [(start[0], start[1]-width/2-gap),(start[0], start[1]-width/2),\
                    (start[0]+length, start[1]-width/2),(start[0]+length, start[1]-width/2-gap)]

    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    final_position = [start[0]+length,start[1]]
    
    return [polygon, final_position]



def createCPWhorLeft(start,length,gap=5,width=10,layer=1):
    polygon= []
    polygon_pts1 = [(start[0], start[1]+width/2+gap),(start[0], start[1]+width/2),\
                   (start[0]-length, start[1]+width/2),(start[0]-length, start[1]+width/2+gap)]
    
    polygon_pts2 = [(start[0], start[1]-width/2-gap),(start[0], start[1]-width/2),\
                   (start[0]-length, start[1]-width/2),(start[0]-length, start[1]-width/2-gap)]

    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    final_position = [start[0]-length,start[1]]
    
    return [polygon, final_position]

def createLauncherLeft(start,straight=200, height=200, taper = 200, gapL=50, gap=5,width=10,layer=1,notch=False):
    polygon = []
    [poly1,newpos] = createCPWhor(start,length=straight,gap=gapL,width=height-2*gapL,layer=layer)
    polygon += poly1   
    
    polygon_pts1 = [(start[0]+straight, start[1]+height/2-gapL),(start[0]+straight, start[1]+height/2),\
                   (start[0]+straight+taper, start[1]+width/2+gap),(start[0]+straight+taper, start[1]+width/2)]
    
    polygon_pts2 = [(start[0]+straight, start[1]-height/2+gapL),(start[0]+straight, start[1]-height/2),\
                   (start[0]+straight+taper, start[1]-width/2-gap),(start[0]+straight+taper, start[1]-width/2)]
    
    polygon_pts3 = [(start[0]-gapL, start[1]-height/2),(start[0], start[1]-height/2),\
                   (start[0], start[1]+height/2),(start[0]-gapL, start[1]+height/2)] 
    
    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    if notch:
        polygon += [gdspy.Polygon(polygon_pts3,layer = layer)]
        
    final_position = [start[0]+straight+taper,start[1]]    
    
    return [polygon, final_position]


def createLauncherRight(start,straight=200, height=200, taper = 200, gapL=50, gap=5,width=10,layer=1,notch=False):
    polygon = []
    [poly1,newpos] = createCPWhor((start[0]-straight,start[1]),length=straight,gap=gapL,width=height-2*gapL,layer=layer)
    polygon += poly1   
    
    polygon_pts1 = [(start[0]-straight, start[1]+height/2-gapL),(start[0]-straight, start[1]+height/2),\
                   (start[0]-straight-taper, start[1]+width/2+gap),(start[0]-straight-taper, start[1]+width/2)]
    
    polygon_pts2 = [(start[0]-straight, start[1]-height/2+gapL),(start[0]-straight, start[1]-height/2),\
                   (start[0]-straight-taper, start[1]-width/2-gap),(start[0]-straight-taper, start[1]-width/2)]
#    
    polygon_pts3 = [(start[0]+gapL, start[1]-height/2),(start[0], start[1]-height/2),\
                   (start[0], start[1]+height/2),(start[0]+gapL, start[1]+height/2)] 
#    
    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    if notch:
        polygon += [gdspy.Polygon(polygon_pts3,layer = layer)]
    final_position = [start[0]-straight-taper,start[1]]   
    return [polygon, final_position]

def createIndhor(start,ind_l=200, ind_w=200, ind_gap=10, taper = 200, gap=5,width=10,layer=1):
    polygon=[]
    [poly1,newpos1] = createLauncherLeft(start,straight=0, height=2*gap+width, taper = taper, gapL=gap, gap=ind_gap,width=ind_w,layer=layer,notch=False)
    polygon += poly1
    
    [poly2,newpos2] = createLauncherRight((start[0]+ind_l+taper,start[1]),straight=0, height=2*gap+width, taper = taper, gapL=gap, gap=ind_gap,width=ind_w,layer=layer,notch=False)
    polygon += poly2
    
    [poly3,newpos3] = createCPWhor((start[0]+taper,start[1]),length=ind_l-taper,gap=ind_gap,width=ind_w,layer=layer)
    polygon += poly3
    final_position = [start[0]+ind_l + taper,start[1]]   
    
    return [polygon, final_position]


def createT(start, straight=200, height=200, gap=5,width=10,layer=1):
    '''
    creates a T point
    '''
    polygon = []
    [poly1,newpos1] = createCPWhor(start,length=straight,gap=gap,width=width,layer=layer)
    polygon += poly1
    
    [poly2,newpos2] = createCPWvertUp((start[0]+gap+width/2+straight,start[1]+width/2),length=height,gap=gap,width=width,layer=layer)
    polygon += poly2
    
    [poly3,newpos3] = createCPWvertUp((start[0]+gap+width/2+straight,start[1]-width/2-height),length=height,gap=gap,width=width,layer=layer)
    polygon += poly3
    
    polygon_pts = [(start[0]+gap+width+straight, start[1]-width/2),(start[0]+gap+width+straight,start[1]+width/2),\
                   (start[0]+2*gap+width+straight,start[1]+width/2),(start[0]+2*gap+width+straight,start[1]-width/2)]

    polygon += [gdspy.Polygon(polygon_pts,layer = layer)]
    final_position1 = [start[0]+straight+gap+width/2,start[1]+width/2 +height] 
    final_position2 = [start[0]+straight+gap+width/2,start[1]-width/2 -height] 
    
    return [polygon, final_position1, final_position2]



def createDoublearc(center, radius= 50, gap=5, width=10,angle=0,layer=1):
    '''
    with angle of 0, pi/2, pi and 3pi/2 quadracircles are drwan anticloclwise
    '''
    circle = []
    circle += [gdspy.Round(center, radius=radius+width/2+gap, inner_radius=radius+width/2,\
                           initial_angle=np.pi/2+angle,final_angle=angle,layer=layer, number_of_points=40)]
    circle += [gdspy.Round(center, radius=radius-width/2, inner_radius=radius-width/2-gap,\
                           initial_angle=np.pi/2+angle,final_angle=angle,layer=layer,number_of_points=40)]
    final_position = [center[0]+radius*np.cos(angle),center[1]+radius*np.sin(angle)]
    
    return [circle, final_position]




def createSemiarc(center, radius= 50, gap=5, width=10,angle=0,layer=1):
    '''
    with angle of 0, pi/2, pi and 3pi/2 semicircles are drwan anticloclwise
    '''
    circle = []
    circle += [gdspy.Round(center, radius=radius+width/2+gap, inner_radius=radius+width/2,\
                           initial_angle=np.pi+angle,final_angle=angle,number_of_points=40,layer=layer)]
    circle += [gdspy.Round(center, radius=radius-width/2, inner_radius=radius-width/2-gap,\
                           initial_angle=np.pi+angle,final_angle=angle,number_of_points=40,layer=layer)]
    final_position = [center[0]+radius*np.cos(angle),center[1]+radius*np.sin(angle)]
    
    return [circle, final_position]

def CreateMeanderDown(center,length,no_meander=2,radius= 50,gap=5,width=10,layer=1):
    '''
    for a given total length and number of meander calculates a symmetric meander
    '''
    no_meander*=2
    polygon=[]
    length_circle = no_meander*np.pi*radius
    length_straight = (length-length_circle)/no_meander
    x0 = center[0]
    y0 = center[1] - length_straight/2.
    
    if length_straight < 100:
        print("number of meanders too many")
        sys.exit()
    
  ## draw straight sections
    for kk in range(no_meander+1):
        if kk==0:
            [poly1,newpos1] = createCPWvertUp((x0,y0),length=length_straight/2,gap=gap,width=width,layer=layer)
            polygon += poly1

        elif kk== no_meander:
            [poly2,newpos2] = createCPWvertUp((x0+kk*radius*2,y0+length_straight/2),\
                                     length=length_straight/2,gap=gap,width=width,layer=layer)
            polygon += poly2
               
        else:
            [poly3,newpos3] = createCPWvertUp((x0+kk*radius*2,y0),\
                                     length=length_straight,gap=gap,width=width,layer=layer)
            polygon += poly3

    ## draw top semicircles
    
    for kk in range(int(no_meander/2)):
        [poly4, newpos4] = createSemiarc((x0+radius*(4*kk+1),y0),radius=radius, gap=gap, width=width,angle=np.pi,layer=layer)
        polygon += poly4
        [poly5, newpos5] = createSemiarc((x0+radius*(4*kk+3),y0+length_straight),radius=radius, gap=gap, width=width,angle=0,layer=layer)
        polygon += poly5
        
    final_position = [center[0]+no_meander*2*radius,center[1]]
    return [polygon, final_position]

def CreateMeanderUp(center,length,no_meander=2,radius= 50,gap=5,width=10,layer=1):
    '''
    for a given total length and number of meander calculates a symmetric meander
    '''
    no_meander*=2
    polygon=[]
    length_circle = no_meander*np.pi*radius
    length_straight = (length-length_circle)/no_meander
    x0 = center[0]
    y0 = center[1] - length_straight/2.
    
    if length_straight < 100:
        print("number of meanders too many")
        sys.exit()
        
    
  ## draw straight sections
    for kk in range(no_meander+1):
        if kk==0:
            [poly1,newpos1] = createCPWvertUp((x0,y0+length_straight/2),\
                                     length=length_straight/2,gap=gap,width=width,layer=layer)
            polygon += poly1
        elif kk== no_meander:
            [poly2,newpos2] = createCPWvertUp((x0+kk*radius*2,y0),\
                                     length=length_straight/2,gap=gap,width=width,layer=layer)
            polygon += poly2                
        else:
            [poly3,newpos3] = createCPWvertUp((x0+kk*radius*2,y0),\
                                     length=length_straight,gap=gap,width=width,layer=layer)
            polygon += poly3
    ## draw top semicircles
    
    for kk in range(int(no_meander/2)):
        [poly4, newpos4] = createSemiarc((x0+radius*(4*kk+3),y0),radius=radius, gap=gap, width=width,angle=np.pi,layer=layer)
        polygon += poly4
        [poly5, newpos5] = createSemiarc((x0+radius*(4*kk+1),y0+length_straight),radius=radius, gap=gap, width=width,angle=0,layer=layer)
        polygon += poly5
        
    final_position = [center[0]+no_meander*2*radius,center[1]]
    return [polygon, final_position]

def CreateResonator(start,no_meander=2,meander_length=5000,hor_length=500,vert_length=1000,radius=50,gap=5,width=10,layer=1):
    polygon=[]

    [notch,pos]= createNotch(start,length=10,width=width,layer=layer)
    polygon += notch
    
    [poly1,pos1] = createCPWhorLeft(pos,length=hor_length,gap=gap,width=width,layer=layer)
    polygon += poly1
    
    [bend,pos2] = createDoublearc(center = (pos1[0],pos1[1]-radius),radius=radius,gap = gap,width=width,angle=np.pi/2,layer=layer)
    pos2 = (pos2[0]-radius,pos2[1]-radius)
    polygon += bend
    
    [poly2,pos3]= createCPWvertDown(pos2,length=vert_length,gap=gap,width=width,layer=layer)
    polygon += poly2
    
    [meander,pos4] = CreateMeanderDown(pos3,length=meander_length,no_meander=no_meander,radius= radius,gap=gap,width=width,layer=layer)
    polygon += meander
    
    return polygon
    
def CreateAlternateMeanderUp(center,length,no_meander=10,radius= 50,gap1=5,width1=10,gap2=8,width2=4, layer=1):
    if 7*(2*gap1 +width1) > 2*radius or 7*(2*gap2 +width2) > 2*radius:
        print("Warning: Distance between adjacent CTL is kept at least (2*gap +width) to reduce parasitic coupling. Please increase radius.")
    if (gap1 +width1/2) != (gap2 +width2/2):
        print("Warning: Please enter values for width and gap such that (gap1 +width1/2) = (gap2 +width2/2) to ensure connectivity.")
    polygon = []
    for i in range(no_meander): 
        if i%2 == 0:
            [poly1,pos] = CreateMeanderUp(center,length,no_meander=1,radius= radius,gap=gap1,width=width1)
            polygon += poly1
            center = pos
        else:
            [poly2,pos] = CreateMeanderUp(center,length,no_meander=1,radius= radius,gap=gap2,width=width2)
            polygon += poly2 
            center = pos
    return [polygon, pos]

def CreateAlternateMeanderDown(center,length,no_meander=10,radius= 50,gap1=5,width1=10,gap2=8,width2=4, layer=1):
    if 7*(2*gap1 +width1) > 2*radius or 7*(2*gap2 +width2) > 2*radius:
        print("Warning: Distance between adjacent CTL is kept at least (2*gap +width) to reduce parasitic coupling. Please increase radius.")
    if (gap1 +width1/2) != (gap2 +width2/2):
        print("Warning: Please enter values for width and gap such that (gap1 +width1/2) = (gap2 +width2/2) to ensure connectivity.")
    polygon = []
    for i in range(no_meander): 
        if i%2 == 0:
            [poly1,pos] = CreateMeanderDown(center,length,no_meander=1,radius= radius,gap=gap1,width=width1)
            polygon += poly1
            center = pos
        else:
            [poly2,pos] = CreateMeanderDown(center,length,no_meander=1,radius= radius,gap=gap2,width=width2)
            polygon += poly2 
            center = pos
    return [polygon, pos]

def CreateIDC(center, resonator_length=240,s=2,g=2,w=2,h=46,layer=1):
    #resonator_length=244 
    no_of_fingers = int(resonator_length/(s+g))
    polygon = []
    for i in range(no_of_fingers):
        if i%2 == 0:
            center_left = (center[0]-w/2,center[1]+g/2+(i/2)*(2*(g+s)))
            center_right = (center[0]+w/2,center[1]+g/2+(i/2)*(2*(g+s)))
            
            polygon_pts_left = [(center_left[0],center_left[1]+g/2), (center_left[0],center_left[1]-g/2), (center_left[0]-h-g,center_left[1]-g/2), (center_left[0]-h-g,center_left[1]+(g/2)+s),\
                                (center_left[0]-h,center_left[1]+(g/2)+s),(center_left[0]-h,center_left[1]+g/2)]
            polygon_pts_right = [(center_right[0],center_right[1]+g/2), (center_right[0],center_right[1]-g/2),(center_right[0]+h+g,center_right[1]-g/2), (center_right[0]+h+g,center_right[1]+(g/2)+s), \
                                (center_right[0]+h,center_right[1]+(g/2)+s),(center_right[0]+h,center_right[1]+g/2)]
            
            polygon_left = [gdspy.Polygon(polygon_pts_left,layer = layer)]
            polygon_right = [gdspy.Polygon(polygon_pts_right,layer = layer)]
            polygon += polygon_left
            polygon += polygon_right
        else:
            center_left = (center[0]-w/2 -h -g/2,center[1]+g+s+(i-1)/2*2*(g+s))
            center_right = (center[0]+w/2 +h +g/2,center[1]+g+s+(i-1)/2*2*(g+s))
            
            polygon_pts_left = [(center_left[0]-g/2,center_left[1]), (center_left[0]+g/2 +h,center_left[1]), (center_left[0]+g/2 +h,center_left[1]+s+g),\
                                (center_left[0]+g/2 +h-g,center_left[1]+s+g), (center_left[0]+g/2 +h-g,center_left[1]+g), (center_left[0]-g/2,center_left[1]+g)]
            polygon_pts_right = [(center_right[0]+g/2,center_right[1]), (center_right[0]-g/2 -h,center_right[1]), (center_right[0]-g/2 -h,center_right[1]+s+g),\
                                (center_right[0]-g/2 -h+g,center_right[1]+s+g), (center_right[0]-g/2 -h+g,center_right[1]+g), (center_right[0]+g/2,center_right[1]+g)]
                
            polygon_left = [gdspy.Polygon(polygon_pts_left,layer = layer)]
            polygon_right = [gdspy.Polygon(polygon_pts_right,layer = layer)]
            polygon += polygon_left
            polygon += polygon_right
    final_position = center[0],center[1]+no_of_fingers*(s+g)
            
    return [polygon,final_position]       


def CreateAlternateHalfMeanderUp(center,quarterwave_resonator_length = 1000,no_of_quarterwave_resonators = 4,radius =50,width1=10,gap1=5,width2=16,gap2 =2,layer =1):
    length_circle = np.pi*radius
    length_straight = quarterwave_resonator_length-length_circle
    polygon = []
    for i in range(no_of_quarterwave_resonators):
        if i%2 == 0:
            [poly1,pos1] = createCPWvertUp(center ,length=length_straight ,gap=gap1,width=width1,layer=layer)
            [poly2,pos2] = createSemiarc((pos1[0]+radius, pos1[1]), radius= radius, gap=gap1, width=width1,angle=0,layer=layer)
            polygon += poly1 + poly2
            center = pos2
        else:
            [poly1,pos_1] = createCPWvertDown(center ,length=length_straight ,gap=gap2,width=width2,layer=layer)
            [poly2,pos_2] = createSemiarc((pos_1[0]+radius, pos_1[1]), radius= radius, gap=gap2, width=width2,angle=-np.pi,layer=layer)
            pos_2 = (pos_2[0]+2*radius,pos_2[1])
            polygon += poly1 + poly2
            center = pos_2
    return [polygon, center]
 
    

def fingercapacitor_coupled_to_launcher(start=[0,0], finger_length=70, finger_count=7, finger_width=5, finger_gap=5, bar_width=5,cut_bar_width=10,layer=1):
    
    """
    Generates a finger capacitor layout using gdspy.
    
    Parameters:
        start ): Starting coordinates (x, y) of the capacitor.
        finger_length : Length of each interdigitated finger.
        finger_count : Number of fingers on one side.
        finger_width: Width of each finger.
        finger_gap: Gap between fingers of the same side.
        bar_width: Width of the vertical bar connecting the fingers.
        Hor_width : Total height (Y-direction) of the layout rectangle.
        cut_bar_width: Width of the final cut bar at the right side for connecting to next component.
      
    """
    # Create a rectangle that serves as the base area to cut from
    adjfinger_gap=2*finger_gap+finger_width   # gap between adjacent fingers pointing in same direction
    Hor_width=finger_width*finger_count+finger_gap*(finger_count-1) # width of the horizontal waveguide connected to launcher 
    polygons= [gdspy.Rectangle(
    (start[0], start[1]-Hor_width/2),
    (start[0] + finger_length + 2*bar_width +2* finger_gap, start[1] + Hor_width/2)
)]
    # Create left-side fingers and subtract from base
    for i in range(math.ceil(finger_count / 2)):
       
        y1 = start[1]-Hor_width/2 + i * (finger_width + adjfinger_gap)
        finger_polygon_1 = gdspy.Polygon([
                [start[0] +bar_width ,y1],
                [start[0] +bar_width,y1 + finger_width],
                [start[0] +bar_width+ finger_length,y1 + finger_width],
                [start[0] +bar_width+ finger_length,y1],
            ], layer=layer)
        polygons = [gdspy.fast_boolean(polygon, finger_polygon_1, 'not', layer=layer) for polygon in polygons]
    
    # Create right-side fingers and subtract    
    for j in range(finger_count-math.ceil(finger_count / 2)):
        
         y2 = start[1]- Hor_width/2 + finger_width+finger_gap+j* (finger_width + adjfinger_gap)
         finger_polygon_2 = gdspy.Polygon([
                 [start[0]+bar_width +finger_length+ finger_gap,y2],
                 [start[0]+bar_width +finger_length+ finger_gap,y2 + finger_width],
                 [start[0]+bar_width + finger_gap,y2 + finger_width],
                 [start[0]+bar_width + finger_gap,y2]
             ], layer=layer)
        
         polygons = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=layer) for polygon in polygons]
    
    # Subtract left vertical bar
    left_bar = gdspy.Polygon([
        [start[0],start[1]-Hor_width/2],
        [start[0]+bar_width,start[1]-Hor_width/2],
        [start[0]+bar_width,start[1]+Hor_width/2 ],
        [start[0],start[1]+Hor_width/2]
        ], layer=layer)
    polygons = [gdspy.fast_boolean(polygon, left_bar, 'not', layer=layer) for polygon in polygons]
    

     # Subtract right bus bar
    right_bar = gdspy.Polygon([
      [start[0] +bar_width+finger_length+finger_gap,start[1]-Hor_width/2],
      [start[0] +2*bar_width+finger_length+finger_gap,start[1]-Hor_width/2],
      [start[0] +2*bar_width+finger_length+finger_gap,start[1]+Hor_width/2],
      [start[0] +bar_width+finger_length+finger_gap,start[1]+Hor_width/2]
      ], layer=layer)
    polygons = [gdspy.fast_boolean(polygon, right_bar, 'not', layer=layer) for polygon in polygons] 
    
    # Subtract cut bar 
    cut_bar = gdspy.Rectangle(
    (start[0] + 2*bar_width +finger_length+ finger_gap,
     start[1] - cut_bar_width / 2),
    (start[0] + 2*bar_width +finger_length+2* finger_gap,
     start[1]+ cut_bar_width / 2)
)

    polygons = [gdspy.fast_boolean(polygon, cut_bar, 'not', layer=layer) for polygon in polygons]
    
    
   
    final_position = [start[0]+ finger_length + 2*bar_width +2* finger_gap, start[1]]
    
    return polygons, final_position



def fingercapacitor_with_both_end_connections(start=[0,0], finger_length=300,clearance=15, finger_count=7, finger_width=20, finger_gap=5, bar_width=50 , cut_bar_width=10, layer=1):
  
    """
    Create a finger capacitor structure using Boolean operations in GDSII.

    Parameters:
        start: Starting (x, y) coordinate for the capacitor layout.
        finger_length : Length of each interdigitated finger.
        finger_count : Number of fingers on the left side.
        finger_width : Width of each finger.
        finger_gap : gap between fingers
        bar_width : Width of the vertical bars that connect fingers.
        Hor_width : Total height (Y) of the horizontal metal strip area.
        clearance : Extra padding added to the background rectangle (for visual margin).
     
    """
    
    adjfinger_gap=2*finger_gap+finger_width   # gap between adjacent fingers pointing in same direction
    Hor_width=finger_width*finger_count+finger_gap*(finger_count-1) # width of the horizontal waveguide connected to launcher 
    # Total width of the layout (X-direction)
    total_width = finger_length + 2 * bar_width+finger_gap
    total_height = Hor_width  #y direction

    # Base rectangle (background block that we subtract fingers and bars from)
    base_rect = gdspy.Rectangle(
        (start[0] - clearance, start[1] - total_height / 2 - clearance),
        (start[0] + total_width+clearance, start[1] + total_height / 2 + clearance)
    )
    
    polygons = [base_rect]
        

      
    for i in range(math.ceil(finger_count / 2)):
       
        y1 = start[1]-Hor_width/2 + i * (finger_width + adjfinger_gap)
        finger_polygon_1 = gdspy.Polygon([
                [start[0] +bar_width ,y1],
                [start[0] +bar_width,y1 + finger_width],
                [start[0] +bar_width+ finger_length,y1 + finger_width],
                [start[0] +bar_width+ finger_length,y1],
            ], layer=layer)
        polygons = [gdspy.fast_boolean(polygon, finger_polygon_1, 'not', layer=layer) for polygon in polygons]
    
    # Create right-side fingers and subtract    
    for j in range(int(finger_count-math.ceil(finger_count / 2))):
        
         y2 = start[1]- Hor_width/2 + finger_width+finger_gap+j* (finger_width + adjfinger_gap)
         finger_polygon_2 = gdspy.Polygon([
                 [start[0]+bar_width +finger_length+ finger_gap,y2],
                 [start[0]+bar_width +finger_length+ finger_gap,y2 + finger_width],
                 [start[0]+bar_width + finger_gap,y2 + finger_width],
                 [start[0]+bar_width + finger_gap,y2]
             ], layer=layer)
        
         polygons = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=layer) for polygon in polygons]
    
   # Subtract left bus bar
    left_bar = gdspy.Polygon([
       [start[0],start[1]-Hor_width/2],
       [start[0]+bar_width,start[1]-Hor_width/2],
       [start[0]+bar_width,start[1]+Hor_width/2 ],
       [start[0],start[1]+Hor_width/2]
       ], layer=layer)
    polygons = [gdspy.fast_boolean(polygon, left_bar, 'not', layer=layer) for polygon in polygons]
   

    # Subtract right bus bar
    right_bar = gdspy.Polygon([
     [start[0] +bar_width+finger_length+finger_gap,start[1]-Hor_width/2],
     [start[0] +2*bar_width+finger_length+finger_gap,start[1]-Hor_width/2],
     [start[0] +2*bar_width+finger_length+finger_gap,start[1]+Hor_width/2],
     [start[0] +bar_width+finger_length+finger_gap,start[1]+Hor_width/2]
     ], layer=layer)
    polygons = [gdspy.fast_boolean(polygon, right_bar, 'not', layer=layer) for polygon in polygons] 
 
    # Subtract cut bar 
    cut_bar1 = gdspy.Rectangle(
    (start[0] + 2*bar_width +finger_length+ finger_gap,
     start[1] - cut_bar_width / 2),
    (start[0] + 2*bar_width +finger_length+ finger_gap+clearance,
     start[1]+ cut_bar_width / 2)
)

    polygons = [gdspy.fast_boolean(polygon, cut_bar1, 'not', layer=layer) for polygon in polygons]
    
    cut_bar2 = gdspy.Rectangle(
    (start[0] -clearance,
     start[1] - cut_bar_width / 2),
    (start[0] ,
     start[1]+ cut_bar_width / 2)
)

    polygons = [gdspy.fast_boolean(polygon, cut_bar2, 'not', layer=layer) for polygon in polygons]
    
    final_position = [start[0]+finger_length + 2*bar_width +finger_gap+clearance, start[1]]
    
    return polygons, final_position


def fingercapacitor_grounded_at_oneend(start=[0,0], finger_length=300,clearance=15, finger_count=7, finger_width=20, finger_gap=5, bar_width=50 , cut_bar_width=10, layer=1):
  
    """
    Create a finger capacitor structure using Boolean operations in GDSII.

    Parameters:
        start : Starting (x, y) coordinate for the capacitor layout.
        finger_length : Length of each interdigitated finger.
        finger_count : Number of fingers on the left side.
        finger_width: Width of each finger.
        finger_gap : gap between adjacent fingers
        bar_width : Width of the vertical bars that connect fingers.
        Hor_width : Total height (Y direction) of the metal strip area.
        clearance: Extra padding added to the background rectangle (for visual margin).
        
    """
    
    adjfinger_gap=2*finger_gap+finger_width   # gap between adjacent fingers pointing in same direction
    Hor_width=finger_width*finger_count+finger_gap*(finger_count-1) # width of the horizontal waveguide connected to launcher 
    # Total width of the layout (X-direction)
    total_width = finger_length +  bar_width+finger_gap
    total_height = Hor_width

    # Base rectangle (background block that we subtract fingers and bars from)
    base_rect = gdspy.Rectangle(
        (start[0] - clearance, start[1] - total_height / 2 - clearance),
        (start[0] + total_width, start[1] + total_height / 2 + clearance)
    )
    
    polygons = [base_rect]
        

      
    for i in range(math.ceil(finger_count / 2)):
       
        y1 = start[1]-Hor_width/2 + i * (finger_width + adjfinger_gap)
        finger_polygon_1 = gdspy.Polygon([
                [start[0] +bar_width ,y1],
                [start[0] +bar_width,y1 + finger_width],
                [start[0] +bar_width+ finger_length,y1 + finger_width],
                [start[0] +bar_width+ finger_length,y1],
            ], layer=layer)
        polygons = [gdspy.fast_boolean(polygon, finger_polygon_1, 'not', layer=layer) for polygon in polygons]
    
    # Create right-side fingers and subtract    
    for j in range(int(finger_count-math.ceil(finger_count / 2))):
        
         y2 = start[1]- Hor_width/2 + finger_width+finger_gap+j* (finger_width + adjfinger_gap)
         finger_polygon_2 = gdspy.Polygon([
                 [start[0]+bar_width +finger_length+ finger_gap,y2],
                 [start[0]+bar_width +finger_length+ finger_gap,y2 + finger_width],
                 [start[0]+bar_width + finger_gap,y2 + finger_width],
                 [start[0]+bar_width + finger_gap,y2]
             ], layer=layer)
        
         polygons = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=layer) for polygon in polygons]
    
   # Subtract left bus bar
    left_bar = gdspy.Polygon([
       [start[0],start[1]-Hor_width/2],
       [start[0]+bar_width,start[1]-Hor_width/2],
       [start[0]+bar_width,start[1]+Hor_width/2 ],
       [start[0],start[1]+Hor_width/2]
       ], layer=layer)
    polygons = [gdspy.fast_boolean(polygon, left_bar, 'not', layer=layer) for polygon in polygons]
   

    # Subtract right bus bar
    right_bar = gdspy.Polygon([
     [start[0] +bar_width+finger_length+finger_gap,start[1]-Hor_width/2],
     [start[0] +2*bar_width+finger_length+finger_gap,start[1]-Hor_width/2],
     [start[0] +2*bar_width+finger_length+finger_gap,start[1]+Hor_width/2],
     [start[0] +bar_width+finger_length+finger_gap,start[1]+Hor_width/2]
     ], layer=layer)
    polygons = [gdspy.fast_boolean(polygon, right_bar, 'not', layer=layer) for polygon in polygons] 
 
    # Subtract cut bar 
    cut_bar1 = gdspy.Rectangle(
    (start[0] + 2*bar_width +finger_length+ finger_gap,
     start[1] - Hor_width / 2),
    (start[0] + 2*bar_width +finger_length+ finger_gap+clearance,
     start[1]+ Hor_width / 2)
)

    polygons = [gdspy.fast_boolean(polygon, cut_bar1, 'not', layer=layer) for polygon in polygons]
    
    cut_bar2 = gdspy.Rectangle(
    (start[0] -clearance,
     start[1] - cut_bar_width / 2),
    (start[0] ,
     start[1]+ cut_bar_width / 2)
)

    polygons = [gdspy.fast_boolean(polygon, cut_bar2, 'not', layer=layer) for polygon in polygons]
    
    final_position = [start[0]+finger_length + bar_width +finger_gap, start[1]]
    
    return polygons, final_position





def Createlauncherup(start=[0,0],straight=200, height=200, taper = 200, gapL=50, gap=6,width=10,layer=1,notch=False):
    """
    Creates a CPW launcher structure extending upward from the starting point.

    Parameters:
        start : Starting coordinate (x, y).
        straight : Length of the vertical CPW section.
        height : Total height of the launcher (center conductor + gaps).
        taper : Length of the taper section.
        gapL : Gap between conductor and ground in the vertical section.
        gap : Final in taper section.
        width : Final taper conductor width.
        notch : Whether to add a horizontal notch at theend.

    """
    polygon = []
    # Vertical CPW segment
    [poly1,newpos] = createCPWvertUp(start,length=straight,gap=gapL,width=height-2*gapL,layer=layer)
    polygon += poly1
    # Right taper (from wide to narrow)
    polygon_pts1 = [(start[0]+height/2, start[1]+straight),
                    (start[0]+height/2-gapL, start[1]+straight),
                    (start[0]+width/2, start[1]+straight+taper),
                    (start[0]+width/2+gap, start[1]+straight+taper)
                    ]
    # Left taper (symmetric)
    polygon_pts2 = [(start[0]-height/2, start[1]+straight),
                    (start[0]-height/2+gapL, start[1]+straight),
                    (start[0]-width/2, start[1]+straight+taper),
                    (start[0]-width/2-gap, start[1]+straight+taper)
                    ]
    # Optional notch (rectangular cut at base)                ]
    polygon_pts3 = [(start[0]-height/2, start[1]-gapL),
                    (start[0]-height/2, start[1]),
                    (start[0]+height/2, start[1]),
                    (start[0]+height/2, start[1]-gapL)
                    ] 
    
    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    if notch:
        polygon += [gdspy.Polygon(polygon_pts3,layer = layer)]
     # Final position after launcher 
    final_position = [start[0],start[1]++straight+taper]    
    
    return [polygon, final_position]



def Createlauncherdown(start, straight=200, height=200, taper=200, gapL=50, gap=6, width=10, layer=1, notch=False):
    """
    Create a vertical coplanar waveguide (CPW) launcher with a downward taper.

    Parameters:
        Parameters:
            start : Starting coordinate (x, y).
            straight : Length of the vertical CPW section.
            height : Total height of the launcher (center conductor + gaps).
            taper : Length of the taper section.
            gapL : Gap between conductor and ground in the vertical section.
            gap :  Final gap in taper section.
            width : Final taper conductor width.
            notch : Whether to add a horizontal notch at the end.
    """
    polygon = []
    
    # === Create vertical CPW pad ===

    [poly1, newpos] = createCPWvertDown(start, length=straight, gap=gapL, width=height-2*gapL, layer=layer)
    polygon += poly1
    
    # === Create taper polygons from pad to narrower center line ===

    # Right side taper polygon

    polygon_pts1 = [(start[0] + height/2, start[1] - straight),
                    (start[0] + height/2 - gapL, start[1] - straight),
                    (start[0] + width/2, start[1] - straight - taper),
                    (start[0] + width/2 + gap, start[1] - straight - taper)]

    polygon_pts2 = [(start[0] - height/2, start[1] - straight),
                    (start[0] - height/2 + gapL, start[1] - straight),
                    (start[0] - width/2, start[1] - straight - taper),
                    (start[0] - width/2 - gap, start[1] - straight - taper)]
    
    polygon_pts3 = [(start[0] - height/2, start[1] + gapL),
                    (start[0] - height/2, start[1]),
                    (start[0] + height/2, start[1]),
                    (start[0] + height/2, start[1] + gapL)]
    
    polygon += [gdspy.Polygon(polygon_pts1, layer=layer)]
    polygon += [gdspy.Polygon(polygon_pts2, layer=layer)]
    
    if notch:
        polygon += [gdspy.Polygon(polygon_pts3, layer=layer)]
     # Final position is at the end of the taper
    final_position = [start[0] , start[1]- straight - taper]
    
    return [polygon, final_position]




def CreateResonatorbottom(start,no_meander=2,meander_length=5000,hor_length=500,vert_length=1000,radius=50,gap=5,width=10,layer=1):
    """
    Creates a bottom-anchored resonator structure using CPW with bends and meanders.

    Parameters:
        start: Starting point (x, y).
        no_meander : Number of meander turns.
        meander_length : Total length of meander section.
        hor_length : Horizontal CPW length before first bend.
        vert_length : Vertical CPW segment length after bend.
        radius : Radius of arcs used in bends and meander.
        gap : Gap between signal and ground.
        width : Width of CPW conductor.
        
    """
    polygon=[]
    # 1. Create the notch at the start of the resonator
    [notch,pos]= createNotch(start,length=10,width=width,layer=layer)
    polygon += notch
    
    # 2. Horizontal CPW segment to the left
    [poly1,pos1] = createCPWhorLeft(pos,length=hor_length,gap=gap,width=width,layer=layer)
    polygon += poly1
    
    # 3. 90-degree bend downward using  arc
    [bend,pos2] = createDoublearc(center = (pos1[0],pos1[1]-radius),radius=radius,gap = gap,width=width,angle=np.pi/2,layer=layer)
    # Adjust final bend position for vertical transition
    pos2 = (pos2[0]-radius,pos2[1]-radius)
    polygon += bend
    # 4. Vertical CPW segment downward
    [poly2,pos3]= createCPWvertDown(pos2,length=vert_length,gap=gap,width=width,layer=layer)
    polygon += poly2
    # 5. Add meandered CPW structure
    [meander,pos4] = CreateMeanderDown(pos3,length=meander_length,no_meander=no_meander,radius= radius,gap=gap,width=width,layer=layer)
    polygon += meander
    
    return polygon



def CreateResonatortop(start,no_meander=2,meander_length=5000,hor_length=500,vert_length=1000,radius=50,gap=5,width=10,layer=1):
    
    """
    Creates a top-anchored CPW resonator with horizontal entry, upward bend, and meander.

    Parameters:
        start: Starting point (x, y).
        no_meander : Number of meander turns.
        meander_length : Total length of meander section.
        hor_length : Horizontal CPW length before first bend.
        vert_length : Vertical CPW segment length after bend.
        radius : Radius of arcs used in bends and meander.
        gap : Gap between signal and ground.
        width : Width of CPW conductor.
    """
    polygon=[]
    # 1. Create the notch at the start of the resonator
    [notch,pos]= createNotch(start,length=10,width=width,layer=layer)
    polygon += notch
    # 2. Horizontal CPW line going left
    [poly1,pos1] = createCPWhorLeft(pos,length=hor_length,gap=gap,width=width,layer=layer)
    polygon += poly1
    # 3. Bend upward using a double arc 
    [bend,pos2] = createDoublearc(center = (pos1[0],pos1[1]+radius),radius=radius,gap = gap,width=width,angle=-np.pi,layer=layer)
    pos2 = (pos2[0],pos2[1])
    polygon += bend
    # 4. Vertical CPW going up
    [poly2,pos3]= createCPWvertUp(pos2,length=vert_length,gap=gap,width=width,layer=layer)
    polygon += poly2
    # 5. Add upward meander section
    [meander,pos4] = CreateMeanderUp(pos3,length=meander_length,no_meander=no_meander,radius= radius,gap=gap,width=width,layer=layer)
    polygon += meander
    
    return polygon




def loop_down(start,addlength=100,length1=100,radius=600,gap=5,width=10,layer=1):
    """
    Creates a loop structure starting from 'start' coordinate.
    
    Parameters:
        start : Starting point (x, y).
        addlength : Vertical length before and after loops.
        length1 : Vertical segment length inside the loop.
        radius : Radius of arcs in the loop.
        gap : Gap between CPW lines.
        width : Width of the CPW center conductor.
        
    """
    polygon=[]
    # Vertical line down of length 'addlength'
    [poly1a,pos1a]=createCPWvertDown(start,length=addlength,gap=gap,width=width,layer=layer)
    polygon+=poly1a
    # First arc bending left and down
    [poly1,pos1]=createDoublearc((pos1a[0]-radius,pos1a[1]),radius=radius,gap = gap,width=width,angle=np.pi*3/2,layer=layer)
    pos1 = (pos1[0]-radius,pos1[1]-radius)
    polygon += poly1
    # Second arc bending right and down 
    [poly2,pos2]=createDoublearc((pos1[0]+radius,pos1[1]),radius=radius,gap = gap,width=width,angle=np.pi/2,layer=layer)
    pos2 = (pos2[0]-radius,pos2[1]-radius)
    polygon += poly2
    # Vertical line down of length 'length1'
    [poly3,pos3]= createCPWvertDown(pos2,length=length1,gap=gap,width=width,layer=layer)
    polygon += poly3
    # Arc bending left 
    [poly4,pos4]=createDoublearc((pos3[0]+radius,pos3[1]),radius=radius,gap = gap,width=width,angle=np.pi,layer=layer)
    pos4 = (pos4[0]+radius,pos4[1]-radius)
    polygon += poly4
    # Horizontal line to the left, length = 4 * radius
    [poly5,pos5]= createCPWhor(pos4,length=4*radius,gap = gap,width= width,layer=layer)
    polygon += poly5
    # Arc bending up and right 
    [poly6,pos6]=createDoublearc((pos5[0],pos5[1]+radius),radius=radius,gap = gap,width=width,angle=np.pi*3/2,layer=layer)
    pos6 = (pos6[0]+radius,pos6[1])
    polygon += poly6
    # Vertical line up of length 'length1'
    [poly7,pos7]= createCPWvertUp((pos6[0],pos6[1]+radius),length=length1,gap=gap,width=width,layer=layer)
    polygon += poly7
    # Arc bending right 
    [poly8,pos8]=createDoublearc((pos7[0]-radius,pos7[1]),radius=radius,gap = gap,width=width,angle=np.pi*2,layer=layer)
    pos8 = (pos8[0],pos8[1]+radius)
    polygon += poly8
    # Arc bending left 
    [poly9,pos9]=createDoublearc((pos8[0]-radius,pos8[1]+radius),radius=radius,gap = gap,width=width,angle=np.pi,layer=layer)
    polygon += poly9
    # Final vertical line up of length 'addlength'
    [poly10,pos10]=createCPWvertUp(pos9,length=addlength,gap=gap,width=width,layer=layer)
    polygon+=poly10
    final_position=pos10
    
    
    return [polygon,final_position]



def loop_up(start,addlength=100,length1=100,radius=600,gap=5,width=10,layer=1):
    """
    Creates an upward loop structure starting from 'start' coordinate.

    Parameters:
        start : Starting point (x, y).
        addlength : Vertical length before and after loops.
        length1 : Vertical segment length inside the loop.
        radius : Radius of arcs in the loop.
        gap : Gap between CPW lines.
        width : Width of the CPW center conductor.
    """
    polygon=[]
    # Vertical line up of length 'addlength'
    [poly1a,pos1a]=createCPWvertUp(start,length=addlength,gap=gap,width=width,layer=layer)
    polygon+=poly1a
    # First double arc  positioned left and up
    [poly1,pos1]=createDoublearc((pos1a[0]-radius,pos1a[1]),radius=radius,gap = gap,width=width,angle=np.pi*2,layer=layer)
    pos1=(pos1[0]-radius,pos1[1]+radius)
    polygon += poly1
    # Second double arc
    [poly2,pos2]=createDoublearc((pos1[0],pos1[1]+radius),radius=radius,gap = gap,width=width,angle=-np.pi,layer=layer)
    polygon += poly2
    # Vertical line up of length 'length1'
    [poly3,pos3]= createCPWvertUp(pos2,length=length1,gap=gap,width=width,layer=layer)
    polygon += poly3
    # Double arc bending right
    [poly4,pos4]=createDoublearc((pos3[0]+radius,pos3[1]),radius=radius,gap = gap,width=width,angle=np.pi/2,layer=layer)
    polygon += poly4
    # Horizontal line right, length = 4 * radius
    [poly5,pos5]= createCPWhor(pos4,length=4*radius,gap = gap,width= width,layer=layer)
    polygon += poly5
    # Double arc  positioned below
    [poly6,pos6]=createDoublearc((pos5[0],pos5[1]-radius),radius=radius,gap = gap,width=width,angle=np.pi*2,layer=layer)
    polygon += poly6
    # Vertical line down of length 'length1'
    [poly7,pos7]= createCPWvertDown(pos6,length=length1,gap=gap,width=width,layer=layer)
    polygon += poly7
    # Double arc bending left and down 
    [poly8,pos8]=createDoublearc((pos7[0]-radius,pos7[1]),radius=radius,gap = gap,width=width,angle=np.pi*3/2,layer=layer)
    pos8 = (pos8[0]+radius,pos8[1])
    polygon += poly8
    # Double arc bending left and down
    [poly9,pos9]=createDoublearc((pos8[0]-radius,pos8[1]-radius),radius=radius,gap = gap,width=width,angle=np.pi/2,layer=layer)
    pos9 = (pos9[0]-radius,pos9[1]-radius)
    polygon += poly9
    # Final vertical line down of length 'addlength'
    [poly10,pos10]=createCPWvertDown(pos9,length=addlength,gap=gap,width=width,layer=layer)
    polygon+=poly10
    final_position=pos10
    return [polygon,final_position]


def aysmmetric_meander_left_both_ends_straight(start,vertical_length1=500,vertical_length2=500,vertical_length4=1000,radius=600,gap1=10,width1=90,gap2=52.5,width2=5,layer=1):
    """
    Generates an asymmetric meander structure with both ends as straight vertical CPW (coplanar waveguide) segments.
    Parameters:
        start : Starting coordinate (x, y).
        vertical_length1 : Vertical length of the narrow CPW sections.
        vertical_length2 : Vertical length of the wider CPW sections.
        vertical_length4: Final vertical extension length.
        radius : Radius of semi-arcs.
        gap1 : Gap for narrow CPW sections.
        width1 : Width of narrow CPW sections.
        gap2 : Gap for wider CPW sections.
        width2 : Width of wider CPW sections.
        
    """
    polygon= []
    # First vertical down segment (narrow CPW) - longest straight leg
    [poly1,pos1]= createCPWvertDown(start,length=vertical_length4,gap=gap1,width=width1,layer=layer)
    polygon += poly1
    # Second vertical down segment (narrow CPW)
    [poly2,pos2]= createCPWvertDown(pos1,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly2
    # First semicircular arc to reverse direction
    [poly3,pos3]=createSemiarc((pos2[0]+radius,pos2[1]),radius=radius,gap=gap1,width=width1,angle=np.pi,layer=layer)
    polygon +=poly3
    # Vertical up segment after the first arc  (narrow CPW)
    [poly4,pos4]=createCPWvertUp((pos3[0]+2*radius,pos3[1]),length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon+=poly4
    # Next vertical up segment (wider CPW)
    [poly5,pos5]=createCPWvertUp(pos4,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon+=poly5
    #  semicircular arc (180 degrees, wider CPW) for meander bend
    [poly6,pos6]=createSemiarc((pos5[0]+radius,pos5[1]),radius=radius,gap=gap2,width=width2,angle=np.pi*2,layer=layer)
    polygon +=poly6
    # Final vertical down segment (wider CPW)
    [poly7,pos7]= createCPWvertDown(pos6,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon += poly7
    
    final_position= pos7
    
    return [polygon,final_position]

def aysmmetric_meander_left_one_end_horizontal(start,vertical_length1=500,vertical_length2=1500,vertical_length4=1000,vertical_length5=200,radius=200,gap1=10,width1=90,gap2=52.5,width2=5,layer=1):
    
    """
    The structure begins at a starting point and creates a series of vertical segments,
    arcs, and ends with a horizontal segment. One end of the structure is a vertical straight line,
    and the other ends horizontally after a meander and arc.

    Parameters:
        start : Starting coordinate (x, y).
        vertical_length1 : Vertical length of the narrow CPW sections.
        vertical_length2 : Vertical length of the wider CPW sections.
        vertical_length4: Final vertical extension length.
        radius : Radius of semi-arcs.
        gap1 : Gap for narrow CPW sections.
        width1 : Width of narrow CPW sections.
        gap2 : Gap for wider CPW sections.
        width2 : Width of wider CPW sections.
        
    """
    polygon= []
    # Initial long vertical downward segment (narrow CPW)
    [poly1,pos1]= createCPWvertDown(start,length=vertical_length4,gap=gap1,width=width1,layer=layer)
    polygon += poly1
    # Additional downward segment (narrow CPW)
    [poly2,pos2]= createCPWvertDown(pos1,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly2
    # First semicircular arc to reverse direction (narrow CPW)
    [poly3,pos3]=createSemiarc((pos2[0]+radius,pos2[1]),radius=radius,gap=gap1,width=width1,angle=np.pi,layer=layer)
    polygon +=poly3
    # Move upward (narrow CPW)
    [poly4,pos4]=createCPWvertUp((pos3[0]+2*radius,pos3[1]),length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon+=poly4
    # Additional vertical up segment (wider CPW)
    [poly5,pos5]=createCPWvertUp(pos4,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon+=poly5
    # semi circular arc for a loop (wide CPW)
    [poly6,pos6]=createSemiarc((pos5[0]+radius,pos5[1]),radius=radius,gap=gap2,width=width2,angle=np.pi*2,layer=layer)
    polygon +=poly6
    #Downward leg (wide CPW), length derived to symmetrize the meander
    [poly7,pos7]= createCPWvertDown(pos6,length=(((vertical_length1+vertical_length2)/2)-radius),gap=gap2,width=width2,layer=layer)
    polygon += poly7
    # A double arc to transition from vertical to horizontal (wide CPW)
    [poly8,pos8]=createDoublearc((pos7[0]+radius,pos7[1]),radius=radius,gap = gap2,width=width2,angle=-np.pi,layer=layer)
    # Adjust position after double arc
    pos8 = (pos8[0]+radius,pos8[1]-radius)
    polygon += poly8
    # Final horizontal segment (wide CPW)
    [poly9,pos9]=createCPWhor(pos8, length=abs(vertical_length2-(((vertical_length1+vertical_length2)/2)-radius)-np.pi*radius/2),gap = gap2,width=width2,layer=layer)
    polygon+= poly9
    
    final_position= pos9
    
    return [polygon,final_position]

def Braggmirror_unitcell1_left(start,addlength=500,length1=100,vertical_length1=1629,vertical_length2=2343,vertical_length3=915,vertical_length4=1428,radius=600,gap1=5,gap2=10,gap=10,width=20,width1=5,width2=10,layer=1):
    """
   Constructs a Bragg mirror unit cell using a combination of vertical CPWs,
    loops, and an asymmetric meander section with both ends straight.

    Parameters:
        start: Starting coordinate (x, y).
        addlength: Additional length parameter for loops.
        length1 : Length parameter used in loops.
        vertical_length1, vertical_length2, vertical_length3, vertical_length4, vertical_length5 (float): Vertical segment lengths.
        radius : Radius for arcs.
        gap1, gap2 : Gap sizes for different CPW widths.
        width1, width2 : Widths for CPW sections.
    """
    polygon= []
    # First vertical downward CPW segment (narrow CPW)
    [poly1,pos1]= createCPWvertDown(start,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly1
    # Loop that curves downward from pos1 (narrow CPW)
    [poly2,pos2]=loop_down(pos1,addlength=addlength,length1=length1,radius=radius,gap=gap1,width=width1,layer=layer)
    polygon +=poly2
    # Vertical upward CPW segment to return from the loop (narrow CPW)
    [poly3,pos3]= createCPWvertUp(pos2,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly3
    # Vertical upward CPW segment (wider CPW)
    [poly4,pos4]= createCPWvertUp(pos3,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon += poly4
    # Loop that curves upward from pos4 (wider CPW)
    [poly5,pos5]=loop_up(pos4,addlength=addlength,length1=length1,radius=radius,gap=gap2,width=width2,layer=layer)
    polygon += poly5
    # Vertical downward CPW segment to return from loop (wider CPW)
    [poly6,pos6]= createCPWvertDown(pos5,length=vertical_length3,gap=gap2,width=width2,layer=layer)
    polygon += poly6
    # Asymmetric meander with both ends as straight vertical CPWs
    [poly7,pos7]=aysmmetric_meander_left_both_ends_straight(pos6,vertical_length1=vertical_length1,vertical_length2=vertical_length2,vertical_length4=vertical_length4,radius= radius,gap1=gap1,width1=width1,gap2=gap2,width2=width2,layer=layer)
    polygon+=poly7
    
    final_position= pos7
    
    return [polygon,final_position]

def Braggmirror_unitcell2_left(start,addlength=500,length1=100,vertical_length1=1629,vertical_length2=2343,vertical_length3=915,vertical_length4=1428,vertical_length5=200,meander_length=2000,radius=600,gap1=10,width1=90,gap2=52.5,width2=5,layer=1):
    """
   Constructs a Bragg mirror unit cell with an asymmetric meander
    where one end transitions to a horizontal segment.

    Parameters:
        start: Starting coordinate (x, y).
        addlength: Additional length parameter for loops.
        length1 : Length parameter used in loops.
        vertical_length1, vertical_length2, vertical_length3, vertical_length4, vertical_length5 (float): Vertical segment lengths.
        radius : Radius for arcs.
        gap1, gap2 : Gap sizes for different CPW widths.
        width1, width2 : Widths for CPW sections.
    """
    polygon= []
    # First narrow vertical downward CPW segment
    [poly1,pos1]= createCPWvertDown(start,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly1
    # Loop downward using narrow CPW parameters
    [poly2,pos2]=loop_down(pos1,addlength=addlength,length1=length1,radius=radius,gap=gap1,width=width1,layer=layer)
    polygon +=poly2
    # Vertical upward narrow CPW segment to return from loop
    [poly3,pos3]= createCPWvertUp(pos2,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly3
    # Vertical upward wider CPW segment
    [poly4,pos4]= createCPWvertUp(pos3,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon += poly4
    # Loop upward using wider CPW parameters
    [poly5,pos5]=loop_up(pos4,addlength=addlength,length1=length1,radius=radius,gap=gap2,width=width2,layer=layer)
    polygon += poly5
    # Vertical downward wider CPW segment after loop
    [poly6,pos6]= createCPWvertDown(pos5,length=vertical_length3,gap=gap2,width=width2,layer=layer)
    polygon += poly6
    # Asymmetric meander with one end horizontal to finish the structure
    [poly7,pos7]=aysmmetric_meander_left_one_end_horizontal(pos6,vertical_length1=vertical_length1,vertical_length2=vertical_length2,vertical_length4=vertical_length4,radius= radius,gap1=gap1,width1=width1,gap2=gap2,width2=width2,layer=layer)
    polygon+=poly7
    
    final_position= pos7
    
    return [polygon,final_position]




def aysmmetric_meander_right_both_ends_straight(start,vertical_length1=500,vertical_length2=500,vertical_length4=1000,radius=600,gap1=10,width1=90,gap2=52.5,width2=5,layer=1):
    
    """
    Generates an asymmetric meander structure oriented to the right, with both ends
    consisting of straight vertical CPW (coplanar waveguide) segments.

    Parameters:
        start : Starting coordinate (x, y).
        vertical_length1 : Vertical length of the narrow CPW sections.
        vertical_length2 : Vertical length of the wider CPW sections.
        vertical_length4: Final vertical extension length.
        radius : Radius of semi-arcs.
        gap1 : Gap for narrow CPW sections.
        width1 : Width of narrow CPW sections.
        gap2 : Gap for wider CPW sections.
        width2 : Width of wider CPW sections.
    """
    polygon= []
    # 1. First vertical downward segment with wider CPW parameters
    [poly1,pos1]= createCPWvertDown(start,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon += poly1
    # 2. Semicircular arc (180 degrees) to reverse direction, using wider CPW parameters
    [poly2,pos2]=createSemiarc((pos1[0]+radius,pos1[1]),radius=radius,gap=gap2,width=width2,angle=np.pi,layer=layer)
    polygon +=poly2
    # 3. Vertical upward segment with wider CPW parameters
    [poly3,pos3]=createCPWvertUp((pos2[0]+2*radius,pos2[1]),length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon+=poly3
    # 4. Vertical upward segment with narrow CPW parameters (transitioning to narrower section)
    [poly4,pos4]=createCPWvertUp(pos3,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon+=poly4
    # 5. semi circular arc (180 degrees) loop using narrow CPW parameters for meander
    [poly5,pos5]=createSemiarc((pos4[0]+radius,pos4[1]),radius=radius,gap=gap1,width=width1,angle=np.pi*2,layer=layer)
    polygon +=poly5
    # 6. Vertical downward segment with narrow CPW parameters after loop
    [poly6,pos6]= createCPWvertDown(pos5,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly6
    # 7. Final vertical downward extension with narrow CPW parameters
    [poly7,pos7]= createCPWvertDown(pos6,length=vertical_length4,gap=gap1,width=width1,layer=layer)
    polygon += poly7
    final_position= pos7
 
    
    return [polygon,final_position]





def Braggmirror_unitcell1_right(start,addlength=500,length1=100,vertical_length1=1629,vertical_length2=2343,vertical_length3=915,vertical_length4=1428,meander_length=2000,radius=600,gap1=10,width1=90,gap2=52.5,width2=5,layer=1):
    
    """
    Constructs a Bragg mirror unit cell oriented to the right side, combining an asymmetric
    meander structure with vertical CPW segments and loops.

    
       Parameters:
           start : Starting coordinate (x, y).
           vertical_length1 : Vertical length of the narrow CPW sections.
           vertical_length2 : Vertical length of the wider CPW sections.
           vertical_length4: Final vertical extension length.
           radius : Radius of semi-arcs.
           gap1 : Gap for narrow CPW sections.
           width1 : Width of narrow CPW sections.
           gap2 : Gap for wider CPW sections.
           width2 : Width of wider CPW sections.
    """
    polygon= []
    # 1. Create asymmetric meander structure with both ends straight oriented right
    [poly1,pos1]= aysmmetric_meander_right_both_ends_straight(start,vertical_length1=vertical_length1,vertical_length2=vertical_length2,vertical_length4=vertical_length4,gap1=gap1,width1=width1,gap2=gap2,width2=width2,radius=radius,layer=layer)
    polygon += poly1
    # 2. Vertical downward segment (wider CPW) after meander
    [poly2,pos2]= createCPWvertDown(pos1,length=vertical_length3,gap=gap2,width=width2,layer=layer)
    polygon += poly2
    # 3. Loop going downward (wider CPW) starting at pos2
    [poly4,pos4]=loop_down(pos2,addlength=addlength,length1=length1,radius=radius,gap=gap2,width=width2,layer=layer)
    polygon +=poly4
    # 4. Vertical upward segment (wider CPW) after downward loop
    [poly5,pos5]= createCPWvertUp(pos4,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon += poly5
    # 5. Vertical upward segment (narrow CPW)
    [poly6,pos6]= createCPWvertUp(pos5,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly6
    # 6. Loop going upward (narrow CPW)
    [poly7,pos7]=loop_up(pos6,addlength=addlength,length1=length1,radius=radius,gap=gap1,width=width1,layer=layer)
    polygon += poly7
    # 7. Final vertical downward segment (narrow CPW) after upward loop
    [poly8,pos8]= createCPWvertDown(pos7,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly8
   

    
    final_position= pos8
    
    return [polygon,final_position]

def aysmmetric_meander_right_one_end_horizontal(start,vertical_length1=500,vertical_length2=1500,vertical_length4=1000,vertical_length5=200,radius=200,gap1=10,width1=90,gap2=52.5,width2=5,layer=1):
    
    """
   Creates an asymmetric meander structure oriented to the right, where one end is horizontal and the other end is vertical.
   
   The structure starts with a horizontal CPW segment, followed by arcs and vertical CPW segments of varying widths,
   ending with a long vertical segment.

  
      Parameters:
          start : Starting coordinate (x, y).
          vertical_length1 : Vertical length of the narrow CPW sections.
          vertical_length2 : Vertical length of the wider CPW sections.
          vertical_length4: Final vertical extension length.
          radius : Radius of semi-arcs.
          gap1 : Gap for narrow CPW sections.
          width1 : Width of narrow CPW sections.
          gap2 : Gap for wider CPW sections.
          width2 : Width of wider CPW sections.
   """
    polygon= []
    # 1. Horizontal CPW segment (wide CPW) starting at `start`
    #    Length adjusted by subtracting vertical_length5 and arc length to fit overall geometry
    [poly1,pos1]=createCPWhor(start,length=abs(vertical_length2-vertical_length5-np.pi*radius/2),gap=gap2,width=width2,layer=layer)
    polygon+=poly1
    # 2. Double arc (wide CPW) transitioning from horizontal to vertical downwards
    [poly2,pos2]=createDoublearc((pos1[0],pos1[1]-radius),radius=radius,gap = gap2,width=width2,angle=np.pi*2,layer=layer)
    polygon += poly2
    # 3. Short vertical downward segment (wide CPW)
    [poly3,pos3]= createCPWvertDown(pos2,length=vertical_length5,gap=gap2,width=width2,layer=layer)
    polygon += poly3
    # 4. Semicircular arc (wide CPW) to reverse direction upward
    [poly4,pos4]=createSemiarc((pos3[0]+radius,pos3[1]),radius=radius,gap=gap2,width=width2,angle=np.pi,layer=layer)
    polygon +=poly4
    # 5. Vertical upward segment (wide CPW)
    [poly5,pos5]=createCPWvertUp((pos4[0]+2*radius,pos4[1]),length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon+=poly5
    # 6. Vertical upward segment (narrow CPW)
    [poly6,pos6]=createCPWvertUp(pos5,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon+=poly6
    # 7. Full circular arc (narrow CPW) forming a loop at the top
    [poly7,pos7]=createSemiarc((pos6[0]+radius,pos6[1]),radius=radius,gap=gap1,width=width1,angle=np.pi*2,layer=layer)
    polygon +=poly7
    # 8. Vertical downward segment (narrow CPW)
    [poly8,pos8]= createCPWvertDown(pos7,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly8
    # 9. Final long vertical downward segment (narrow CPW)
    [poly9,pos9]= createCPWvertDown(pos8,length=vertical_length4,gap=gap1,width=width1,layer=layer)
    polygon += poly9
    final_position= pos9
    
    return [polygon,final_position]




def Braggmirror_unitcell2_right(start,addlength=500,length1=100,vertical_length1=1629,vertical_length2=2343,vertical_length3=915,vertical_length4=1428,vertical_length5=200,meander_length=2000,radius=600,gap1=10,width1=90,gap2=52.5,width2=5,layer=1):
    
    """
   Constructs the right half of a Bragg mirror unit cell combining asymmetric meander segments,
   loops, and vertical coplanar waveguide (CPW) sections with varying widths and gaps.
   
   The structure begins with an asymmetric meander that ends horizontally on one side,
   followed by vertical segments and loops creating a complex meander-like pattern.
   
   Parameters:
       start : Starting coordinate (x, y).
       vertical_length1 : Vertical length of the narrow CPW sections.
       vertical_length2 : Vertical length of the wider CPW sections.
       vertical_length4: Final vertical extension length.
       radius : Radius of semi-arcs.
       gap1 : Gap for narrow CPW sections.
       width1 : Width of narrow CPW sections.
       gap2 : Gap for wider CPW sections.
       width2 : Width of wider CPW sections.
   """
    polygon= []
    # 1. Start with an asymmetric meander on the right, ending horizontally
    [poly1,pos1]= aysmmetric_meander_right_one_end_horizontal(start,vertical_length1=vertical_length1,vertical_length2=vertical_length2,vertical_length4=vertical_length4,vertical_length5=vertical_length5,gap1=gap1,width1=width1,gap2=gap2,width2=width2,radius=radius,layer=layer)
    polygon += poly1
    # 2. Add a vertical downward CPW segment (wide section)
    [poly2,pos2]= createCPWvertDown(pos1,length=vertical_length3,gap=gap2,width=width2,layer=layer)
    polygon += poly2
    # 3. Add a loop downward (wide CPW)
    [poly3,pos3]=loop_down(pos2,addlength=addlength,length1=length1,radius=radius,gap=gap2,width=width2,layer=layer)
    polygon +=poly3
    # 4. Add a vertical upward CPW segment (wide section)
    [poly4,pos4]= createCPWvertUp(pos3,length=vertical_length2,gap=gap2,width=width2,layer=layer)
    polygon += poly4
    # 5. Add a vertical upward CPW segment (narrow section)
    [poly5,pos5]= createCPWvertUp(pos4,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly5
    # 6. Add a loop upward (narrow CPW)
    [poly6,pos6]=loop_up(pos5,addlength=addlength,length1=length1,radius=radius,gap=gap1,width=width1,layer=layer)
    polygon += poly6
    # 7. Finish with a vertical downward CPW segment (narrow section)
    [poly7,pos7]= createCPWvertDown(pos6,length=vertical_length1,gap=gap1,width=width1,layer=layer)
    polygon += poly7
    final_position=pos7
    return [polygon,final_position]





def slantvertical(start,gap1=3,width1=12,gap2=5,width2=8,vertical_length=15,layer=1):
    
    """
    Create a slant transition polygon from narrower to wider CPW dimensions going upward.

    Parameters:
        start: Starting (x, y) coordinate of the transition.
        vertical_length: Vertical distance of the slant upward.
        gap1: Gap size for the narrower CPW (at start y).
        width1: Width for the narrower CPW (at start y).
        gap2: Gap size for the wider CPW (at higher y).
        width2: Width for the wider CPW (at higher y).
        layer: GDSII layer number.

    Returns:
        polygon: List of two gdspy.Polygon objects representing the tapered transition.
        final_position: Tuple with the coordinate at the top of the slant.
    """
    polygon= []
    polygon_pts1 = [(start[0]-(width1+2*gap1)/2, start[1]),
                    (start[0]-(width2+2*gap2)/2, start[1]+vertical_length),
                    (start[0]-(width2/2), start[1]+vertical_length),
                    (start[0]-width1/2, start[1])
    ]
    polygon_pts2 = [(start[0]+(width1+2*gap1)/2, start[1]),
                    (start[0]+(width2+2*gap2)/2, start[1]+15),
                     (start[0]+(width2/2), start[1]+15),
                     (start[0]+width1/2, start[1])
    ]
    

    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    
    final_position = [start[0],start[1]+vertical_length]
    
    return [polygon, final_position]



def slantright(start,gap1=30,width1=100,width2=8,gap2=21,horizontal_length=30,layer=1):
    
    """
    Create a horizontal slant transition polygon from wider to narrower CPW dimensions going right.

    Parameters:
        start: Starting (x, y) coordinate of the transition.
        horizontal_offset: Horizontal distance of the slant to the right.
        gap1: Gap size for the wider CPW (at start x).
        width1: Width for the wider CPW (at start x).
        width2: Width for the narrower CPW (at higher x).
        gap2: Gap size for the narrower CPW (at higher x).
        layer: GDSII layer number.

    Returns:
        polygon: List of two gdspy.Polygon objects representing the tapered transition.
        final_position: Tuple with the coordinate at the end of the slant.
    """
    polygon= []
    polygon_pts1 = [(start[0], start[1]+width1/2+gap1),
                    (start[0], start[1]+width1/2),
                    (start[0]+horizontal_length, start[1]+width2/2),
                    (start[0]+horizontal_length, start[1]+width2/2+gap2)
    ]
    polygon_pts2 = [(start[0], start[1]-width1/2-gap1),
                    (start[0], start[1]-width1/2),
                    (start[0]+horizontal_length, start[1]-width2/2),
                    (start[0]+horizontal_length, start[1]-width2/2-gap2)
    ]
    

    polygon += [gdspy.Polygon(polygon_pts1,layer = layer)]
    polygon += [gdspy.Polygon(polygon_pts2,layer = layer)]
    
    final_position = [start[0]+horizontal_length,start[1]]
    
    return [polygon, final_position]



def createTup(start=[0,0], straight=200, T_length=200,gap1=10, gap=5,width=10,width1=60,layer=1):
    """
    Create a T-shaped CPW (coplanar waveguide) structure starting from 'start' and going up

    Parameters:
        start: Tuple or list of (x, y) coordinates - starting point.
        straight: Length of the vertical center conductor segment going up
        T_length: length or arm
        gap1= gap of the vertical center conductor.
        gap: Gap of narrow horizontal CPW arms (left and right)
        width: Width of the narrow horizontal CPW arms (left and right).
        width1: Width of the vertical center conductor.
        layer: GDSII layer number.

    Returns:
        polygon: List of gdspy.Polygon objects representing the T-structure.
        final_position: Tuple of (x, y) coordinates at the end of the structure.
    """
    polygon = []
    
    # Vertical center conductor going up from 'start'
    [poly1,pos1] = createCPWvertUp(start,length=straight,gap=gap1,width=width1,layer=layer)
    polygon += poly1
    
    # Right horizontal arm starting from right edge of vertical line
    [poly2,pos2] = createCPWhor((pos1[0]+width1/2,pos1[1]+width/2+gap),length=T_length+gap,gap=gap,width=width,layer=layer)
    polygon += poly2
    
    # Left horizontal arm starting from left edge of vertical line
    [poly3,pos3] = createCPWhorLeft((pos1[0]-width1/2,pos1[1]+width/2+gap),length=T_length+gap,gap=gap,width=width,layer=layer)
    polygon += poly3
    
    # Top horizontal bridging polygon connecting the two arms
    polygon_pts = [(start[0]-width1/2, start[1]+gap+width+straight),
                   (start[0]+width1/2,start[1]+gap+width+straight),
                   (start[0]+width1/2,start[1]+2*gap+width+straight),
                   
                   
                   (start[0]-width1/2,start[1]+2*gap+width+straight)]

   
    polygon += [gdspy.Polygon(polygon_pts,layer = layer)]
    
    final_position1 = [start[0]+T_length+width1/2,start[1]+straight+gap+width/2] 
    final_position2 = [start[0]-T_length-width1/2,start[1]+straight+gap+width/2 ] 
   
    return [polygon, final_position1,final_position2]



def createTdown(start, straight=200, T_length=200, gap1=10, gap=5,width=10,width1=60,layer=1):
    """
    Create an inverted T-shaped CPW structure starting from 'start', going down.

    Parameters:
        start: Tuple or list of (x, y) coordinates - starting point.
        straight: Length of the vertical center conductor segment going down.
        T_length: length or arm
        gap1= gap of the vertical center conductor.
        gap: Gap of narrow horizontal CPW arms (left and right)
        width: Width of the narrow horizontal CPW arms (left and right).
        width1: Width of the vertical center conductor.
        layer: GDSII layer number.

    Returns:
        polygon: List of gdspy.Polygon objects representing the inverted T-structure.
        final_position: Tuple of (x, y) coordinates at the end of the structure.
    """
    polygon = []
    
    # Vertical center conductor going down from 'start'
    [poly1,pos1] = createCPWvertDown(start,length=straight,gap=gap1,width=width1,layer=layer)
    polygon += poly1
    
    # Right horizontal arm starting from right edge of vertical line
    [poly2,pos2] = createCPWhor((pos1[0]+width1/2,pos1[1]-width/2-gap),length=T_length+gap,gap=gap,width=width,layer=layer)
    polygon += poly2
    # Left horizontal arm starting from left edge of vertical line 
    [poly3,newpos3] = createCPWhorLeft((pos1[0]-width1/2,pos1[1]-width/2-gap),length=T_length+gap,gap=gap,width=width,layer=layer)
    polygon += poly3
   
    # Bottom horizontal bridging polygon connecting the two arms
    polygon_pts = [(start[0]-width1/2, start[1]-gap-width-straight),
                   (start[0]+width1/2,start[1]-gap-width-straight),
                 (start[0]+width1/2,start[1]-2*gap-width-straight),
                 (start[0]-width1/2,start[1]-2*gap-width-straight)]
    
    

    polygon += [gdspy.Polygon(polygon_pts,layer = layer)]
    # Final position is at the right end of the structure, same y-level as start adjusted by height (usually negative here)
    final_position1 = [start[0]+T_length+width1/2,start[1]-straight+gap+width/2] 
    final_position2 = [start[0]-T_length-width1/2,start[1]-straight+gap+width/2 ] 
   
    return [polygon, final_position1,final_position2]






def create_rectangle_hor_centered(start=[0,0], length=100, width=50, layer=1):
    """
    Create a horizontal rectangle centered on the x-axis.

    Parameters:
        start (tuple): Starting point (x, y) at the bottom center of the rectangle.
        length (float): Total horizontal length of the rectangle.
        width (float): Vertical height (thickness) of the rectangle.
        layer (int): GDSII layer number to assign the shape.

    Returns:
        tuple:
            - List containing a gdspy.Rectangle object.
            - Final (x, y) position at the top center of the rectangle.
    """
    polygons = []

    # Define lower-left and upper-right corners based on center x-alignment
    lower_left = (start[0] - length / 2, start[1])
    upper_right = (start[0] + length / 2, start[1] + width)

    # Create the rectangle on the specified layer
    
    rectangle = gdspy.Rectangle(lower_left, upper_right, layer=layer)
    polygons.append(rectangle)

    # Final position: top-center of the rectangle
    
    final_position = (start[0], start[1] + width)

    return polygons, final_position



def create_rectangle_vert_centered(start=[0,0], length=100, width=50, layer=1):
    """
    Create a vertical rectangle centered on the y-axis.

    Parameters:
        start (tuple): Starting point (x, y) at the center of the rectangle.
        length (float): Total vertical length of the rectangle.
        width (float): Horizontal thickness (width) of the rectangle.
        layer (int): GDSII layer number to assign the rectangle.

    Returns:
        tuple:
            - List containing the gdspy.Rectangle object.
            - Final (x, y) position at the right-center of the rectangle.
    """
    polygons = []

    # Define lower-left and upper-right corners (vertical orientation)
    lower_left = (start[0], start[1] - length / 2)
    upper_right = (start[0] + width, start[1] + length / 2)

    # Create the rectangle
    rectangle = gdspy.Rectangle(lower_left, upper_right, layer=layer)
    polygons.append(rectangle)

    # Final position is the right-center edge of the rectangle
    final_position = (start[0] + width, start[1])

    return polygons, final_position


def capacitor_inductor_horizontal_fingers(start=[0,0],rectangle_gap=50,width_bar=200,induct_width=5,induct_gap=100,finger_gap=50,total_finger_count=4,finger_width=80,finger_length=750,body_layer=1,inductor_layer=None):
    
    """
    Creates a capacitor-inductor layout with horizontal orientation of interdigitated fingers and inductor.

    Parameters:
        start (list): Starting point [x, y] of the full structure.
        rectangle_gap (float): Gap between bars and bounding rectangle.
        width_bar (float): Width of the vertical bars on each side.
        inductor_length (float): Length of the central inductor region.
        induct_width (float): Width of the central inductor trace.
        induct_gap (float): Vertical spacing between the inductor and adjacent fingers.
        finger_count (int): Number of fingers on each side of inductor (on one side). total fingers will be 4 times of this number .
        finger_width (float): Width of each finger.
        finger_length (float): Length of each finger
        body_layer (int): GDS layer for the capacitor body (the ground-plane
            cutout: outer rectangle minus bus bars minus fingers minus the
            inductor slot).
        inductor_layer (int): GDS layer for the central inductor strip. The
            slot is still cut out of body_layer so the ground-plane gap is
            correct, but a solid copy of the strip is also placed on
            inductor_layer so it can be given its own mesh size in
            HFSS/Sonnet/etc. Defaults to body_layer (old single-layer
            behaviour) if not given.
    
    Returns:
        tuple:
            - List of gdspy polygons representing the layout (body_layer
              geometry plus the inductor_layer strip, mixed in one list --
              each polygon already carries its own .layer attribute).
            - Final position (same as input start point).
    """
    if inductor_layer is None:
        inductor_layer = body_layer

    polygon = []
    finger_count=int(total_finger_count/4)
    # Calculate inductor length and overall dimensions
    inductor_length=finger_gap+finger_length
    length_bar=(4*finger_count*finger_width+(2*finger_count-1)*2*finger_gap+induct_width+2*induct_gap) #Length of vertical bar
    height=length_bar+2*rectangle_gap   # total height of outer bounding rectangle. sum of vertical bar length and gaps on both sides
    length=2*width_bar+inductor_length+2*rectangle_gap  # total length of outer bounding rectangle
    
    # Outer bounding rectangle(background area)
    [poly1,pos1] = create_rectangle_vert_centered(start,length=height,width=length,layer=body_layer)
    polygon+=poly1
    
    # Left bar (subtracted from outer)
    [poly2,pos2]= create_rectangle_vert_centered((start[0]+rectangle_gap,start[1]),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly2, 'not', layer=body_layer)]
    
    # Right bar (subtracted from outer)
    [poly3,pos3]= create_rectangle_vert_centered((start[0]+width_bar+rectangle_gap+inductor_length,start[1]),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly3, 'not', layer=body_layer)]
    
    # Central inductor strip: cut its slot out of the body, and keep a solid
    # copy of the strip itself on inductor_layer for independent meshing.
    [poly4,pos4]=create_rectangle_vert_centered((start[0]+width_bar+rectangle_gap,start[1]),length=induct_width,width=inductor_length,layer=inductor_layer)
    inductor_polygon = list(poly4)
    polygon=[gdspy.fast_boolean(polygon, poly4, 'not', layer=body_layer)]
    
    
   # === Fingers on Left Side (Top and Bottom) ===
    for i in range(finger_count):
        # Bottom-left finger
        x1 = start[0]+width_bar+rectangle_gap
        y1= start[1]- induct_gap - induct_width/2-i *2* (finger_width + finger_gap)
        finger_polygon_1 = gdspy.Polygon([
            [x1, y1],
            [x1, y1-finger_width],
            [x1+ finger_length, y1 -finger_width],
            [x1+finger_length, y1],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_1, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Top-left finger
        x1 = start[0]+width_bar+rectangle_gap
        y1= start[1]+ induct_gap + induct_width/2+(finger_width + finger_gap)+j *2* (finger_width + finger_gap)
        finger_polygon_2 = gdspy.Polygon([
             [x1, y1],
             [x1, y1+finger_width],
             [x1+ finger_length, y1 +finger_width],
             [x1+finger_length, y1],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=body_layer)]
        
    # === Fingers on Right Side (Top and Bottom) ===    
    for i in range(finger_count):
        # Top-right finger
        x1 = start[0]+width_bar+inductor_length+rectangle_gap
        y1= start[1]+ induct_gap + induct_width/2+i *2* (finger_width + finger_gap)
        finger_polygon_3 = gdspy.Polygon([
            [x1, y1],
            [x1, y1+finger_width],
            [x1- finger_length, y1+finger_width ],
            [x1-finger_length, y1],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_3, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Bottom-right finger
        x1 = start[0]+width_bar+inductor_length+rectangle_gap
        y1= start[1] - induct_gap - induct_width/2-(finger_width + finger_gap)-j *2* (finger_width + finger_gap)
        finger_polygon_4 = gdspy.Polygon([
             [x1, y1],
             [x1, y1-finger_width],
             [x1- finger_length, y1 -finger_width],
             [x1-finger_length, y1],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_4, 'not', layer=body_layer)]
    
    final_position= (start[0]+length,start[1])
    
    # Return final geometry and position. inductor_polygon is returned
    # separately (NOT merged into `polygon`) so callers can keep it out of
    # any downstream boolean subtraction against a ground-plane sheet --
    # merging it in would carve the inductor's footprint out of the metal
    # as an unwanted gap, which defeats the point of giving it its own layer.
    return [polygon,final_position,inductor_polygon]

def capacitor_inductor_horizontal_fingers_withconnect(start=[0,0],rectangle_gap=50,width_bar=200,induct_width=5,induct_gap=100,finger_gap=50,total_finger_count=3,finger_width=80,finger_length=750,cut_bar_width=10,body_layer=1,inductor_layer=None):
    
    """
    Creates a capacitor-inductor layout with horizontal orientation of interdigitated fingers and inductor.

    Parameters:
        start (list): Starting point [x, y] of the full structure.
        rectangle_gap (float): Gap between bars and bounding rectangle.
        width_bar (float): Width of the vertical bars on each side.
        inductor_length (float): Length of the central inductor region.
        induct_width (float): Width of the central inductor trace.
        induct_gap (float): Vertical spacing between the inductor and adjacent fingers.
        finger_count (int): Number of fingers on each side of inductor (on one side). total fingers will be 4 times of this number .
        finger_width (float): Width of each finger.
        finger_length (float): Length of each finger
        body_layer (int): GDS layer for the capacitor body.
        inductor_layer (int): GDS layer for the central inductor strip
            (solid copy, in addition to cutting its slot from body_layer).
            Defaults to body_layer if not given.
    
    Returns:
        tuple:
            - List of gdspy polygons representing the layout.
            - Final position (same as input start point).
    """
    if inductor_layer is None:
        inductor_layer = body_layer

    polygon = []
    finger_count=math.ceil(total_finger_count/4)
    # Calculate inductor length and overall dimensions
    inductor_length=finger_gap+finger_length
    length_bar=(4*finger_count*finger_width+(2*finger_count-1)*2*finger_gap+induct_width+2*induct_gap) #Length of vertical bar
    height=length_bar+2*rectangle_gap   # total height of outer bounding rectangle. sum of vertical bar length and gaps on both sides
    length=2*width_bar+inductor_length+2*rectangle_gap  # total length of outer bounding rectangle
    
    # Outer bounding rectangle(background area)
    [poly1,pos1] = create_rectangle_vert_centered(start,length=height,width=length,layer=body_layer)
    polygon+=poly1
    
    # Left bar (subtracted from outer)
    [poly2,pos2]= create_rectangle_vert_centered((start[0]+rectangle_gap,start[1]),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly2, 'not', layer=body_layer)]
    
    # Right bar (subtracted from outer)
    [poly3,pos3]= create_rectangle_vert_centered((start[0]+width_bar+rectangle_gap+inductor_length,start[1]),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly3, 'not', layer=body_layer)]
    
    # Central inductor strip: cut its slot out of the body, and keep a solid
    # copy of the strip itself on inductor_layer for independent meshing.
    [poly4,pos4]=create_rectangle_vert_centered((start[0]+width_bar+rectangle_gap,start[1]),length=induct_width,width=inductor_length,layer=inductor_layer)
    inductor_polygon = list(poly4)
    polygon=[gdspy.fast_boolean(polygon, poly4, 'not', layer=body_layer)]
    
    
   # === Fingers on Left Side (Top and Bottom) ===
    for i in range(finger_count):
        # Bottom-left finger
        x1 = start[0]+width_bar+rectangle_gap
        y1= start[1]- induct_gap - induct_width/2-i *2* (finger_width + finger_gap)
        finger_polygon_1 = gdspy.Polygon([
            [x1, y1],
            [x1, y1-finger_width],
            [x1+ finger_length, y1 -finger_width],
            [x1+finger_length, y1],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_1, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Top-left finger
        x1 = start[0]+width_bar+rectangle_gap
        y1= start[1]+ induct_gap + induct_width/2+(finger_width + finger_gap)+j *2* (finger_width + finger_gap)
        finger_polygon_2 = gdspy.Polygon([
             [x1, y1],
             [x1, y1+finger_width],
             [x1+ finger_length, y1 +finger_width],
             [x1+finger_length, y1],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=body_layer)]
        
    # === Fingers on Right Side (Top and Bottom) ===    
    for i in range(finger_count):
        # Top-right finger
        x1 = start[0]+width_bar+inductor_length+rectangle_gap
        y1= start[1]+ induct_gap + induct_width/2+i *2* (finger_width + finger_gap)
        finger_polygon_3 = gdspy.Polygon([
            [x1, y1],
            [x1, y1+finger_width],
            [x1- finger_length, y1+finger_width ],
            [x1-finger_length, y1],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_3, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Bottom-right finger
        x1 = start[0]+width_bar+inductor_length+rectangle_gap
        y1= start[1] - induct_gap - induct_width/2-(finger_width + finger_gap)-j *2* (finger_width + finger_gap)
        finger_polygon_4 = gdspy.Polygon([
             [x1, y1],
             [x1, y1-finger_width],
             [x1- finger_length, y1 -finger_width],
             [x1-finger_length, y1],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_4, 'not', layer=body_layer)]
        
    # Subtract cut bar (used possibly for probe or separation)
    cut_bar1 = gdspy.Rectangle(
    (start[0] ,
     start[1] - cut_bar_width / 2),
    (start[0] + rectangle_gap,
     start[1]+ cut_bar_width / 2)
)

    polygon = [gdspy.fast_boolean(polygon, cut_bar1, 'not', layer=body_layer) ]
    
    cut_bar2 = gdspy.Rectangle(
    (start[0] +length-rectangle_gap,
     start[1] - cut_bar_width / 2),
    (start[0] +length,
     start[1]+ cut_bar_width / 2)
)

    polygon = [gdspy.fast_boolean(polygon, cut_bar2, 'not', layer=body_layer)]
    
    final_position= (start[0]+length,start[1])
    
    # Return final geometry and position (inductor_polygon kept separate --
    # see note in capacitor_inductor_horizontal_fingers above).
    return [polygon,final_position,inductor_polygon]

def capacitor_inductor_vertical_fingers(start=[0,0],rectangle_gap=50,width_bar=200,finger_gap=50,induct_width=5,induct_gap=50,total_finger_count=4,finger_width=50,finger_length=700,body_layer=1,inductor_layer=None):
    """
    Creates a capacitor-inductor layout with vertical orientation of interdigitated fingers and inductor.

    Parameters:
       start (list): Starting coordinate [x, y].
       rectangle_gap (float): Gap between bars and bounding rectangle.
       width_bar (float): Width of the horizontal bars.
       inductor_length (float): Length of the central inductor section.
       induct_width (float): Width of the central inductor trace.
       induct_gap (float): Horizontal spacing between the inductor and adjacent fingers.
       finger_count (int): Number of fingers on each side of inductor (on one side). total fingers will be 4 times of this number .
       finger_width (float): Width of each finger.
       finger_length (float): Length of each finger .
       body_layer (int): GDS layer for the capacitor body.
       inductor_layer (int): GDS layer for the central inductor strip
           (solid copy, in addition to cutting its slot from body_layer).
           Defaults to body_layer if not given.
   
    Returns:
       tuple: (list of gdspy polygons, final position tuple)
    """
    if inductor_layer is None:
        inductor_layer = body_layer

    polygon = []
    finger_count=int(total_finger_count/4)
    # Calculate inductor length and overall dimensions
    inductor_length=finger_length+finger_gap
    length_bar=(4*finger_count*finger_width+(2*finger_count-1)*2*finger_gap+induct_width+2*induct_gap)    #Length of horizontal bar
    height=length_bar+2*rectangle_gap                    # total height of outer bounding rectangle. sum of horizontal bar length and gaps on both sides
    length=2*width_bar+inductor_length+2*rectangle_gap   # total length of outer bounding rectangle
    
    
    start=(start[0]+height/2,start[1]-length/2)
    # Outer bounding rectangle(background area)
    [poly1,pos1] = create_rectangle_hor_centered(start,length=height,width=length,layer=body_layer)
    polygon+=poly1
    
    # Bottom horizontal bar (cutout)
    [poly2,pos2]= create_rectangle_hor_centered((start[0],start[1]+rectangle_gap),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly2, 'not', layer=body_layer)]
    
    # Top horizontal bar (cutout)
    [poly3,pos3]= create_rectangle_hor_centered((start[0],start[1]+rectangle_gap+width_bar+inductor_length),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly3, 'not', layer=body_layer)]
    
    # Central inductor strip: cut its slot out of the body, and keep a solid
    # copy of the strip itself on inductor_layer for independent meshing.
    [poly4,pos4]=create_rectangle_hor_centered((start[0],start[1]+rectangle_gap+width_bar),length=induct_width,width=inductor_length,layer=inductor_layer)
    inductor_polygon = list(poly4)
    polygon=[gdspy.fast_boolean(polygon, poly4, 'not', layer=body_layer)]
    
    
    # === Fingers on Bottom Side (Right and Left) ===
    for i in range(finger_count):
        # Right side fingers on bottom
        x1 = start[0] + induct_gap + induct_width/2+i *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+rectangle_gap
        finger_polygon_1 = gdspy.Polygon([
            [x1, y1],
            [x1+finger_width, y1],
            [x1+finger_width, y1 + finger_length],
            [x1, y1+finger_length],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_1, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Left side fingers on bottom
        x1 = start[0] - induct_gap - induct_width/2-(finger_width + finger_gap)-j *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+rectangle_gap
        finger_polygon_2 = gdspy.Polygon([
             [x1, y1],
             [x1-finger_width, y1],
             [x1-finger_width, y1 + finger_length],
             [x1, y1+finger_length],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=body_layer)]
        
    # === Fingers on Top Side (Right and Left) ===    
    for i in range(finger_count):
        # Right side fingers on top
        x1 = start[0] + induct_gap + induct_width/2+(finger_width + finger_gap)+i *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+inductor_length+rectangle_gap
        finger_polygon_3 = gdspy.Polygon([
            [x1, y1],
            [x1+finger_width, y1],
            [x1+finger_width, y1 - finger_length],
            [x1, y1-finger_length],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_3, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Left side fingers on top
        x1 = start[0] - induct_gap - induct_width/2-j *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+inductor_length+rectangle_gap
        finger_polygon_4 = gdspy.Polygon([
             [x1, y1],
             [x1-finger_width, y1],
             [x1-finger_width, y1 - finger_length],
             [x1, y1-finger_length],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_4, 'not', layer=body_layer)]
        
        
 
    final_position= (start[0]+height,start[1])
    # inductor_polygon kept separate -- see note in
    # capacitor_inductor_horizontal_fingers above.
    return [ polygon,final_position,inductor_polygon]

def capacitor_inductor_vertical_fingers_withconnect(start=[0,0],rectangle_gap=50,width_bar=200,finger_gap=50,induct_width=5,induct_gap=50,total_finger_count=4,finger_width=50,finger_length=700,cut_bar_width=20,body_layer=1,inductor_layer=None):
    """
    Creates a capacitor-inductor layout with vertical orientation of interdigitated fingers and inductor.

    Parameters:
       start (list): Starting coordinate [x, y].
       rectangle_gap (float): Gap between bars and bounding rectangle.
       width_bar (float): Width of the horizontal bars.
       inductor_length (float): Length of the central inductor section.
       induct_width (float): Width of the central inductor trace.
       induct_gap (float): Horizontal spacing between the inductor and adjacent fingers.
       finger_count (int): Number of fingers on each side of inductor (on one side). total fingers will be 4 times of this number .
       finger_width (float): Width of each finger.
       finger_length (float): Length of each finger .
       body_layer (int): GDS layer for the capacitor body.
       inductor_layer (int): GDS layer for the central inductor strip
           (solid copy, in addition to cutting its slot from body_layer).
           Defaults to body_layer if not given.
   
    Returns:
       tuple: (list of gdspy polygons, final position tuple)
    """
    if inductor_layer is None:
        inductor_layer = body_layer

    polygon = []
    finger_count=int(total_finger_count/4)
    # Calculate inductor length and overall dimensions
    inductor_length=finger_length+finger_gap
    length_bar=(4*finger_count*finger_width+(2*finger_count-1)*2*finger_gap+induct_width+2*induct_gap)    #Length of horizontal bar
    height=length_bar+2*rectangle_gap                    # total height of outer bounding rectangle. sum of horizontal bar length and gaps on both sides
    length=2*width_bar+inductor_length+2*rectangle_gap   # total length of outer bounding rectangle
    
    
    start=(start[0]+height/2,start[1]-length/2)
    # Outer bounding rectangle(background area)
    [poly1,pos1] = create_rectangle_hor_centered(start,length=height,width=length,layer=body_layer)
    polygon+=poly1
    
    # Bottom horizontal bar (cutout)
    [poly2,pos2]= create_rectangle_hor_centered((start[0],start[1]+rectangle_gap),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly2, 'not', layer=body_layer)]
    
    # Top horizontal bar (cutout)
    [poly3,pos3]= create_rectangle_hor_centered((start[0],start[1]+rectangle_gap+width_bar+inductor_length),length=length_bar,width=width_bar,layer=body_layer)
    polygon = [gdspy.fast_boolean(polygon, poly3, 'not', layer=body_layer)]
    
    # Central inductor strip: cut its slot out of the body, and keep a solid
    # copy of the strip itself on inductor_layer for independent meshing.
    [poly4,pos4]=create_rectangle_hor_centered((start[0],start[1]+rectangle_gap+width_bar),length=induct_width,width=inductor_length,layer=inductor_layer)
    inductor_polygon = list(poly4)
    polygon=[gdspy.fast_boolean(polygon, poly4, 'not', layer=body_layer)]
    
    
    # === Fingers on Bottom Side (Right and Left) ===
    for i in range(finger_count):
        # Right side fingers on bottom
        x1 = start[0] + induct_gap + induct_width/2+i *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+rectangle_gap
        finger_polygon_1 = gdspy.Polygon([
            [x1, y1],
            [x1+finger_width, y1],
            [x1+finger_width, y1 + finger_length],
            [x1, y1+finger_length],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_1, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Left side fingers on bottom
        x1 = start[0] - induct_gap - induct_width/2-(finger_width + finger_gap)-j *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+rectangle_gap
        finger_polygon_2 = gdspy.Polygon([
             [x1, y1],
             [x1-finger_width, y1],
             [x1-finger_width, y1 + finger_length],
             [x1, y1+finger_length],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=body_layer)]
        
    # === Fingers on Top Side (Right and Left) ===    
    for i in range(finger_count):
        # Right side fingers on top
        x1 = start[0] + induct_gap + induct_width/2+(finger_width + finger_gap)+i *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+inductor_length+rectangle_gap
        finger_polygon_3 = gdspy.Polygon([
            [x1, y1],
            [x1+finger_width, y1],
            [x1+finger_width, y1 - finger_length],
            [x1, y1-finger_length],
        ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_3, 'not', layer=body_layer)]
        
        
    for j in range(finger_count):
        # Left side fingers on top
        x1 = start[0] - induct_gap - induct_width/2-j *2* (finger_width + finger_gap)
        y1= start[1]+width_bar+inductor_length+rectangle_gap
        finger_polygon_4 = gdspy.Polygon([
             [x1, y1],
             [x1-finger_width, y1],
             [x1-finger_width, y1 - finger_length],
             [x1, y1-finger_length],
         ], layer=body_layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_4, 'not', layer=body_layer)]

    # Subtract cut bars (used possibly for probe or separation). NOTE: these
    # were previously nested inside the last finger loop above (an
    # indentation slip), so they were re-subtracted once per finger for no
    # benefit -- harmless but wasteful. Moved outside the loop here so they
    # run exactly once.
    cut_bar1 = gdspy.Rectangle(
        (start[0]-height/2, start[1]+length/2 - cut_bar_width/2),
        (start[0]-height/2 + rectangle_gap, start[1]+length/2 + cut_bar_width/2)
    )
    polygon = [gdspy.fast_boolean(polygon, cut_bar1, 'not', layer=body_layer)]

    cut_bar2 = gdspy.Rectangle(
        (start[0] +height/2-rectangle_gap, start[1]+length/2 - cut_bar_width/2),
        (start[0] +height, start[1]+length/2 + cut_bar_width/2)
    )
    polygon = [gdspy.fast_boolean(polygon, cut_bar2, 'not', layer=body_layer)]

    final_position= (start[0]+height/2,start[1]+length/2)
    # inductor_polygon kept separate -- see note in
    # capacitor_inductor_horizontal_fingers above.
    return polygon,final_position,inductor_polygon


def inductor(start,radius=10, no_of_meanders=3, meander_length=500, width=5,height=50, layer=1):
   
    """
    Create a meandered inductor with semicircular bends.

    Args:
        start (list or tuple): Starting coordinate [x, y].
        radius (float): Radius of semicircular bends.
        no_of_meanders (int): Number of meanders.
        meander_length (float): Total length of the meandered section.
        width (float): Width of the inductor line.
        height (float): Height margin around the inductor.
        layer (int): GDSII layer.

    Returns:
        polygons (list): List of gdspy Polygon objects for the inductor.
        final_position (tuple): Coordinates [x, y] at the end of the meander.
    
   """
    start = [start[0], start[1] + radius]
    n = no_of_meanders
    # Calculate length of each straight vertical segment between bends
    length = (meander_length - (n * np.pi * radius)) / (4 * n)
    
    polygons = [gdspy.Rectangle(
    (start[0], start[1]-length-height),
    (start[0] +4*radius*n+2*radius, start[1] + length+height))]
    
    # Start with semicircle on the left bottom corner (quarter circle)
    semicircle1 = gdspy.Round(
        start, radius=radius+width/2, 
        inner_radius=radius - width/2, initial_angle=np.pi*2, final_angle=np.pi*3/2, 
        layer=layer, number_of_points=40
    )
    polygons= [gdspy.fast_boolean(polygon,semicircle1, "not", layer=layer)for polygon in polygons]

    
    

    for _ in range(n):
        # Vertical segment upwards
        polygon_pts1 =gdspy.Polygon( [
            [start[0]+radius-width/2, start[1]],
            [start[0] +radius+width/2, start[1]],
            [start[0]+radius+width/2, start[1] + length],
            [start[0]+radius-width/2, start[1] + length],
        ],layer=layer)
        polygons= [gdspy.fast_boolean(polygon,polygon_pts1, "not", layer=layer)for polygon in polygons]

        # Top semicircle (half circle)
        semicircle2 = gdspy.Round(
            (start[0] +2* radius, start[1] + length), radius=radius+width/2, 
            inner_radius=radius - width/2, initial_angle=0, final_angle=np.pi, 
            layer=layer, number_of_points=40
        )
        polygons= [gdspy.fast_boolean(polygon,semicircle2, "not", layer=layer)for polygon in polygons]

       # Vertical segment downwards on the next vertical line
        polygon_pts2 = gdspy.Polygon( [
            [start[0] + 3 * radius-width/2 , start[1]],
            [start[0] + 3 * radius+width/2, start[1]],
            [start[0] + 3 * radius+width/2, start[1] + length],
            [start[0] + 3 * radius-width/2 , start[1] + length],
        ],layer=layer)
        polygons= [gdspy.fast_boolean(polygon,polygon_pts2, "not", layer=layer)for polygon in polygons]


        # Third vertical segment (going downward)
        polygon_pts3 = gdspy.Polygon([
            [start[0] + 3 * radius-width/2 , start[1]],
            [start[0] + 3 * radius+width/2, start[1]],
            [start[0] + 3 * radius+width/2, start[1] - length],
            [start[0] + 3 * radius-width/2 , start[1] - length],
            ],layer=layer)
        polygons= [gdspy.fast_boolean(polygon,polygon_pts3, "not", layer=layer)for polygon in polygons]
       
        # Bottom semicircle (half circle)
        semicircle3 = gdspy.Round(
            (start[0] + 4 * radius , start[1] - length), radius=radius+width/2, 
            inner_radius=radius - width/2, initial_angle=np.pi, final_angle=2 * np.pi, 
            layer=layer, number_of_points=40
        )
        polygons= [gdspy.fast_boolean(polygon,semicircle3, "not", layer=layer)for polygon in polygons]

        # Fourth Vertical segment upwards on next vertical line
        polygon_pts4 = gdspy.Polygon([
            [start[0] + 5 * radius-width/2, start[1]],
            [start[0] + 5 * radius+width/2 , start[1]],
            [start[0] + 5 * radius+width/2 , start[1] - length],
            [start[0] + 5 * radius-width/2, start[1] - length]
        ],layer=layer)
        polygons= [gdspy.fast_boolean(polygon,polygon_pts4, "not", layer=layer)for polygon in polygons]

        # Update `start` for the next meander
        start = [start[0] + 4 * radius, start[1]]
    
        # Final semicircle closing on the right side
    semicircle4 = gdspy.Round(
        (start[0]+2*radius,start[1]), radius=radius+width/2, 
        inner_radius=radius - width/2, initial_angle=np.pi, final_angle=np.pi/2, 
        layer=layer, number_of_points=40
    )
    polygons= [gdspy.fast_boolean(polygon,semicircle4, "not", layer=layer)for polygon in polygons]
    final_position = (start[0]+2*radius,start[1]+radius)
    
    

    return [polygons, final_position]

def create_rectangle_hor(start, length, width, layer=1):
    """
    Creates a horizontally aligned rectangle (extending in X-direction) using the specified dimensions.
    
    Parameters:
        start : Starting point — defines the center of the base edge of the rectangle.
        length : Total length of the rectangle (along X-axis).
        width : Total height of the rectangle (along Y-axis).
    """
    polygon=[]
    # Define the rectangle from center-bottom to center-top
    rectangle = gdspy.Rectangle((start[0]-length/2,start[1]),(start[0]+length/2,start[1]+width), layer=layer)
    polygon.append(rectangle)
    final_position=(start[0],start[1]+width)
    return polygon,final_position

def create_rectangle_vert(start, length, width, layer=1):
    """
    Creates a vertically aligned rectangle (extending in Y-direction) using the specified dimensions.

    Parameters:
        start :Starting point — defines the center of the left edge of the rectangle.
        length :Total height of the rectangle (along Y-axis).
        width :Total width of the rectangle (along X-axis).
        
    """
    polygon=[]
    # Define rectangle from center-left to center-right
    rectangle = gdspy.Rectangle((start[0],start[1]-length/2),(start[0]+width,start[1]+length/2), layer=layer)
    polygon.append(rectangle)
    final_position=(start[0]+width,start[1])
    return polygon,final_position

def capacitor_inductor_half1(start=[0,0],width_bar=25,finger_gap=5,finger_count=4,finger_width=20,finger_length=145,layer=1):
    
    """
    Constructs one half of a symmetric interdigitated capacitor structure (with bars and fingers),. This version produces
    the "half" structure, which can later be mirrored and combined for the full structure.

    The structure consists of:
      - A rectangular base bar.
      - A surrounding rectangle for defining the subtraction region.
      - Alternating up/down finger elements connecting from top and bottom bars.
      - All shapes are boolean-subtracted from a larger surrounding box to define the final design.

    Parameters:
        start : Starting coordinate of the structure's center base.
        width_bar :Width of the horizontal bar (center conductor).
        finger_gap : Gap between the capacitor fingers.
        finger_count : Number of fingers on each side (top and bottom).
        finger_width : Width of each finger.
        finger_length : Length of each finger (from the bar outward).
        layer : GDS layer for the whole structure.

    """
    
    
    polygon = []
    # Calculate length of horizontal bar, total length and total capacitor width
    length_bar=((2*finger_count-1)*finger_width+(2*finger_count)*finger_gap)
    length=length_bar
    width=width_bar+finger_length+finger_gap
    # Center of the full rectangle that contains all elements
    start_new=(start[0]+length/2,start[1])
   
    # Base surrounding rectangle: defines the area where everything is subtracted from
    [poly1a,pos1a] = create_rectangle_hor(start_new,length=length,width=width,layer=layer)
    polygon+=poly1a
    # Inner horizontal bar at center (subtracted)
    [poly1,pos1]= create_rectangle_hor((start_new[0],start_new[1]),length=length_bar,width=width_bar,layer=layer)
    polygon = [gdspy.fast_boolean(polygon, poly1, 'not', layer=layer)]
    # Top bar (also subtracted to isolate upper fingers)
    [poly2,pos2]= create_rectangle_hor((start_new[0],start_new[1]+width_bar+finger_length+finger_gap),length=length_bar,width=width_bar,layer=layer)
    polygon = [gdspy.fast_boolean(polygon, poly2, 'not', layer=layer)]
    
    # Bottom fingers (attached to center bar)
   
    for j in range(finger_count):
        x1 = start_new[0] +length_bar/2-finger_gap-j *2* (finger_width + finger_gap)
        y1= start_new[1]+width_bar
        finger_polygon_2 = gdspy.Polygon([
             [x1, y1],
             [x1-finger_width, y1],
             [x1-finger_width, y1 + finger_length],
             [x1, y1+finger_length],
         ], layer=layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=layer)]
        
        
    # Top fingers (attached to top bar, pointing downward)
        
    for j in range(finger_count):
        x1 = start_new[0] +length_bar/2-2*finger_gap-finger_width-j *2* (finger_width + finger_gap)
        y1= start_new[1]+width_bar+finger_length+finger_gap
        finger_polygon_4 = gdspy.Polygon([
             [x1, y1],
             [x1-finger_width, y1],
             [x1-finger_width, y1 - finger_length],
             [x1, y1-finger_length],
         ], layer=layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_4, 'not', layer=layer)]
        
        
   
    final_position= (start[0]+length,start[1])
    return polygon,final_position


def capacitor_inductor_half1_mirrored_x(start=[0,0],width_bar=25,finger_gap=5,finger_count=4,finger_width=20,finger_length=145,layer=1):
    
    """
    Constructs the mirrored version (in the -Y direction) of an interdigitated capacitor/inductor half structure.
    This structure is commonly used in microwave circuits such as resonators or LC filters.

    The mirrored design complements `capacitor_inductor_half1` by producing the opposite half (flipped along the X-axis),
    allowing them to be paired together to form a full, symmetric component.

    Parameters:
        start : Starting coordinate of the structure’s center base.
        width_bar : Width of the central horizontal bar (metal).
        finger_gap : Gap between adjacent fingers.
        finger_count : Number of fingers per side (top and bottom).
        finger_width : Width of each individual finger.
        finger_length :Length of each finger extending from the bar outward.
        layer : GDS layer for the whole structure.

    """
    polygon = []
    
    # Total bar length across all fingers and gaps
    length_bar=((2*finger_count-1)*finger_width+(2*finger_count)*finger_gap)
    length=length_bar
    width=width_bar+finger_length+finger_gap
    
    # Center point of the full mirrored rectangle
    start_new=start[0]+length/2,start[1]
    
    # Create the main outer mirrored rectangle (using negative width to flip along X)
    [poly1a,pos1a] = create_rectangle_hor((start_new[0], start_new[1]), length=length, width=-width, layer=layer)
    polygon+=poly1a

   # Subtract center bar region (mirrored)
    [poly1,pos1]= create_rectangle_hor((start_new[0], start_new[1]), length=length_bar, width=-width_bar, layer=layer)
    polygon = [gdspy.fast_boolean(polygon, poly1, 'not', layer=layer)]

    # Subtract bottom bar (mirrored, placed below central bar)
    [poly2,pos2]= create_rectangle_hor((start_new[0], start_new[1] - width_bar - finger_length-finger_gap - width_bar), length=length_bar, width=width_bar, layer=layer)
    polygon = [gdspy.fast_boolean(polygon, poly2, 'not', layer=layer)]
    
    # Create bottom fingers (pointing downward from center bar)
    
    for j in range(finger_count):
        x1 = start_new[0] +length_bar/2-finger_gap-j *2* (finger_width + finger_gap)
        
        y1_mirrored = start_new[1]-width_bar

        
        finger_polygon_2 = gdspy.Polygon([
             [x1, y1_mirrored],
             [x1-finger_width, y1_mirrored],
             [x1-finger_width, y1_mirrored - finger_length],
             [x1, y1_mirrored - finger_length],
        ], layer=layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_2, 'not', layer=layer)]
    
    
    # Create top fingers (attached to bottom bar, pointing upward)
    
    for j in range(finger_count):
        x1 = start_new[0] +length_bar/2-2*finger_gap-finger_width-j *2* (finger_width + finger_gap)
        
        y1_mirrored = start_new[1]  -finger_length-finger_gap-width_bar

        
        finger_polygon_4 = gdspy.Polygon([
             [x1, y1_mirrored],
             [x1-finger_width, y1_mirrored],
             [x1-finger_width, y1_mirrored + finger_length],
             [x1, y1_mirrored + finger_length],
        ], layer=layer)
        polygon = [gdspy.fast_boolean(polygon, finger_polygon_4, 'not', layer=layer)]

    
    final_position=(start[0]+length, start[1])
    return polygon,final_position






#Create GDS Library
#lib = gdspy.GdsLibrary()
#cell = lib.new_cell('abcd')

#cell.add(Braggmirror_unitcell2_left([0,0])[0])




 #View the layout
#gdspy.LayoutViewer(lib)