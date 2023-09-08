#from tf import HomogeneousMatrix
import time 

from tf import HomogeneousMatrix 
# Importing the HomogeneousMatrix class from the tf module
# It is a class of methods

import matplotlib.pyplot as plt
# Importing the pyplot module from the matplotlib library

#import control as co

import pandas as pd
# Importing the pandas library

import matplotlib.patches as patches
# Importing the patches module from the matplotlib library

import matplotlib as mpl

from functools import partial

from mpl_toolkits.mplot3d import Axes3D
# Importing the Axes3D module from the mpl_toolkits.mplot3d library

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 

from matplotlib.figure import Figure 

from math import *
# Importing the math module

import tkinter as tk
# Importing the tkinter module

import math

from matplotlib import cm

import numpy as np

from tkinter import Tk, Label, Button ,Frame,Entry,Radiobutton ,IntVar,Toplevel

from tkinter import StringVar

def show_simulation(eps) : 
    
    epx1, epy1, epz1, epx2, epy2, epz2, epx3, epy3, epz3,epts,eptstotal,setpoint_q1,setpoint_q2,setpoint_q3=eps

    joint1 = HomogeneousMatrix()
    joint2 = HomogeneousMatrix()
    joint3 = HomogeneousMatrix()
    joint4 = HomogeneousMatrix()
    
    joint1.set_theta(float(en1.get()))
    joint1.set_d(float(en2.get()))
    joint1.set_a(float(en3.get()))
    joint1.set_alpha(float(en4.get()))
    q1 = float(en4.get())*(np.pi/180)
    print("q1: ",q1)
    joint2.set_theta(float(en5.get()))
    joint2.set_d(float(en6.get()))
    joint2.set_a(float(en7.get()))
    joint2.set_alpha(float(en8.get()))
    q2= float(en8.get())*(np.pi/180)
    print("q2: ",q2)
    joint3.set_theta(float(en9.get()))
    joint3.set_d(float(en10.get()))
    joint3.set_a(float(en11.get()))
    joint3.set_alpha(float(en12.get()))
    q3 = float(en12.get())*(np.pi/180)
    print("q3: ",q3)
    joint4.set_theta(float(en13.get()))
    joint4.set_d(float(en14.get()))
    joint4.set_a(float(en15.get()))
    joint4.set_alpha(float(en16.get()))

   
    joint2.set_parent(joint1.get())
    joint3.set_parent(joint2.get())
    joint4.set_parent(joint3.get())  
    
    
    Y = np.array([joint1[1, 3], 
         joint2[1, 3], joint3[1, 3],joint4[1,3]])*-1
    
    Z = np.array([joint1[2, 3],
         joint2[2, 3], joint3[2, 3],joint4[2,3]])
    print("000000000")
    print(Y , Z)
    print("00000000000000")
    top =Toplevel()
   
    top.title("simulation")
   
    top.geometry("700x700")  

    figure = Figure(figsize = (5, 5), 
                     dpi = 100)    
    ax = figure.add_subplot(111)
    
    
    chart_type = FigureCanvasTkAgg(figure, top) 
    
    chart_type.get_tk_widget().pack()


    ax.scatter(Z, Y, color = "green",s=100)   
    
    ax.plot(Z, Y ,linewidth=2)
    
    m1= float(epx2.get())
    print("m1 :",m1)
    m2= float(epy2.get())
    print("m2: ",m2)
    m3= float(epz2.get())
    print("m3:",m3)
    KP = float(epx3.get())
    print("kp:",KP)
    KD = float(epy3.get())
    print("kd:" ,KD)
    KI = float(epz3.get())
    print("KI:",KI)
    start_time= float(epts.get())
    
    simulation_time = float(eptstotal.get())
    print("simulation time",simulation_time)
    
    g= 9.8
    
    
    l1= float(epx1.get())
    l2= float(epy1.get())
    l3= float(epz1.get())
    
    print("l1,l2,l3 :",l1,l2,l3)
    
    I1 = (1/3)*m1*l1*l1
    I2 = (1/3)*m2*l2*l2
    I3 = (1/3)*m3*l3*l3
    
    lc1=(l1 /2)
    lc2=(l2 /2)
    lc3=(l3 /2)
    
    p9 = m3*g*lc3

    p8 = m2*g*lc2 + m3*g*l2
    
    p7 = m2*g*lc2 + m3*g*l2
    
    p6= m3*lc3*lc3+I3
    
    p5 = m2*lc2*lc2 + I2 + m3*l2*l2 + m3*lc3*lc3+I3

    p4 = m3*l1*lc3 

    p3 = m3*l2*lc3
    
    p2 = m2*l1*lc2 + m3*l1*l2
    
    p1 = m1*lc1*lc1+I1+m2*l1*l1+m2*lc2*lc2+I2+m3*l1*l1+m3*l2*l2+m3*lc3*lc3+I3
    
    step_functionx = np.arange(0,simulation_time,1)
    
    q1dot ,q2dot ,q3dot = 0,0,0
    
                
    stepfun1= float(setpoint_q1.get())*(np.pi/180)
    stepfun2= float(setpoint_q2.get())*(np.pi/180)
    stepfun3= float(setpoint_q3.get())*(np.pi/180)
    
    totalerror1= [0]
    totalerror2= [0]
    totalerror3= [0]
    
    total_alpha1= [0]
    total_alpha2= [0]
    total_alpha3= [0]
    
    total_total_alpha1=[0]
    total_total_alpha2=[0]
    total_total_alpha3=[0]
    
    errors=[]
    qs1= []
    qs2= []
    qs3= []
    PIDS=[]
    moshtaghz=[]
    qdot = np.array([0,0,0])
    antegerals=[]
    Glists=[]
    zarbs=[]
    zarb2s=[]
    """
    t1= time.time()
    while True :
        
        if time.time()-t1 >2 :
            
             ax.remove()
             break
    """
    for t in step_functionx: 
        
            int_matrix= np.array([
            [1.0*I1 + 1.0*I2 + 1.0*I3 + 1.0*lc1**2*m1*sin(q1)**2 + 1.0*lc1**2*m1*cos(q1)**2 + 0.5*m2*((-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*(-l1*sin(q1) - lc2*sin(q1 + q2)) + (l1*cos(q1) + lc2*cos(q1 + q2))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))) + 0.5*m3*((-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))), 1.0*I2 + 1.0*I3 + 0.5*m2*(-lc2*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*sin(q1 + q2) + lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*cos(q1 + q2)) + 0.5*m3*((-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))), 1.0*I3 + 0.5*m3*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*cos(q1 + q2 + q3))],
            [                                                                                                                               1.0*I2 + 1.0*I3 + 0.5*m2*(-2*lc2*(-l1*sin(q1) - lc2*sin(q1 + q2))*sin(q1 + q2) + 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*cos(q1 + q2)) + 0.5*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))),                                                                                         1.0*I2 + 1.0*I3 + 0.5*m2*(2*lc2**2*sin(q1 + q2)**2 + 2*lc2**2*cos(q1 + q2)**2) + 0.5*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))),                               1.0*I3 + 0.5*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*cos(q1 + q2 + q3))],
            [                                                                                                                                                                                                                                                                                                          1.0*I3 + 0.5*m3*(-2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*cos(q1 + q2 + q3)),                                                                                                                                                                                                              1.0*I3 + 0.5*m3*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*cos(q1 + q2 + q3)),                                                                                                              1.0*I3 + 0.5*m3*(2*lc3**2*sin(q1 + q2 + q3)**2 + 2*lc3**2*cos(q1 + q2 + q3)**2)]])
            
            Cmatrix= np.array([
            [0.25*m2*(-lc2*(-2*l1*cos(q1) - 2*lc2*cos(q1 + q2))*sin(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.5*m2*((-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*(-l1*cos(q1) - lc2*cos(q1 + q2)) + (-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*(l1*cos(q1) + lc2*cos(q1 + q2)) + (-l1*sin(q1) - lc2*sin(q1 + q2))*(-2*l1*cos(q1) - 2*lc2*cos(q1 + q2)) + (-l1*sin(q1) - lc2*sin(q1 + q2))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))) + 0.25*m2*(-lc2*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(-l1*sin(q1) - lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) - 0.25*m3*q3dot*(-2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) + 2*lc3*(-l1*sin(q1) - lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*cos(q1) - l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*((-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3)) + (lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))) + 0.25*m3*((-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.5*m3*((-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l1*cos(q1) - l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-l1*sin(q1) - lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.25*m3*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) + lc3*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.5*q1dot*(0.5*m2*((-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*(-l1*cos(q1) - lc2*cos(q1 + q2)) + (-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*(l1*cos(q1) + lc2*cos(q1 + q2)) + (-l1*sin(q1) - lc2*sin(q1 + q2))*(-2*l1*cos(q1) - 2*lc2*cos(q1 + q2)) + (-l1*sin(q1) - lc2*sin(q1 + q2))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))) + 0.5*m3*((-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l1*cos(q1) - l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-l1*sin(q1) - lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3)))) - 0.5*q2dot*(0.5*m2*(-2*lc2*(-l1*cos(q1) - lc2*cos(q1 + q2))*sin(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.5*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l1*cos(q1) - l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)))), 0.5*m2*(-lc2*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*cos(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.25*m2*(-lc2*(-2*l1*cos(q1) - 2*lc2*cos(q1 + q2))*sin(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.25*m2*(-lc2*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(-l1*sin(q1) - lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) - 0.25*m3*q2dot*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) - 0.25*m3*q3dot*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.5*m3*((-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.25*m3*((-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3)) + (lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))) + 0.25*m3*((-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.25*m3*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.5*q1dot*(0.5*m2*(-lc2*(-2*l1*cos(q1) - 2*lc2*cos(q1 + q2))*sin(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.5*m3*((-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3)) + (lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)))), -0.25*m3*q1dot*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) + lc3*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.25*m3*q2dot*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.5*m3*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) + lc3*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l1*cos(q1) - 2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3))],
            [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          0.25*m2*(-2*lc2*(-l1*sin(q1) - lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.5*m2*(-2*lc2*(-l1*cos(q1) - lc2*cos(q1 + q2))*sin(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2)) - 0.25*m3*q3dot*(-2*lc3*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.25*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.5*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l1*cos(q1) - l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))) + 0.25*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.5*q1dot*(0.5*m2*(-lc2*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(-l1*sin(q1) - lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.5*m3*((-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3)))) - 0.5*q2dot*(0.5*m2*(-2*lc2*(-l1*sin(q1) - lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.5*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3)))),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0.25*m2*(-2*lc2*(-l1*sin(q1) - lc2*sin(q1 + q2))*cos(q1 + q2) - 2*lc2*(l1*cos(q1) + lc2*cos(q1 + q2))*sin(q1 + q2)) - 0.25*m3*q2dot*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) - 0.25*m3*q3dot*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.75*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.25*m3*((-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3)) + (-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))) + 0.25*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.5*q1dot*(0.5*m2*(-lc2*(-2*l1*sin(q1) - 2*lc2*sin(q1 + q2))*cos(q1 + q2) - lc2*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2))*sin(q1 + q2)) + 0.5*m3*((-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3)) + (-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3)) + (-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3)) + (-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3)))),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -0.25*m3*q1dot*(-lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.25*m3*q2dot*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.5*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.5*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(-2*l2*cos(q1 + q2) - 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + lc3*(-2*lc2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3))],
            [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              -0.25*m3*q1dot*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.25*m3*q2dot*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.25*m3*q3dot*(-2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.5*m3*(-2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) + 2*lc3*(-l1*sin(q1) - lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*cos(q1) - l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              -0.25*m3*q1dot*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.25*m3*q2dot*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.25*m3*q3dot*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.75*m3*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l2*cos(q1 + q2) - lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3) + 2*lc3*(-lc2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     -0.25*m3*q1dot*(-lc3*(-2*l1*sin(q1) - 2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*l1*cos(q1) + 2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) - 0.25*m3*q2dot*(-lc3*(-2*l2*sin(q1 + q2) - 2*lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - lc3*(2*lc2*cos(q1 + q2) + 2*lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3)) + 0.25*m3*(-2*lc3*(-l1*sin(q1) - l2*sin(q1 + q2) - lc3*sin(q1 + q2 + q3))*cos(q1 + q2 + q3) - 2*lc3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))*sin(q1 + q2 + q3))]])
            


            Gmatrix=np.array([
            [g*lc1*m1*cos(q1) + g*m2*(l1*cos(q1) + lc2*cos(q1 + q2)) + g*m3*(l1*cos(q1) + lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))],
            [                                               g*lc2*m2*cos(q1 + q2) + g*m3*(lc2*cos(q1 + q2) + lc3*cos(q1 + q2 + q3))],
            [                                                                                            g*lc3*m3*cos(q1 + q2 + q3)]])            



            
            int_matrix = np.linalg.inv(int_matrix)

            print("darageha",stepfun1,stepfun2,stepfun3,q1,q2,q3)
            
            qs1.append((stepfun1,q1))
            qs2.append((stepfun2,q2))
            qs3.append((stepfun3,q3))
            
            error1 = (stepfun1- q1)
            
            error2 = (stepfun2- q2)
            
            error3= (stepfun3 - q3)
            
            errors.append(error3)
            
            entegeral1= (sum(totalerror1) +error1)*KI
            entegeral2= (sum(totalerror2) +error2)*KI
            entegeral3= (sum(totalerror3) +error3)*KI
            
            antegerals.append(entegeral3)
            

            totalerror1.append(error1)
            totalerror2.append(error2)
            totalerror3.append(error3)

            
            moshtagh1= ((totalerror1[-1] - totalerror1[-2])/1)*KD
            moshtagh2= ((totalerror2[-1] - totalerror2[-2])/1)*KD
            moshtagh3= ((totalerror3[-1] - totalerror3[-2])/1)*KD
            

            
            moshtaghz.append(moshtagh3)
            PID1= moshtagh1 + entegeral1 +KP*error1
            PID2 = moshtagh2 + entegeral2 +KP*error2
            PID3 = moshtagh3 + entegeral3 +KP*error3
            PIDS.append(PID3)
            PID= np.array([PID1,PID2,PID3])
            zarb= np.matmul(Cmatrix,qdot)
            zarb2= np.matmul(int_matrix,(PID-zarb-Gmatrix))
            zarb2s.append(zarb2[2])
            zarbs.append(zarb[2])
            alpha = zarb2
            print(type(alpha[0]))
 
            print(int_matrix)
            print("00000")
            print(PID)
            print("8888")
            print(Cmatrix)
            print("000000")
            print(Gmatrix)
            print("###############")
            print(alpha)
            
            total_alpha_entegeral1= alpha[0] + sum(total_alpha1)
            total_alpha_entegeral2= alpha[1] + sum(total_alpha2)
            total_alpha_entegeral3= alpha[2] + sum(total_alpha3)
            
            total_total_alpha_entegeral1= total_alpha_entegeral1 + sum(total_total_alpha1)
            total_total_alpha_entegeral2= total_alpha_entegeral2 + sum(total_total_alpha2)
            total_total_alpha_entegeral3= total_alpha_entegeral3 + sum(total_total_alpha3)
            total_alpha1.append(alpha[0])
            total_alpha2.append(alpha[1])
            total_alpha3.append(alpha[2])
            
            total_total_alpha1.append(total_alpha_entegeral1)
            total_total_alpha2.append(total_alpha_entegeral2)
            total_total_alpha3.append(total_alpha_entegeral3)
            
            
            
            q1= (total_total_alpha_entegeral1)
            q2= (total_total_alpha_entegeral2)
            q3= (total_total_alpha_entegeral3)
            
            q1dot =total_alpha_entegeral1
            q2dot =total_alpha_entegeral2
            q3dot= total_alpha_entegeral3
            
            qdot= np.array([q1dot,q2dot,q3dot])
            
            """
            joint1 = HomogeneousMatrix()
            joint2 = HomogeneousMatrix()
            joint3 = HomogeneousMatrix()
            joint4 = HomogeneousMatrix()
            
            print("en1:",float(en1.get()))
            joint1.set_theta(0)
            print("en2:",float(en2.get()))
            joint1.set_d(0)
            print("en3:",float(en3.get()))
            joint1.set_a(0)
            print("en4:",float(en4.get()))
            print(q1,"***:***")
            joint1.set_alpha(q1)
            print("en5:",float(en5.get()))
            joint2.set_theta(0)
            print("en6:",float(en6.get()))
            joint2.set_d(l1)
            print("en7:",float(en7.get()))
            joint2.set_a(0)
            print("en8:",float(en8.get()))
            joint2.set_alpha(q2)
            print("en9:",float(en9.get()))
            joint3.set_theta(0)
            print("en10:",float(en10.get()))
            joint3.set_d(l2)
            print("en11:",float(en11.get()))
            joint3.set_a(0)
            print("en12:",float(en12.get()))
            joint3.set_alpha(q3)
            print("en13:",float(en13.get()))
        
            joint4.set_theta(0)
            print("en14:",float(en14.get()))
            joint4.set_d(l3)
            print("en15:",float(en15.get()))
            joint4.set_a(0)
            print("en116:",float(en16.get()))
            joint4.set_alpha(0)
        
           
            joint2.set_parent(joint1.get())
            joint3.set_parent(joint2.get())
            joint4.set_parent(joint3.get())  
            
            
            Y = np.array([joint1[1, 3], 
                 joint2[1, 3], joint3[1, 3],joint4[1,3]])*-1
            
            Z = np.array([joint1[2, 3],
                 joint2[2, 3], joint3[2, 3],joint4[2,3]])
            top =Toplevel()
           
            top.title("simulation"+str(t))
           
            top.geometry("700x700")
            
            figure = Figure(figsize = (5, 5), 
                             dpi = 100)    
            ax = figure.add_subplot(111)
            
            
            chart_type = FigureCanvasTkAgg(figure, top) 
            
            chart_type.get_tk_widget().pack()
            
            ax.scatter(Z, Y, color = "green",s=100)   
    
            ax.plot(Z, Y ,linewidth=2)
            
            
            t1= time.time()
            
            
            while True :
                
                if time.time()-t1 >2 :
                    
                     ax.remove()
                     break
            """
    #plt.plot([i for i,j,k in errors])   
    
    #plt.plot([j for i,j,k in errors])  
    
    #plt.plot([k for i,j,k in errors])  
    plt.plot([i for i,j in qs1])
    plt.plot([j for i,j in qs1])
    plt.show()
    
    plt.plot([i for i,j in qs2])
    plt.plot([j for i,j in qs2])
    plt.show()

    plt.plot([i for i,j in qs3])
    plt.plot([j for i,j in qs3])
    plt.show()
    
    plt.plot(total_alpha3)
    plt.title("shetab")
    plt.show()
    plt.plot(PIDS)
    plt.title("PIDS")
    
    plt.show()
    plt.plot(moshtaghz)
    plt.title("moshtaghz")

    plt.show()
    plt.plot(errors)
    plt.title("errors")

    plt.show()
    plt.plot(antegerals)
    plt.title("antegeral")
    plt.show()
    plt.plot(Glists)
    plt.title("G")
    plt.show()
    plt.plot(zarbs)
    plt.title("zarb")
    plt.show()
    plt.plot(total_total_alpha3)
    plt.title("qs")
    plt.show()
    plt.plot(zarb2s)
    plt.title("zarb2")
    plt.show()  
def start_simulation():
    

    
    top2 =Toplevel()
   
    top2.title("forward kinematic")
   
    top2.geometry("1000x250")  
      
    #figure2 = Figure(figsize = (5, 5), 
    #             dpi = 100)   

    #ax2 = figure2.gca(projection='3d')
    

        
    v11 = StringVar(top2, value='100')  
    v22 = StringVar(top2, value='100')  
    v33 = StringVar(top2, value='100')  
    v44 = StringVar(top2, value='1')  
    v55 = StringVar(top2, value='1')  
    v66 = StringVar(top2, value='1')  
    v77 = StringVar(top2, value='3')  
    v88 = StringVar(top2, value='2')  
    v99 = StringVar(top2, value='1')  
    v1010 = StringVar(top2, value='100')  
    v1111 = StringVar(top2, value='5')  
    v1212 = StringVar(top2, value='30')  
    v1313 = StringVar(top2, value='60')  
    v1414 = StringVar(top2, value='50')  

 
    epx1 =Entry(top2,textvariable=v11)
    epx1.place(x=50, y=200)
    epx1.config(width =5)
    epy1 =Entry(top2,textvariable=v22)
    epy1.place(x=150, y=200)
    epy1.config(width =5)
    epz1 =Entry(top2,textvariable=v33)
    epz1.place(x=250, y=200)
    epz1.config(width =5)  
    epx2 =Entry(top2,textvariable=v44)
    epx2.place(x=50, y=100)
    epx2.config(width =5)
    epy2 =Entry(top2,textvariable=v55)
    epy2.place(x=150, y=100)
    epy2.config(width =5)
    epz2 =Entry(top2,textvariable=v66)
    epz2.place(x=250, y=100)
    epz2.config(width =5)     
    epx3 =Entry(top2,textvariable=v77)
    epx3.place(x=400, y=100)
    epx3.config(width =5)
    epy3 =Entry(top2,textvariable=v88)
    epy3.place(x=450, y=100)
    epy3.config(width =5)
    epz3 =Entry(top2,textvariable=v99)
    epz3.place(x=500, y=100)
    epz3.config(width =5)
    epts =Entry(top2,textvariable=v1010)
    epts.place(x=400, y=200)
    epts.config(width =5)   
    eptstotal =Entry(top2,textvariable=v1111)
    eptstotal.place(x=500, y=200)
    eptstotal.config(width =5)  
    eptstep_q1 =Entry(top2,textvariable=v1212)
    eptstep_q1.place(x=650, y=200)
    eptstep_q1.config(width =5)   
    eptstep_q2 =Entry(top2,textvariable=v1313)
    eptstep_q2.place(x=750, y=200)
    eptstep_q2.config(width =5)   
    eptstep_q3 =Entry(top2,textvariable=v1414)
    eptstep_q3.place(x=850, y=200)
    eptstep_q3.config(width =5)   
    labepx1 =Label(top2 , text ="tool_link1")
    labepy1 =Label(top2 , text ="tool_link2")
    labepz1 =Label(top2 , text ="tool_link3")
    labepx2 =Label(top2 , text ="jerm_link1")
    labepy2 =Label(top2 , text ="jerm_link2")
    labepz2 =Label(top2 , text ="jerm_link3")
    labepx3 =Label(top2 , text ="KP")
    labepy3 =Label(top2 , text ="KD")
    labepz3 =Label(top2 , text ="KI")
    labets =Label(top2 , text ="start_time")
    labetstotal =Label(top2 , text ="simulatiom_time")
    labetstep_q1 =Label(top2 , text ="step_q1")
    labetstep_q2 =Label(top2 , text ="step_q2")
    labetstep_q3 =Label(top2 , text ="step_q3")
    labepx1.place(x =50 ,y=150)
    labepx1.config(font=("mitra", 10 ,"bold"))
    labepy1.place(x =150 ,y=150)
    labepy1.config(font=("mitra", 10 ,"bold"))       
    labepz1.place(x =250 ,y=150)
    labepz1.config(font=("mitra", 10 ,"bold"))
    labepx2.place(x =50 ,y=50)
    labepx2.config(font=("mitra", 10 ,"bold"))
    labepy2.place(x =150 ,y=50)
    labepy2.config(font=("mitra", 10 ,"bold"))       
    labepz2.place(x =250 ,y=50)
    labepz2.config(font=("mitra", 10 ,"bold")) 
    labepx3.place(x =400 ,y=50)
    labepx3.config(font=("mitra", 10 ,"bold"))
    labepy3.place(x =450 ,y=50)
    labepy3.config(font=("mitra", 10 ,"bold"))       
    labepz3.place(x =500 ,y=50)
    labepz3.config(font=("mitra", 10 ,"bold"))  
    labets.place(x =400 ,y=150)
    labets.config(font=("mitra", 10 ,"bold")) 
    labetstotal.place(x =500 ,y=150)
    labetstotal.config(font=("mitra", 10 ,"bold")) 
    labetstotal.place(x =500 ,y=150)
    labetstep_q1.config(font=("mitra", 10 ,"bold")) 
    labetstep_q1.place(x =650 ,y=150)
    labetstep_q2.config(font=("mitra", 10 ,"bold")) 
    labetstep_q2.place(x =750 ,y=150)
    labetstep_q3.config(font=("mitra", 10 ,"bold")) 
    labetstep_q3.place(x =850 ,y=150)
    eps= (epx1, epy1, epz1, epx2, epy2, epz2, epx3, epy3, epz3,epts,eptstotal,
          eptstep_q1,eptstep_q2,eptstep_q3)
    bottom6 =Button(top2 , text ="show_simulation",
                command=lambda: show_simulation(eps))

    bottom6.config(fg="blue",padx =50, pady=10)
    bottom6.pack(padx =5, pady =5)
    bottom6.place(x=600,y=50)
    

def barkhord(Y,Z,z,y,r,ax) :
    
     y0,y1,y2,y3 =Y[0] ,Y[1] ,Y[2] ,Y[3]
     
     z0,z1,z2,z3 = Z[0] ,Z[1] ,Z[2] , Z[3]
     
     print(y0,z0,y1,z1,y2,z2,y3,z3)
     
     shib1 = (y1 -y0) / (z1-z0)
     
     print("shib1:",shib1)
     
     shib2 = (y2-y1) / (z2-z1)
     
     print("shib2:",shib2)

     
     shib3=  (y3-y2) / (z3-z2)
     
     print("shib3:",shib3)
     
     
     b1 = y1 - shib1*z1
     b2 = y2 - shib2*z2
     b3 = y3 - shib3*z3
     
     b11 = y + (1/shib1)*z
     b22 = y + (1/shib2)*z
     b33 = y + (1/shib3)*z
     
     zta1 = (b11- b1)/(shib1 + 1/shib1)
     zta2 = (b22- b2)/(shib2 + 1/shib2)
     zta3 = (b33- b3)/(shib3 + 1/shib3)
          
     yta1= shib1*zta1+b1
    
     yta2= shib2*zta2+b2
     
     yta3= shib3*zta3+b3

     fase1= math.sqrt(abs((yta1-y)*(yta1-y)+(zta1-z)*(zta1-z)))
     
     fase2= math.sqrt(abs((yta2-y)*(yta2-y)+(zta2-z)*(zta2-z)))
     
     fase3= math.sqrt(abs((yta3-y)*(yta3-y)+(zta3-z)*(zta3-z)))
     
     #ax.scatter([zta1,zta2,zta3],[yta1 ,yta2 ,yta3],color="red")
     
     if zta1 > max(z1,z0) or  zta1 < min(z1,z0) :
         
             fase11 = math.sqrt(abs((y1-y)*(y1-y)+(z1-z)*(z1-z)))
             fase12 = math.sqrt(abs((y0-y)*(y0-y)+(z0-z)*(z0-z)))
             fase1 =min(fase11,fase12)
             print("fase1 shamelmishe")
             
     if zta2 > max(z2,z1) or  zta2 < min(z1,z2) :
         
             fase21 = math.sqrt(abs((y2-y)*(y2-y)+(z2-z)*(z2-z)))
             fase22 = math.sqrt(abs((y1-y)*(y1-y)+(z1-z)*(z1-z)))
             fase2 =min(fase21,fase22)
             print("fase2 shamelmishe")

     if zta3 > max(z2,z3) or  zta3 < min(z2,z3) :
         
             fase31 = math.sqrt(abs((y3-y)*(y3-y)+(z3-z)*(z3-z)))
             fase32 = math.sqrt(abs((y2-y)*(y2-y)+(z2-z)*(z2-z)))
             fase3 =min(fase31,fase32) 
             print("fase3 shamelmishe")

             
     ax.plot([z, zta1] ,[y, yta1])
     ax.plot([z, zta2] ,[y, yta2])
     ax.plot([z, zta3] ,[y, yta3])
     
     print([zta1,zta2,zta3],[yta1 ,yta2 ,yta3])
     
     if fase1 <r :
         
         print("000000000 barkhordkarde 000000")
     if fase2 <r :
         
         print("000000000 barkhordkarde 000000")         
    
     if fase3 <r :
         
         print("000000000 barkhordkarde 000000")     
     
     
     

def show_maneh(ax) :
    
    mane = pd.read_csv('ObsData.txt')

    for i in iter(mane.values.reshape(7)) :
         
         x, y , r = tuple(i.split())
         
         x , y, r= int(x)/100 , int(y)/100, int(r)/100
         
         print(x, y ,r)
        
         circle  = mpl.patches.Circle((x,y), radius=r)
        
         ax.add_patch(circle)
         
def rectangle(ax,z1,y1,z2,y2,arz) : 
    
    x= (y2-y1)/(z2-z1)
     
    if  x > 0:
         
         angle = math.atan(x)*(180/np.pi)
         
    if  x < 0 :
         
         angle = 180 - math.atan(abs(x))*(180/np.pi)

    
    tool =  math.sqrt(abs((y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)))
    
    print(angle)
    
    rect= mpl.patches.Rectangle((z1-arz/2,y1-arz/2),tool,
                           arz, angle=angle,color="red")
    ax.add_patch(rect)
    
def two_dim_forwardkinematic() :
    
    joint1 = HomogeneousMatrix()
    joint2 = HomogeneousMatrix()
    joint3 = HomogeneousMatrix()
    joint4 = HomogeneousMatrix()
    
    print("en1:",float(en1.get()))
    joint1.set_theta(float(en1.get()))
    print("en2:",float(en2.get()))
    joint1.set_d(float(en2.get()))
    print("en3:",float(en3.get()))
    joint1.set_a(float(en3.get()))
    print("en4:",float(en4.get()))
    joint1.set_alpha(float(en4.get()))
    print("en5:",float(en5.get()))
    joint2.set_theta(float(en5.get()))
    print("en6:",float(en6.get()))
    joint2.set_d(float(en6.get()))
    print("en7:",float(en7.get()))
    joint2.set_a(float(en7.get()))
    print("en8:",float(en8.get()))
    joint2.set_alpha(float(en8.get()))
    print("en9:",float(en9.get()))

    joint3.set_theta(float(en9.get()))
    print("en10:",float(en10.get()))
    joint3.set_d(float(en10.get()))
    print("en11:",float(en11.get()))
    joint3.set_a(float(en11.get()))
    print("en12:",float(en12.get()))
    joint3.set_alpha(float(en12.get()))
    print("en13:",float(en13.get()))

    joint4.set_theta(float(en13.get()))
    print("en14:",float(en14.get()))
    joint4.set_d(float(en14.get()))
    print("en15:",float(en15.get()))
    joint4.set_a(float(en15.get()))
    print("en116:",float(en16.get()))
    joint4.set_alpha(float(en16.get()))

   
    joint2.set_parent(joint1.get())
    joint3.set_parent(joint2.get())
    joint4.set_parent(joint3.get())  
    
    
    Y = np.array([joint1[1, 3], 
         joint2[1, 3], joint3[1, 3],joint4[1,3]])*-1
    
    Z = np.array([joint1[2, 3],
         joint2[2, 3], joint3[2, 3],joint4[2,3]])
    print("000000000")
    print(Y , Z)
    print("00000000000000")
    top =Toplevel()
   
    top.title("forward kinematic")
   
    top.geometry("700x700")  
    
    figure = Figure(figsize = (5, 5), 
                     dpi = 100)    
    ax = figure.add_subplot(111)
    
    chart_type = FigureCanvasTkAgg(figure, top) 
    
    chart_type.get_tk_widget().pack()


    
    ax.scatter(Z, Y, color = "green",s=100)

    
    rectangle(ax,Z[0],Y[0],Z[1],Y[1],float(en17.get()))   
    rectangle(ax,Z[1],Y[1],Z[2],Y[2],float(en18.get()))   
    rectangle(ax,Z[2],Y[2],Z[3],Y[3],float(en19.get())) 
    
    show_maneh(ax)
    
    
    for  z, y in zip(Z, Y):
        label = '(%d, %d) ' % (y, z)
        ax.text(z, y, label,size=10)

    ax.plot(Z, Y ,linewidth=2)
    
    mane = pd.read_csv('ObsData.txt')

    
    for i in iter(mane.values.reshape(7)) :
     
         x, y , r = tuple(i.split())
         
         x , y, r= int(x)/100 , int(y)/100, int(r)/100
         
         barkhord(Y,Z,x,y,r,ax)
         
    plt.show()
    
    toolbar = NavigationToolbar2Tk(chart_type, 
                                   top) 
    toolbar.update()

   
    chart_type.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
             


def calcute_invkinematic2(px,py,pz) :
    
        q3 = np.sqrt((px*px+py*py+np.power((pz-float(en6.get())),2)))
    
        theta2=(math.atan((pz-float(en6.get()))/np.sqrt(px*px+py*py)) + np.pi/2 )*(180/np.pi) 
    
        theta11=math.atan(py/px)*(180/np.pi)
        
        theta1=theta11 + 90
        
        print("#####",q3,theta2,theta1,"######")      
              
        joint1 = HomogeneousMatrix()
        joint2 = HomogeneousMatrix()
        joint3 = HomogeneousMatrix()
        joint4 = HomogeneousMatrix()
        
        joint1.set_theta(theta1)
        joint1.set_d(0)
        joint1.set_a(0)
        joint1.set_alpha(0)
        
        joint2.set_theta(0)
        joint2.set_d(float(en6.get()))
        joint2.set_a(0)
        joint2.set_alpha(theta2)
    
        joint3.set_theta(0)
        joint3.set_d(q3)
        joint3.set_a(0)
        joint3.set_alpha(0)
    
        joint4.set_theta(0)
        joint4.set_d(0)
        joint4.set_a(0)
        joint4.set_alpha(0)
    
       
        joint2.set_parent(joint1.get())
        joint3.set_parent(joint2.get())
        joint4.set_parent(joint3.get())  
        
        X = [joint1[0, 3],
             joint2[0, 3], joint3[0, 3],joint4[0,3]
        ]
        Y = [joint1[1, 3], 
             joint2[1, 3], joint3[1, 3],joint4[1,3]]
        Z = [joint1[2, 3],
             joint2[2, 3], joint3[2, 3],joint4[2,3]]
       
        print(X)
        print(Y)
        print(Z)
       
        top3 =Toplevel()
       
        top3.title("forward kinematic")
         
        top3.geometry("700x700")  
          
        figure3 = Figure(figsize = (5, 5), 
                     dpi = 100)   
    
        ax3 = figure3.gca(projection='3d')
        
        ax3.scatter3D(X, Y, Z, color = "green",s=100)
        for  x, y, z in zip(X, Y, Z):
            label = '(%d, %d, %d) ' % (x, y, z)
            ax3.text(x, y, z, label,size=10)
        #ax= Axes3D(X,Y,Z)
        ax3.plot3D(X, Y, Z)
        
        z1=[0,0],[0,0],[0,20]
        y1=[0,0],[0,20],[0,0]
        x1=[0,20],[0,0],[0,0]
        
        plot_pose(ax3,x1,y1,z1) 
        
        ax3.set_xlim3d([-100,100])
        ax3.set_ylim3d([-100, 100])
    
    #    set_axes_equal(ax)
    
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')    

    
   
      
        chart_type = FigureCanvasTkAgg(figure3, top3)
   
        chart_type.draw() 

  
        toolbar = NavigationToolbar2Tk(chart_type, 
                                   top3) 
        toolbar.update()

   
        chart_type.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
           
    
def calcute_invkinematic1(px,py,pz) :
         
        tetha = -1*math.atan(px/py)*(180/np.pi)
        
        d3=np.sqrt(px*px+py*py)
        d2=pz/2
        d1=pz/2
        
        joint1 = HomogeneousMatrix()
        joint2 = HomogeneousMatrix()
        joint3 = HomogeneousMatrix()
        joint4 = HomogeneousMatrix()
        
        joint1.set_theta(tetha)
        joint1.set_d(0)
        joint1.set_a(0)
        joint1.set_alpha(0)
        joint2.set_theta(0)
        joint2.set_d(d1)
        joint2.set_a(0)
        joint2.set_alpha(0)
    
        joint3.set_theta(0)
        joint3.set_d(d2)
        joint3.set_a(0)
        joint3.set_alpha(-90)
    
        joint4.set_theta(0)
        joint4.set_d(d3)
        joint4.set_a(0)
        joint4.set_alpha(0)
    
       
        joint2.set_parent(joint1.get())
        joint3.set_parent(joint2.get())
        joint4.set_parent(joint3.get())  
        
        X = [joint1[0, 3],
             joint2[0, 3], joint3[0, 3],joint4[0,3]
        ]
        Y = [joint1[1, 3], 
             joint2[1, 3], joint3[1, 3],joint4[1,3]]
        Z = [joint1[2, 3],
             joint2[2, 3], joint3[2, 3],joint4[2,3]]
       
        print(X)
        print(Y)
        print(Z)
       
        top3 =Toplevel()
       
        top3.title("forward kinematic")
         
        top3.geometry("700x700")  
          
        figure3 = Figure(figsize = (5, 5), 
                     dpi = 100)   
    
        ax3 = figure3.gca(projection='3d')
        
        ax3.scatter3D(X, Y, Z, color = "green",s=100)
        for  x, y, z in zip(X, Y, Z):
            label = '(%d, %d, %d) ' % (x, y, z)
            ax3.text(x, y, z, label,size=10)
        #ax= Axes3D(X,Y,Z)
        ax3.plot3D(X, Y, Z)
        
        z1=[0,0],[0,0],[0,20]
        y1=[0,0],[0,20],[0,0]
        x1=[0,20],[0,0],[0,0]
        
        plot_pose(ax3,x1,y1,z1) 
        
        ax3.set_xlim3d([-100,100])
        ax3.set_ylim3d([-100, 100])
    
    #    set_axes_equal(ax)
    
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')    

    
   
      
        chart_type = FigureCanvasTkAgg(figure3, top3)
   
        chart_type.draw() 

  
        toolbar = NavigationToolbar2Tk(chart_type, 
                                   top3) 
        toolbar.update()

   
        chart_type.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
       
        
def plot_pose(ax,x1,y1,z1) : 
    
    ax.plot3D(z1[0],z1[1],z1[2])   
    ax.plot3D(x1[0],x1[1],x1[2])     
    ax.plot3D(y1[0],y1[1],y1[2])   
    
    ax.text(z1[0][1],z1[1][1],z1[2][1],"z1",size=10)    
    ax.text(x1[0][1], x1[1][1], x1[2][1],"x1",size=10)    
    ax.text(y1[0][1], y1[1][1], y1[2][1],"y1",size=10) 

def inverse_kinematic() :
    
    option1= int(c1.get())
    option2= int(c3.get())
    option3= int(c5.get())
  
    top2 =Toplevel()
   
    top2.title("forward kinematic")
   
    top2.geometry("700x250")  
      
    #figure2 = Figure(figsize = (5, 5), 
    #             dpi = 100)   

    #ax2 = figure2.gca(projection='3d')
    

        
    v11 = StringVar(top2, value='50')  
    v22 = StringVar(top2, value='50')  
    v33 = StringVar(top2, value='50')  
 
 
    epx =Entry(top2,textvariable=v11)
    epx.place(x=50, y=200)
    epx.config(width =5)
    epy =Entry(top2,textvariable=v22)
    epy.place(x=150, y=200)
    epy.config(width =5)
    epz =Entry(top2,textvariable=v33)
    epz.place(x=250, y=200)
    epz.config(width =5)  
   
    labepx =Label(top2 , text ="px")
    labepy =Label(top2 , text ="py")
    labepz =Label(top2 , text ="pz")

    labepx.place(x =50 ,y=150)
    labepx.config(font=("mitra", 10 ,"bold"))
    labepy.place(x =150 ,y=150)
    labepy.config(font=("mitra", 10 ,"bold"))       
    labepz.place(x =250 ,y=150)
    labepz.config(font=("mitra", 10 ,"bold"))
       
    if option1==1 and option2==1 and option3==1 :       
               bottomdavarani =Button(top2 , text ="ok ",
                      command=lambda: calcute_invkinematic1(float(epx.get()),
                                                           float(epy.get()),
                                                           float(epz.get())
                                                           ))
               bottomdavarani.config(fg="red" ,padx =50, pady=10)
               bottomdavarani.pack(padx =5, pady =5)    
    if option1==2 and option2==2 and option3==1 :       

               bottomdavarani =Button(top2 , text ="ok ",
                      command=lambda: calcute_invkinematic2(float(epx.get()),
                                                           float(epy.get()),
                                                           float(epz.get())
                                                           ))
               bottomdavarani.config(fg="red" ,padx =50, pady=10)
               bottomdavarani.pack(padx =5, pady =5)         
        
    
    
    
def forwardkinematicfun() :
    
    joint1 = HomogeneousMatrix()
    joint2 = HomogeneousMatrix()
    joint3 = HomogeneousMatrix()
    joint4 = HomogeneousMatrix()
    
    print("en1:",float(en1.get()))
    joint1.set_theta(float(en1.get()))
    print("en2:",float(en2.get()))
    joint1.set_d(float(en2.get()))
    print("en3:",float(en3.get()))
    joint1.set_a(float(en3.get()))
    print("en4:",float(en4.get()))
    joint1.set_alpha(float(en4.get()))
    print("en5:",float(en5.get()))
    joint2.set_theta(float(en5.get()))
    print("en6:",float(en6.get()))
    joint2.set_d(float(en6.get()))
    print("en7:",float(en7.get()))
    joint2.set_a(float(en7.get()))
    print("en8:",float(en8.get()))
    joint2.set_alpha(float(en8.get()))
    print("en9:",float(en9.get()))

    joint3.set_theta(float(en9.get()))
    print("en10:",float(en10.get()))
    joint3.set_d(float(en10.get()))
    print("en11:",float(en11.get()))
    joint3.set_a(float(en11.get()))
    print("en12:",float(en12.get()))
    joint3.set_alpha(float(en12.get()))
    print("en13:",float(en13.get()))

    joint4.set_theta(float(en13.get()))
    print("en14:",float(en14.get()))
    joint4.set_d(float(en14.get()))
    print("en15:",float(en15.get()))
    joint4.set_a(float(en15.get()))
    print("en116:",float(en16.get()))
    joint4.set_alpha(float(en16.get()))

   
    joint2.set_parent(joint1.get())
    joint3.set_parent(joint2.get())
    joint4.set_parent(joint3.get())  
    
    X = [joint1[0, 3],
         joint2[0, 3], joint3[0, 3],joint4[0,3]
    ]
    Y = [joint1[1, 3], 
         joint2[1, 3], joint3[1, 3],joint4[1,3]]
    Z = [joint1[2, 3],
         joint2[2, 3], joint3[2, 3],joint4[2,3]]
   
    print(X)
    print(Y)
    print(Z)
    
    top =Toplevel()
   
    top.title("forward kinematic")
   
    top.geometry("700x700")  
      
    figure = Figure(figsize = (5, 5), 
                 dpi = 100)   

    ax = figure.gca(projection='3d')
    
    ax.scatter3D(X, Y, Z, color = "green",s=100)
    for  x, y, z in zip(X, Y, Z):
        label = '(%d, %d, %d) ' % (x, y, z)
        ax.text(x, y, z, label,size=10)
    #ax= Axes3D(X,Y,Z)
    ax.plot3D(X, Y, Z)
    
    z1=[0,0],[0,0],[0,20]
    y1=[0,0],[0,20],[0,0]
    x1=[0,20],[0,0],[0,0]
    
    plot_pose(ax,x1,y1,z1) 
    

    ax.set_xlim3d([-100,100])
    ax.set_ylim3d([-100, 100])

#    set_axes_equal(ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')    

   
      
    chart_type = FigureCanvasTkAgg(figure, top)
   
    chart_type.draw() 

  
    toolbar = NavigationToolbar2Tk(chart_type, 
                                   top) 
    toolbar.update()

   
    chart_type.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)





win = Tk()
win.geometry("1500x500")  
win.resizable(False, False)
win.title("GUI_robotic")


labeljoint1r =Label(win , text ="joint-1")
labeljoint1r.config(font=("mitra", 10 ,"bold"))     
labeljoint2r =Label(win , text ="joint-2")
labeljoint2r.config(font=("mitra", 10 ,"bold")) 
labeljoint3r =Label(win , text ="joint-3")
labeljoint3r.config(font=("mitra", 10 ,"bold"))         
labeljoint1r.place(x=950 ,y=50)
labeljoint2r.place(x=950 ,y=100)
labeljoint3r.place(x=950 ,y=150)

c1 = IntVar()
ch1 =Radiobutton(win , text="laghzeshi" )
ch1.config(variable=c1)
ch1.place(x=1000 ,y=50)
ch2 =Radiobutton(win , text="davarani")
ch2.config(variable=c1)    
ch2.place(x=1050 ,y=50)
ch1.config(value="1")
ch2.config(value="2")

c3 = IntVar()
ch3 =Radiobutton(win , text="laghzeshi" )
ch3.config(variable=c3)
ch3.place(x=1000 ,y=100)
ch4 =Radiobutton(win , text="davarani" )
ch4.config(variable=c3)
ch4.place(x=1050 ,y=100)
ch3.config(value="1")
ch4.config(value="2")

c5 = IntVar()
ch5 =Radiobutton(win , text="laghzeshi")
ch5.config(variable=c5)
ch5.place(x=1000 ,y=150)
ch6 =Radiobutton(win , text="davarani")
ch6.config(variable=c5)
ch6.place(x=1050 ,y=150)
ch5.config(value="1")
ch6.config(value="2")        


label1 =Label(win , text ="DH-parameter")
label1.config(font=("mitra", 10 ,"bold"))  
labeljoint1 =Label(win , text ="joint-1")
labeljoint1.config(font=("mitra", 10 ,"bold"))     
labeljoint2 =Label(win , text ="joint-2")
labeljoint2.config(font=("mitra", 10 ,"bold")) 
labeljoint3 =Label(win , text ="joint-3")
labeljoint3.config(font=("mitra", 10 ,"bold")) 
labeljoint4 =Label(win , text ="e-er")
labeljoint4.config(font=("mitra", 10 ,"bold"))          
label1.place(x=10,y=10)
labeljoint1.place(x=5,y=100)
labeljoint2.place(x=5,y=200)
labeljoint3.place(x=5,y=300)
labeljoint4.place(x=5,y=400)

v1 = StringVar(win, value='20')  
v2 = StringVar(win, value='50')  
v3 = StringVar(win, value='0')  
v4 = StringVar(win, value='90')  
v5 = StringVar(win, value='70')  
v6 = StringVar(win, value='0')  
v7 = StringVar(win, value='0')  
v8 = StringVar(win, value='90')  
v9 = StringVar(win, value='50')  
v10 = StringVar(win, value='50')  
v11 = StringVar(win, value='0')  
v12 = StringVar(win, value='0')  
v13 = StringVar(win, value='0')  
v14 = StringVar(win, value='0')  
v15 = StringVar(win, value='0')  
v16 = StringVar(win, value='0')  

v17 = StringVar(win, value='0')
v18 = StringVar(win ,  value ='0')
v19 = StringVar (win  , value ='0')


en1 =Entry(win,textvariable=v1)
en1.place(x=50, y=100)
en1.config(width =10)
en2 =Entry(win,textvariable=v2)
en2.place(x=150, y=100)
en2.config(width =10)
en3 =Entry(win,textvariable=v3)
en3.place(x=250, y=100)
en3.config(width =10)  
en4 =Entry(win,textvariable=v4)
en4.place(x=350, y=100)
en4.config(width =10) 
label21 =Label(win , text ="teta")
label22 =Label(win , text ="d")
label23 =Label(win , text ="a")
label24 =Label(win , text ="alpha ")
label21.place(x =50 ,y=60)
label21.config(font=("mitra", 10 ,"bold"))
label22.place(x =150 ,y=60)
label22.config(font=("mitra", 10 ,"bold"))       
label23.place(x =250 ,y=60)
label23.config(font=("mitra", 10 ,"bold")) 
label24.place(x =350 ,y=60)
label24.config(font=("mitra", 10 ,"bold"))

en5 =Entry(win,textvariable=v5)
en5.place(x=50, y=200)
en5.config(width =10)
en6 =Entry(win,textvariable=v6)
en6.place(x=150, y=200)
en6.config(width =10)
en7 =Entry(win,textvariable=v7)
en7.place(x=250, y=200)
en7.config(width =10)  
en8 =Entry(win,textvariable=v8)
en8.place(x=350, y=200)
en8.config(width =10) 
  


en9 =Entry(win,textvariable=v9)
en9.place(x=50, y=300)
en9.config(width =10)
en10 =Entry(win,textvariable=v10)
en10.place(x=150, y=300)
en10.config(width =10)
en11 =Entry(win,textvariable=v11)
en11.place(x=250, y=300)
en11.config(width =10)  
en12 =Entry(win,textvariable=v12)
en12.place(x=350, y=300)
en12.config(width =10) 

en13 =Entry(win,textvariable=v13)
en13.place(x=50, y=400)
en13.config(width =10)
en14 =Entry(win,textvariable=v14)
en14.place(x=150, y=400)
en14.config(width =10)
en15 =Entry(win,textvariable=v15)
en15.place(x=250, y=400)
en15.config(width =10)  
en16 =Entry(win,textvariable=v16)
en16.place(x=350, y=400)
en16.config(width =10)

en17 =Entry(win,textvariable=v17)
en17.place(x=500, y=400)
en17.config(width =10)

en18 =Entry(win,textvariable=v18)
en18.place(x=600, y=400)
en18.config(width =10)

en19 =Entry(win,textvariable=v19)
en19.place(x=700, y=400)
en19.config(width =10)

labelarz1 =Label(win , text ="ARZ_1")
labelarz1.config(font=("mitra", 10 ,"bold"))   

labelarz2 =Label(win , text ="ARZ_2")
labelarz2.config(font=("mitra", 10 ,"bold"))  

labelarz3 =Label(win , text ="ARZ_3")
labelarz3.config(font=("mitra", 10 ,"bold")) 

labelarz1.place(x =500 ,y=350)
labelarz1.config(font=("mitra", 10 ,"bold"))

labelarz2.place(x =600 ,y=350)
labelarz2.config(font=("mitra", 10 ,"bold"))

labelarz3.place(x =700 ,y=350)
labelarz3.config(font=("mitra", 10 ,"bold"))


frame = Frame( win , bd =2 , relief = "sunken") 

frame.pack()


bottom1 =Button(frame , text ="close ",command= win.destroy)
bottom1.config(fg="red" ,padx =50, pady=10)
bottom1.pack(padx =5, pady =5)


bottom2 =Button(frame , text ="forward-kinematic ", 
                command=forwardkinematicfun)
bottom2.config(fg="blue",padx =50, pady=10)
bottom2.pack(padx =5, pady =5)

bottom3 =Button(frame , text ="inverse-kinematic ",
                command=inverse_kinematic)
bottom3.config(fg="blue",padx =50, pady=10)
bottom3.pack(padx =5, pady =5)

bottom4 =Button(frame , text ="two-dim-forwardkinematic",
                command=two_dim_forwardkinematic)

bottom4.config(fg="blue",padx =50, pady=10)
bottom4.pack(padx =5, pady =5)

bottom5 =Button(frame , text ="start_simulation",
                command=start_simulation)

bottom5.config(fg="blue",padx =50, pady=10)
bottom5.pack(padx =5, pady =5)

frame.place(x= 500, y=60)
frame.config (padx =1 , pady =1)


win.mainloop()




