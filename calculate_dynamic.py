# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:07:36 2021

@author: ASUS
"""

from sympy import sin, cos, Matrix,simplify
from sympy import symbols
from sympy.abc import rho, phi
import numpy as np

def Rz(q) :
    
    return Matrix([[cos(q) , -sin(q) ,0],
                   [sin(q) , cos(q) , 0],
                   [0,       0,         1]])
    

q1dot,q2dot,q3dot,q1,q2,q3 = symbols('q1dot q2dot q3dot q1 q2 q3')

m1,l1,lc1,I1= symbols(" m1 l1 lc1 I1 ")
m2,l2,lc2,I2= symbols(" m2 l2 lc2 I2 ")
m3,l3,lc3,I3= symbols(" m3 l3 lc3 I3 ")
g= symbols("g")
q =Matrix([q1 ,q2,q3])

qdot=Matrix([q1dot,q2dot,q3dot])

Pc1=Matrix([lc1*cos(q1),
     lc1*sin(q1),
     0])

Pc2=Matrix([l1*cos(q1)+lc2*cos(q1+q2),
      l1*sin(q1)+lc2*sin(q1+q2),
      0])
Pc3=Matrix([l1*cos(q1)+l2*cos(q1+q2)+lc3*cos(q1+q2+q3),
      l1*sin(q1)+lc2*sin(q1+q2)+lc3*sin(q1+q2+q3),
      0])

Vc1a=Pc1.jacobian(q)*qdot
Vc2a=Pc2.jacobian(q)*qdot
Vc3a=Pc3.jacobian(q)*qdot

W1=q1dot*Matrix([0 ,0, 1])
W2=W1+q2dot*Matrix([0, 0 ,1])
W3=W2+q3dot*Matrix([0, 0 ,1])

V0=0

Vc1b=W1.cross(Rz(q1)* Matrix([lc1, 0, 0 ]))

V1= W1.cross(Rz(q1)*Matrix([l1, 0, 0 ]))

Vc2b=W2.cross(Rz(q1+q2)*Matrix([lc2 , 0  ,0 ]))


K1=1/2*m1*(Vc1a.T)*Vc1a + 1/2*I1*W1.T*W1
K2=1/2*m2*(Vc2a.T*Vc2a) + 1/2*I2*W2.T*W2
K3=1/2*m3*(Vc3a.T*Vc3a) + 1/2*I3*W3.T*W3
KK= K1+K2+K3


G=Matrix([0, g ,0 ])

P1=m1*G.T*Pc1;
P2=m2*G.T*Pc2;
P3=m3*G.T*Pc3;
PP=P1+P2+P3

D_mat=KK.jacobian(qdot)
D_mat=D_mat.jacobian(qdot)

n=3

C_mat= Matrix([[0,0,0],
               [0,0,0],
               [0,0,0]])

for k in range(n):
    
    for j in range(n):
        
        C_mat[k,j]=0*g
        
        for i in range(n):
            
            C_mat[k,j]=Matrix([C_mat[k,j]])+1/2*(Matrix([D_mat[k,j]]).jacobian(Matrix([q[i]])) + Matrix([D_mat[k,i]]).jacobian(Matrix([q[j]])) - Matrix([D_mat[i,j]]).jacobian(Matrix([q[k]]))*qdot[i])
        

G_vec= simplify(PP.jacobian(q).T)
C= simplify(C_mat)
D= simplify(D_mat)


