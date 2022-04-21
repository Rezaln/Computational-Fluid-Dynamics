#========================================================
#              2D Convection Diffusion Finite Volume
#========================================================
import time
import matplotlib
import math
import numpy as np
import matplotlib.pyplot as plt

# matplotlib qt

#============================= PARAMETERS ==================================================================
LX = 1.0                 ##~~ LENGTH X
LY = LX                  ##~~ LENGTH Y
NX = 50                  ##~~ NODES X
NY = NX                  ##~~ NODES Y

DX = LX/(NX-1)           ##~~ STEP SIZE X
DY = LY/(NY-1)           ##~~ STEP SIZE Y
U = 1.0                  ##~~ CONVECTIVITY X
V = 1.0                  ##~~ CONVECTIVITY Y

A = 1E-2                 ##~~ DIFFUSITIVITY X,Y
PI = 4.0*math.atan(1.0)  ##~~ PI=3.1416

TOL=1E-5               ##~~ CONVERGENCE TOLERANCE
MAXIT=10000             ##~~ MAX ITERATION

#=============================== INITIAL & BOUNDARY CONDITION ================================================
X = np.zeros((NX,1))          ##~~ X
Y = np.zeros((NY,1))          ##~~ Y
for I in range(NX):
    X[I,0]=I*DX
    
for I in range(NY):
    Y[I,0]=I*DY

TC = np.zeros((NY,NX));        ##~~ TEMPERATURE.CENTRAL DIFFERENCING SCHEME
T1 = np.zeros((NY,NX));        ##~~ TEMPERATURE.1st UPWIND DIFFERENCING SCHEME
T2 = np.zeros((NY,NX));        ##~~ TEMPERATURE.2nd UPWIND DIFFERENCING SCHEME
TH = np.zeros((NY,NX));        ##~~ TEMPERATURE.HYBRID DIFFERENCING SCHEME
TE = np.zeros((NY,NX));        ##~~ TEMPERATURE.POWER LAW DIFFERENCING SCHEME

BCL = 0.0;                ##~~ BOUNDARY CONDITION, LEFT
BCR = 0.0;                ##~~ BOUNDARY CONDITION, RIGHT
BCB = 100;                ##~~ BOUNDARY CONDITION, BOTTOM
BCT = 100;                ##~~ BOUNDARY CONDITION, TOP

TC[:,0]=BCL; TC[:,NX-1]=BCR; TC[0,:]=BCT; TC[NY-1,:]=BCB;
T1[:,0]=BCL; T1[:,NX-1]=BCR; T1[0,:]=BCT; T1[NY-1,:]=BCB;
T2[:,0]=BCL; T2[:,NX-1]=BCR; T2[0,:]=BCT; T2[NY-1,:]=BCB;
TH[:,0]=BCL; TH[:,NX-1]=BCR; TH[0,:]=BCT; TH[NY-1,:]=BCB;
TE[:,0]=BCL; TE[:,NX-1]=BCR; TE[0,:]=BCT; TE[NY-1,:]=BCB;


#==================================== SOLUTION . CENTRAL ==========================================================
tic1= time.time()
K=1;                    ##~~ ITERATION
ERR = 1+TOL;              ##~~ STEP BY STEP ERROR
TCOLD = TC.copy();               ##~~ INITIAL GUESS

while (ERR>TOL and K<MAXIT):
    for I1 in range(2,NY):
   
        if (I1==2 or I1==NY-1):
            SP=-2*A/DY
        else:
            SP=0
        
        for I2 in range(2,NX):   
       
            TERM1=0.5*U*(TCOLD[I1-1,I2+1-1]-TCOLD[I1-1,I2-1-1]);
            TERM2=0.5*V*(TCOLD[I1+1-1,I2-1]-TCOLD[I1-1-1,I2-1]);
            SU=X[I2-1]*Y[I1-1];
            TERM3=A*(((TCOLD[I1-1,I2+1-1]+TCOLD[I1-1,I2-1-1])/DX)+((TCOLD[I1+1-1,I2-1]+TCOLD[I1-1-1,I2-1])/DY));
            if (I2==2 or I2==NX-1):
                SP=-2*A/DX
            else:
                SP=0;
            
            COEFF=-2*A*((1/DX)+(1/DY)) + SP;
            RIGHT=TERM1 + TERM2 - SU - TERM3;
            TC[I1-1,I2-1]=RIGHT/COEFF;
        
    
    ERR=np.linalg.norm(abs(TC-TCOLD));
    TCOLD=TC.copy()
    K=K+1
    
    
toc1=time.time()    
print ("Central Iterations {}".format(K))
print('Central Execution Time:{:11.3f}s'.format(toc1-tic1))

#===================================== SOLUTION . UPWIND ===========================================================
tic2= time.time()
K=1;                    ##~~ ITERATION
ERR=1+TOL;              ##~~ STEP BY STEP ERROR
T1OLD=T1.copy();               ##~~ INITIAL GUESS

while (ERR>TOL and K<MAXIT):
    for I1 in range(2,NY):
    
        if (I1==2 or I1==NY-1):
            SP=-2*A/DY
        else:
            SP=0
            
        for I2 in range(2,NX):
        
            TERM1=U*(T1OLD[I1-1,I2-1-1]);
            TERM2=V*(T1OLD[I1-1-1,I2-1]);
            SU=X[I2-1]*Y[I1-1];
            TERM3=A*(((T1OLD[I1-1,I2+1-1]+T1OLD[I1-1,I2-1-1])/DX)+((T1OLD[I1+1-1,I2-1]+T1OLD[I1-1-1,I2-1])/DY));
            if (I2==2 or I2==NX-1):
                SP=-2*A/DX;
            else:
                SP=0;
            
            COEFF=2*A*((1/DX)+(1/DY)) - SP + (U+V);
            RIGHT=TERM1 + TERM2 + SU + TERM3;
            T1[I1-1,I2-1]=RIGHT/COEFF;
        
    
    ERR=np.linalg.norm(abs(T1-T1OLD));
    T1OLD=T1.copy();
    K=K+1;

toc2=time.time()
print ("1st UPWIND Iterations {}".format(K))
print('1st UPWIND Execution Time:{:11.3f}s'.format(toc2-tic2))

#========================================= SOLUTION . UPWIND2 ==========================================================
tic3=time.time()
K=1;                    ##~~ ITERATION
ERR=1+TOL;              ##~~ STEP BY STEP ERROR
T2OLD=T2.copy();               ##~~ INITIAL GUESS

while (ERR>TOL and K<MAXIT):
    for I1 in range(2,NY):
    
        if (I1==2 or I1==NY-1):
            SP=-2*A/DY;
        else:
            SP=0;
            
        for I2 in range(2,NX):
         
            if (I2==2):
                TERM1=U*(T2OLD[I1-1,I2-1-1])
            else:
                TERM1=-U*((3/8)*T2OLD[I1-1,I2+1-1] - (7/8)*T2OLD[I1-1,I2-1-1]+ (1/8)*T2OLD[I1-1,I2-2-1]);
            
            if (I1==2):
                TERM2=V*(T2OLD[I1-1-1,I2-1])
            else:
                TERM2=-V*((3/8)*T2OLD[I1+1-1,I2-1] - (7/8)*T2OLD[I1-1-1,I2-1]+ (1/8)*T2OLD[I1-2-1,I2-1]);
            
            SU=X[I2-1]*Y[I1-1];
            TERM3=A*(((T2OLD[I1-1,I2+1-1]+T2OLD[I1-1,I2-1-1])/DX)+((T2OLD[I1+1-1,I2-1]+T2OLD[I1-1-1,I2-1])/DY));
            if (I2==2 or I2==NX-1):
                SP=-2*A/DX;
            else:
                SP=0;
            
            COEFF=2*A*((1/DX)+(1/DY)) - SP + 1.5*(U+V)/4.0;
            RIGHT=TERM1 + TERM2 + SU + TERM3;
            T2[I1-1,I2-1]=RIGHT/COEFF;
       
    
    ERR=np.linalg.norm(abs(T2-T2OLD));
    T2OLD=T2.copy();
    K=K+1;

toc3=time.time()
print ("2nd UPWIND Iterations {}".format(K))
print('2nd UPWIND Execution Time:{:11.3f}s'.format(toc3-tic3))

#======================================== SOLUTION . HYBRID ===========================================================
tic4= time.time()
K=1;                    ##~~ ITERATION
ERR=1+TOL;              ##~~ STEP BY STEP ERROR
THOLD=TH.copy();               ##~~ INITIAL GUESS

PEC=U/(A/DX)
while (ERR>TOL and K<MAXIT):
    for I1 in range(2,NY):

        if (I1==2 or I1==NY-1):
            SP=-2*A/DY;
        else:
            SP=0;

        for I2 in range(2,NX):

            if (PEC < 2.0):
                TERM1=0.5*U*(THOLD[I1-1,I2+1-1]-THOLD[I1-1,I2-1-1]);
                TERM2=0.5*V*(THOLD[I1+1-1,I2-1]-THOLD[I1-1-1,I2-1]);
            else:
                TERM1=U*(THOLD[I1-1,I2-1-1]);
                TERM2=V*(THOLD[I1-1-1,I2-1]);

            SU=X[I2-1]*Y[I1-1];
            TERM3=A*(((THOLD[I1-1,I2+1-1]+THOLD[I1-1,I2-1-1])/DX)+((THOLD[I1+1-1,I2-1]+THOLD[I1-1-1,I2-1])/DY));
            if (I2==2 or I2==NX-1):
                SP=-2*A/DX;
            else:
                SP=0;

            COEFF=2*A*((1/DX)+(1/DY)) - SP + (U+V);
            RIGHT=TERM1 + TERM2 + SU + TERM3;
            TH[I1-1,I2-1]=RIGHT/COEFF;


    ERR=np.linalg.norm(abs(TH-THOLD));
    THOLD=TH.copy();
    K=K+1;
    
toc4=time.time()
print ("HYBRID Iterations {}".format(K))
print('HYBRID Execution Time:{:11.3f}s'.format(toc4-tic4))

#=================================== SOLUTION . POWER LAW ========================================================
tic5= time.time()
K=1;                    ##~~ ITERATION
ERR=1+TOL;              ##~~ STEP BY STEP ERROR
TEOLD=T1.copy();               ##~~ INITIAL GUESS

EX=((1 - 0.1*PEC)**5)/PEC;
while (ERR>TOL and K<MAXIT):
    for I1 in range(2,NY):

        if (I1==2 or I1==NY-1):
            SP=-2*A/DY;
        else:
            SP=0;

        for I2 in range(2,NX):

            TERM1=-(EX*(TE[I1-1,I2+1-1])+(1+EX)*(TE[I1-1,I2-1-1]));
            TERM2=-(EX*(TE[I1+1-1,I2-1])+(1+EX)*(TE[I1-1-1,I2-1]));
            SU=X[I2-1]*Y[I1-1];
            TERM3=A*(((TEOLD[I1-1,I2+1-1]+TEOLD[I1-1,I2-1-1])/DX)+((TEOLD[I1+1-1,I2-1]+TEOLD[I1-1-1,I2-1])/DY))
            if (I2==2 or I2==NX-1):
                SP=-2*A/DX;
            else:
                SP=0;

            COEFF=2*A*((1/DX)+(1/DY)) - SP + (U*(1+2*EX)+V*(-2-EX));
            RIGHT=-TERM1 - TERM2 + SU + TERM3;
            TE[I1-1,I2-1]=RIGHT/COEFF;

    ERR=np.linalg.norm(abs(TE-TEOLD));
    TEOLD=TE.copy();
    K=K+1;

toc5 = time.time
print ("POWER LAW Iterations : {}".format(K))
print('POWER LAW Execution Time:{:11.3f}s'.format(toc5-tic5))

## PLOT ========================================================================
X, Y = np.meshgrid(X,Y)

plt.figure()
plt.contourf(X,Y,TC)
plt.clabel(plt.contourf(X,Y,TC))
#colorbar
plt.colorbar()
#colormap(jet)
plt.set_cmap('jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title('T.CENTRAL')
plt.grid()
#axis square

plt.figure()
plt.contourf(X,Y,T1)
plt.clabel(plt.contourf(X,Y,T1))
plt.colorbar()
#colormap(jet)
plt.set_cmap('jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title('T.UPWIND')
plt.grid()
#axis square

plt.figure()
plt.contourf(X,Y,T2)
plt.clabel(plt.contourf(X,Y,T2))
plt.colorbar()
plt.set_cmap('jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title('T.UPWIND 2nd')
plt.grid()
#axis square

plt.figure()
plt.contourf(X,Y,TH)
plt.clabel(plt.contourf(X,Y,TH))
plt.colorbar()
plt.set_cmap('jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title('T.HYBRID')
plt.grid()
#axis square

plt.figure()
plt.contourf(X,Y,TE)
plt.clabel(plt.contourf(X,Y,TE))
plt.colorbar()
plt.set_cmap('jet')
plt.xlabel('x')
plt.ylabel('y')
plt.title('T.POWER LAW')
plt.grid()
#axis square
plt.show()