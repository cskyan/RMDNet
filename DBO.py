import numpy as np
import math
from sklearn.metrics import roc_auc_score

class funtion():
    def __init__(self):
        print("starting DBO")


def Bounds(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[0, i]:
            temp[i] = Lb[0, i]
        elif temp[i] > Ub[0, i]:
            temp[i] = Ub[0, i]

    return temp
def Boundss(ss, LLb, UUb):
    temp = ss
    for i in range(len(ss)):
        if temp[i] < LLb[0, i]:
            temp[i] = LLb[0, i]
        elif temp[i] > UUb[0, i]:
            temp[i] = UUb[0, i]
    return temp
def swapfun(ss):
    temp = ss
    o = np.zeros((1,len(temp)))
    for i in range(len(ss)):
        o[0,i]=temp[i]
    return o

def DBO(CnnPre, CnnLstmPre, ResNetPre, X_labels):

    pop = 30
    M = 1000 #iter
    c = 0  #ub
    d = 30000 #lb
    dim = 1
    P_percent = 0.2
    curauc = 0
    patience = 0
    pNum = round(pop*P_percent)
    lb = c*np.ones((1, dim))
    ub = d*np.ones((1, dim))
    X = np.zeros((pop, dim))
    Y = np.zeros((pop, dim))
    fit = np.zeros((pop, 1))
    CnnPre=np.array(CnnPre)
    CnnLstmPre = np.array(CnnLstmPre)
    ResNetPre = np.array(ResNetPre)
    for i in range(pop):
        X[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        Y[i, :] = lb + (ub - lb) * np.random.rand(1, dim)
        x1 = int(np.clip(np.round(X[i, 0]), 300, 29700))
        y1 = int(np.clip(np.round(Y[i, 0]), 300, 29700))
        z1 = 30000 - x1 - y1
        if z1 < 300:
            z1 = 300
            remain = 29700
            x1 = min(x1, remain)
            y1 = remain - x1
        pre = (x1 * CnnPre + y1 * CnnLstmPre + z1 * ResNetPre) / 30000
        pre1 = roc_auc_score(X_labels, pre)
        fit[i, 0] = -pre1
        X[i, :] = x1
        Y[i, :] = y1
    pFit = fit
    pX = X
    pY = Y
    XX = pX
    YY = pY
    fMin = np.min(fit[:, 0])
    bestI = np.argmin(fit[:, 0])
    bestX = X[bestI, :]
    bestX = bestX[None,:]
    bestY = Y[bestI, :]
    bestY = bestY[None, :]

    Convergence_curve = np.zeros((1, M))

    for t in range(M):
        # sortIndex = np.argsort(pFit.T)
        fmax = np.max(pFit[:, 0])
        B = np.argmax(pFit[:, 0])
        worse = X[B, :]                          #
        worse1 = Y[B, :]
        r2 = np.random.rand(1)
        # v0 = 0.5
        # v = v0 + t/(3*M)
        for i in range(pNum):
            if r2 < 0.9:
                # r1 = np.random.rand(1)
                a = np.random.rand(1)
                if a > 0.1:
                    a = 1
                else:
                    a = -1
                X[i, :] = pX[i, :]+0.3*np.abs(pX[i, :]-worse)+a*0.1*(XX[i, :]) #Equation(1)
                Y[i, :] = pY[i, :] + 0.3 * np.abs(pY[i, :] - worse1) + a * 0.1 * (YY[i, :])
            else:
                aaa = np.random.randint(180, size=1)
                if aaa == 0 or aaa == 90 or aaa == 180:
                    X[i, :] = pX[i, :]
                    Y[i, :] = pY[i, :]
                theta = aaa * math.pi/180
                X[i, :] = pX[i, :]+math.tan(theta)*np.abs(pX[i, :]-XX[i, :]) #Equation(2)
                Y[i, :] = pY[i, :] + math.tan(theta) * np.abs(pY[i, :] - YY[i, :])

            X[i, :] = Bounds(X[i, :], lb, ub)
            Y[i, :] = Bounds(Y[i, :], lb, ub)
            x1 = int(np.clip(np.round(X[i, 0]), 300, 29700))
            y1 = int(np.clip(np.round(Y[i, 0]), 300, 29700))
            z1 = 30000 - x1 - y1

            if z1 < 300:
                z1 = 300
                remain = 29700
                x1 = min(x1, remain)
                y1 = remain - x1
            pre = (x1 * CnnPre + y1 * CnnLstmPre + z1 * ResNetPre) / 30000
            pre1 = roc_auc_score(X_labels, pre)
            fit[i, 0] = -pre1
        bestII = np.argmin(fit[:, 0])
        bestXX = X[bestII, :]
        bestXX = bestXX[None,:]
        bestYY = Y[bestII, :]
        bestYY = bestYY[None,:]

        R = 1 - t/M
        Xnew1 = bestXX*(1-R)
        Xnew2 = bestXX*(1+R)
        Xnew1 = Bounds(Xnew1, lb, ub)    #Equation(3)
        Xnew2 = Bounds(Xnew2, lb,ub)
        Xnew11 = bestX*(1-R)
        Xnew22 = bestX*(1+R)             # Equation(5)
        Xnew11 = Bounds(Xnew11, lb, ub)
        Xnew22 = Bounds(Xnew22, lb, ub)
        xLB=swapfun(Xnew1)
        xUB=swapfun(Xnew2)

        Xnew1Y = bestYY * (1 - R)
        Xnew2Y = bestYY * (1 + R)
        Xnew1Y = Bounds(Xnew1Y, lb, ub)  # Equation(3)
        Xnew2Y = Bounds(Xnew2Y, lb, ub)
        Xnew11Y = bestY * (1 - R)
        Xnew22Y = bestY * (1 + R)  # Equation(5)
        Xnew11Y = Bounds(Xnew11Y, lb, ub)
        Xnew22Y = Bounds(Xnew22Y, lb, ub)
        xLB1 = swapfun(Xnew11)
        xUB1 = swapfun(Xnew22)

        for i in range(pNum, 12):      # Equation(4)
            X[i, :] = bestXX + (np.random.rand(1, dim))*(np.array(pX[i, :])-Xnew1)+(np.random.rand(1, dim))*(np.array(pX[i, :])-Xnew2)
            Y[i, :] = bestYY + (np.random.rand(1, dim)) * (np.array(pY[i, :]) - Xnew1Y) + (np.random.rand(1, dim)) * (
                        np.array(pY[i, :]) - Xnew2Y)
            X[i, :] = Bounds(X[i, :], xLB, xUB)
            Y[i, :] = Bounds(Y[i, :], xLB1, xUB1)
            x1 = int(np.clip(np.round(X[i, 0]), 300, 29700))
            y1 = int(np.clip(np.round(Y[i, 0]), 300, 29700))
            z1 = 30000 - x1 - y1
            if z1 < 300:
                z1 = 300
                remain = 29700
                x1 = min(x1, remain)
                y1 = remain - x1
            pre = (x1 * CnnPre + y1 * CnnLstmPre + z1 * ResNetPre) / 30000
            pre1 = roc_auc_score(X_labels, pre)
            fit[i, 0] = -pre1
        for i in range(13, 19):           # Equation(6)
            X[i, :] = pX[i, :]+ ((np.random.randn(1))*(pX[i, :] - Xnew11)+((np.random.rand(1, dim))*(pX[i, :]-Xnew22)))
            Y[i, :] = pY[i, :] + ((np.random.randn(1)) * (pY[i, :] - Xnew11Y) + (
                        (np.random.rand(1, dim)) * (pY[i, :] - Xnew22Y)))
            X[i, :] = Bounds(X[i, :], lb, ub)
            Y[i,:] = Bounds(Y[i, :], lb, ub)
            x1 = int(np.clip(np.round(X[i, 0]), 300, 29700))
            y1 = int(np.clip(np.round(Y[i, 0]), 300, 29700))
            z1 = 30000 - x1 - y1
            if z1 < 300:
                z1 = 300
                remain = 29700
                x1 = min(x1, remain)
                y1 = remain - x1
            pre = (x1 * CnnPre + y1 * CnnLstmPre + z1 * ResNetPre) / 30000
            pre1 = roc_auc_score(X_labels, pre)
            fit[i, 0] = -pre1
        for j in range(20, pop-1):           # Equation(7)
            X[j, :] = bestX +np.random.randn(1, dim)*(np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :]-bestX))/2
            Y[j, :] = bestY + np.random.randn(1, dim) * (np.abs(pY[j, :] - bestYY) + np.abs(pY[j, :] - bestY)) / 2
            X[j, :] = Bounds(X[j, :], lb, ub)
            Y[j, :] = Bounds(Y[j, :], lb, ub)
            x1 = int(np.clip(np.round(X[i, 0]), 300, 29700))
            y1 = int(np.clip(np.round(Y[i, 0]), 300, 29700))
            z1 = 30000 - x1 - y1
            if z1 < 300:
                z1 = 300
                remain = 29700
                x1 = min(x1, remain)
                y1 = remain - x1
            pre = (x1 * CnnPre + y1 * CnnLstmPre + z1 * ResNetPre) / 30000
            pre1 = roc_auc_score(X_labels, pre)
            fit[j, 0] = -pre1

        # Update the individual's best fitness vlaue and the global best fitness value
        XX = pX
        YY = pY
        for i in range(pop):
            if fit[i, 0] < pFit[i, 0]:
                pFit[i, 0] = fit[i, 0]
                pX[i, :] = X[i, :]
                pY[i, :] = Y[i, :]
            if pFit[i, 0] < fMin :
                fMin = pFit[i, 0]
                bestX = pX[i, :]
                bestY = pY[i, :]

        Convergence_curve[0, t] = fMin
        preauc=-fMin
        if abs(preauc < curauc):
            patience += 1
        if patience >= 10:
            break
    bestX = np.clip(np.round(bestX), 300, 29700)
    bestY = np.clip(np.round(bestY), 300, 29700)
    bestZ = 30000 - bestX - bestY

    if (bestZ < 300).any():
        bestZ = 300
        remain = 29700
        bestX = np.minimum(bestX, remain)
        bestY = remain - bestX

    x_ratio = bestX / 30000
    y_ratio = bestY / 30000
    z_ratio = bestZ / 30000

    return -fMin, x_ratio, y_ratio, z_ratio
