import numpy as np
import random
import math
from scipy.special import gamma
import sys
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


def levy(d):
    beta = 1.5

    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2)) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)) ** (
                1 / beta)

    u = np.random.rand(d) * sigma
    v = np.random.rand(d)
    step = np.real(u / (np.abs(v) ** (1 / beta)))

    L = 0.01 * step
    return L

def DBO(CnnPre, CnnLstmPre, ResNetPre, X_labels):

    pop = 30
    M = 50 #
    c = 0  #ub
    d = 10000 #lb
    dim = 1
    P_percent = 0.2
    pNum = round(pop*P_percent)
    lb = c*np.ones((1, dim))
    ub = d*np.ones((1, dim))
    X = np.zeros((pop, dim))
    Y = np.zeros((pop, dim))
    X[1, :] = np.random.rand(1) * 10000
    Y[1, :] = np.random.rand(1) * 10000
    for i in range(pop-1):
        X[i+1, :] = X[i,:]+0.5-np.mod(2.2 * np.sin(2 * np.pi * X[i,:]) / (2 * np.pi), 1)
        Y[i+1, :] = Y[i,:]+0.5-np.mod(2.2 * np.sin(2 * np.pi * Y[i,:]) / (2 * np.pi), 1)
        if X[i+1, :] > 10000:
            X[i + 1, :] = 10000
        if X[i+1, :] < 0:
            X[i + 1, :] = 0

    fit = np.zeros((pop, 1))
    CnnPre=np.array(CnnPre)
    CnnLstmPre = np.array(CnnLstmPre)
    ResNetPre = np.array(ResNetPre)
    for i in range(pop):
        # X[i, :] = lb+(ub-lb)*np.random.rand(1, dim)
        pre = (np.round(X[i,:])*CnnPre + np.round(Y[i,:])*CnnLstmPre + (30000-np.round(X[i,:])-np.round(Y[i,:]))*ResNetPre) / 30000
        pre1 = roc_auc_score(X_labels, pre)
        fit[i, 0] = -pre1
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
            pre = (np.round(X[i,:])*CnnPre + np.round(Y[i,:])*CnnLstmPre + (30000-np.round(X[i,:])-np.round(Y[i,:]))*ResNetPre)/300
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
            pre = (np.round(X[i,:])*CnnPre + np.round(Y[i,:])*CnnLstmPre + (30000-np.round(X[i,:])-np.round(Y[i,:]))*ResNetPre) / 30000
            pre1 = roc_auc_score(X_labels, pre)
            fit[i, 0] = -pre1
        for i in range(13, 19):           # Equation(6)
            X[i, :] = pX[i, :]+ ((np.random.randn(1))*(pX[i, :] - Xnew11)+((np.random.rand(1, dim))*(pX[i, :]-Xnew22)))
            Y[i, :] = pY[i, :] + ((np.random.randn(1)) * (pY[i, :] - Xnew11Y) + (
                        (np.random.rand(1, dim)) * (pY[i, :] - Xnew22Y)))
            X[i, :] = Bounds(X[i, :], lb, ub)
            Y[i,:] = Bounds(Y[i, :], lb, ub)
            pre = (np.round(X[i,:])*CnnPre + np.round(Y[i,:])*CnnLstmPre + np.round((30000-X[i,:]-X[i,:]))*ResNetPre)/30000
            pre1 = roc_auc_score(X_labels, pre)
            fit[i, 0] = -pre1
        for j in range(20, pop-1):           # Equation(7)
            X[j, :] = levy(dim)*bestX + np.random.randn(1, dim)*(np.abs(pX[j, :] - bestXX) + np.abs(pX[j, :]-bestX))/2
            Y[j, :] = levy(dim)*bestY + np.random.randn(1, dim) * (np.abs(pY[j, :] - bestYY) + np.abs(pY[j, :] - bestY)) / 2
            X[j, :] = Bounds(X[j, :], lb, ub)
            Y[j, :] = Bounds(Y[j, :], lb, ub)
            pre = (np.round(X[j,:])*CnnPre + np.round(Y[j,:])*CnnLstmPre + (30000-np.round(X[j,:])-np.round(X[j,:]))*ResNetPre)/30000
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

    return fMin, bestX, bestY




