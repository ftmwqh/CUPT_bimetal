import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family']=['sans-serif']
plt.rcParams['font.sans-serif']=['SimHei']

sigma=5.67*10**(-8)
print(sigma)
A1=4.09*10**(-4)
A2=1.26*10**(-4)
Sbet=3.6*10**(-5)
class Bimetal(object):
    def __init__(self,ont,offt,heat_capacity,tb,h_air,epsilon):
        self.ont=ont
        self.offt=offt
        self.heat_capacity=heat_capacity
        self.tc=tb
        self.T=tb+273.15
        self.h_air=h_air
        self.epsilon=epsilon

    def P_to_environ(self,T0,Tnow):
        return self.epsilon*sigma*A1*(Tnow**4-T0**4)+self.h_air*A1*(Tnow-T0)

class Heater(object):
    def __init__(self,heat_capacity,P,tb,h_air,epsilon):
        self.heat_capacity=heat_capacity
        self.P=P
        self.tb=tb
        self.T = tb + 273.15
        self.h_air=h_air
        self.epsilon=epsilon

    def P_to_environ(self,T0,Tnow):
        return self.epsilon * sigma * A2 * (Tnow ** 4 - T0 ** 4) + self.h_air * A1 * (Tnow - T0)


def solution(b,h,h_bet,dt,N):
    b.T[0]=b.ont+273.15
    h.T[0]=b.ont+273.15
    i=0
    switch=1
    for i in range(1,N):
        if switch==1:
            b.T[i]=b.T[i-1]+(-b.P_to_environ(295.15,b.T[i-1])+h_bet*Sbet*(h.T[i-1]-b.T[i-1]))*dt/b.heat_capacity
            h.T[i]=h.T[i-1]+(h.P-h.P_to_environ(295.15,h.T[i-1])-h_bet*Sbet*(h.T[i-1]-b.T[i-1]))*dt/h.heat_capacity
            if b.T[i]>(b.offt+273.15):
                switch=0
                continue
        if switch==0:
            b.T[i] = b.T[i - 1] + (-b.P_to_environ(295.15, b.T[i - 1]) + h_bet * Sbet * (
                        h.T[i - 1] - b.T[i - 1])) * dt / b.heat_capacity
            h.T[i] = h.T[i - 1] + ( - h.P_to_environ(295.15, h.T[i - 1]) - h_bet * Sbet * (
                        h.T[i - 1] - b.T[i - 1])) * dt / h.heat_capacity
            if b.T[i]<(b.ont+273.15):
                switch=1
                t = np.arange(0, i * dt, dt)
                vib_T=t[-1]
                break
    if np.shape(t)[0]!=i:
        t=t[0:-1]
    b.tb=b.T-273.15
    h.tb=h.T-273.15
    #plt.plot(t,b.tb[0:i])
    #plt.plot(t,h.tb[0:i])
    #plt.show()
    return vib_T,t,b.tb[0:i],h.tb[0:i]

def read_sta(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    m = len(lines)
    n = len(lines[1].split(' '))
    I = np.zeros((m, n), dtype=float)

    i = 0
    for line in lines:
        list = line.strip('\n').split(' ')
        I[i] = list[0:n]
        i += 1
    I = I.T
    return I

def draw(title,b,h,sta,h_bet,dt,N):
    x,t,h,b=solution(b,h,h_bet,dt,N)
    statistics=sta
    plt.title(title)
    plt.xlabel('时间/s')
    plt.ylabel('温度/'r'$^\circ C$')
    plt.plot(t,h,label="理论双金属",color='green')
    plt.plot(t,b,label="理论加热棒",color='blue')
    plt.plot(statistics[0],statistics[1],label="实验双金属",color='black')
    plt.plot(statistics[0],statistics[2],label="实验加热棒",color='red')
    plt.legend(loc='upper right')
    plt.show()

sta45C4W=read_sta('T45C4W.txt')
sta45C12W=read_sta('T45C12W.txt')
sta45C40W=read_sta('T45C40W.txt')
sta55C4W=read_sta('T55C4W.txt')
sta55C12W=read_sta('T55C12W.txt')
sta55C40W=read_sta('T55C40W.txt')
sta65C4W=read_sta('T65C4W.txt')
sta65C12W=read_sta('T65C12W.txt')
sta65C40W=read_sta('T65C40W.txt')
sta75C4W=read_sta('T75C4W.txt')
sta75C12W=read_sta('T75C12W.txt')
sta75C40W=read_sta('T75C40W.txt')

b40=Bimetal(31.5,44,1.70,np.zeros(20000),40,0.2)
b45=Bimetal(31.5,49,1.70,np.zeros(20000),40,0.2)
b50=Bimetal(33.5,54,1.70,np.zeros(20000),40,0.2)
b55=Bimetal(34,59,1.70,np.zeros(20000),40,0.2)
b60=Bimetal(37.5,64,1.70,np.zeros(20000),40,0.2)
b65=Bimetal(41,69,1.70,np.zeros(20000),40,0.2)
b70=Bimetal(43.5,78,1.70,np.zeros(20000),40,0.2)
b75=Bimetal(46,89,1.70,np.zeros(20000),30,0.2)
h4=Heater(0.362,4,np.zeros(20000),20,0.5)
h12=Heater(0.362,12,np.zeros(20000),20,0.5)
h40=Heater(0.362,30,np.zeros(20000),20,0.5)
'''solution(b45,h4,3000,0.1,2000)
solution(b45,h12,3000,0.1,2000)
solution(b45,h40,3000,0.1,2000)

solution(b55,h4,3000,0.1,2000)
solution(b55,h12,3000,0.1,2000)
solution(b55,h40,3000,0.1,2000)

solution(b65,h4,3000,0.1,2000)
solution(b65,h12,3000,0.1,2000)
solution(b65,h40,3000,0.1,2000)

solution(b75,h4,4000,0.1,2000)
solution(b75,h12,4000,0.1,2000)
solution(b75,h40,4000,0.1,2000)'''
def gather_vid_T(Bim,sta4,sta12,sta40,h_bet):
    statistics=np.zeros(3)
    statistics[0]=sta4[0][-1]
    statistics[1] = sta12[0][-1]
    statistics[2] = sta40[0][-1]
    P1=[4,12,30]
    P=np.arange(4,40,0.1)
    Time=[]
    for i in range(0,360):
        a,b,c,d=solution(Bim,Heater(0.362,P[i],np.zeros(20000),20,0.5),h_bet,0.1,2000)
        Time.append(a)
    plt.plot(P,Time,label='理论周期')
    plt.plot(P1,statistics,marker='o',label='实验周期')
    plt.xlabel('加热棒功率/W')
    plt.ylabel('周期/s')
    plt.legend(loc='upper left')
    plt.show()

    pass
'''draw(r'45$^\circ$C双金属4W加热棒',b45,h4,read_sta('T45C4W.txt'),3000,0.1,2000)
draw(r'45$^\circ$C双金属12W加热棒',b45,h12,read_sta('T45C12W.txt'),3000,0.1,2000)
draw(r'45$^\circ$C双金属30W加热棒',b45,h40,read_sta('T45C40W.txt'),3000,0.1,2000)
draw(r'55$^\circ$C双金属4W加热棒',b55,h4,read_sta('T55C4W.txt'),3000,0.1,2000)
draw(r'55$^\circ$C双金属12W加热棒',b55,h12,read_sta('T55C12W.txt'),3000,0.1,2000)
draw(r'55$^\circ$C双金属30W加热棒',b55,h40,read_sta('T55C40W.txt'),3000,0.1,2000)
draw(r'65$^\circ$C双金属4W加热棒',b65,h4,read_sta('T65C4W.txt'),3000,0.1,2000)
draw(r'65$^\circ$C双金属12W加热棒',b65,h12,read_sta('T65C12W.txt'),3000,0.1,2000)
draw(r'65$^\circ$C双金属30W加热棒',b65,h40,read_sta('T65C40W.txt'),3000,0.1,2000)
draw(r'75$^\circ$C双金属4W加热棒',b75,h4,read_sta('T75C4W.txt'),4000,0.1,2000)
draw(r'75$^\circ$C双金属12W加热棒',b75,h12,read_sta('T75C12W.txt'),4000,0.1,2000)
draw(r'75$^\circ$C双金属30W加热棒',b75,h40,read_sta('T75C40W.txt'),4000,0.1,2000)'''

'''gather_vid_T(b45,read_sta('T45C4W.txt'),read_sta('T45C12W.txt'),read_sta('T45C40W.txt'),3000)
gather_vid_T(b55,read_sta('T55C4W.txt'),read_sta('T55C12W.txt'),read_sta('T55C40W.txt'),3000)
gather_vid_T(b65,read_sta('T65C4W.txt'),read_sta('T65C12W.txt'),read_sta('T65C40W.txt'),3000)
gather_vid_T(b75,read_sta('T75C4W.txt'),read_sta('T75C12W.txt'),read_sta('T75C40W.txt'),4000)'''

def gather_made(t1_5,t4,t12,t20,t30,t40,Bim,h_bet,title):
    statistics = np.zeros(6)
    statistics[0] = t1_5
    statistics[1] = t4
    statistics[2] = t12
    statistics[3] = t20
    statistics[4]=t30
    statistics[5]=t40
    P1 = [1.5,4, 12,20,30,40]
    P = np.arange(1.5, 40, 0.1)
    Time = []
    for i in range(0, 385):
        a, b, c, d = solution(Bim, Heater(0.362, P[i], np.zeros(20000), 20, 0.5), h_bet, 0.01, 20000)
        Time.append(a)
    plt.plot(P, Time, label='理论周期')
    plt.plot(P1, statistics, marker='o', label='实验周期')
    plt.xlabel('加热棒功率/W')
    plt.ylabel('周期/s')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

def gather_made_over70(t4,t12,t20,t30,t40,Bim,h_bet,title):
    statistics = np.zeros(5)

    statistics[0] = t4
    statistics[1] = t12
    statistics[2] = t20
    statistics[3]=t30
    statistics[4]=t40
    P1 = [4, 12,20,30,40]
    P = np.arange(4, 40, 0.1)
    Time = []
    for i in range(0, 360):
        a, b, c, d = solution(Bim, Heater(0.362, P[i], np.zeros(20000), 20, 0.5), h_bet, 0.01, 20000)
        Time.append(a)
    plt.plot(P, Time, label='理论周期')
    plt.plot(P1, statistics, marker='o', label='实验周期')
    plt.xlabel('加热棒功率/W')
    plt.ylabel('周期/s')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()

#gather_made(116,104,110,118,126,138,b40,3000,r'40$^\circ$C双金属')
#gather_made(126,108,134,140,150,170,b45,3000,r'45$^\circ$C双金属')
#gather_made(132,106,111,126,140,146,b50,3000,r'50$^\circ$C双金属')
#gather_made(144,104,118,132,148,154,b55,3000,r'55$^\circ$C双金属')
#gather_made(172,110,114,122,136,144,b60,3000,r'60$^\circ$C双金属')
#gather_made(220,96,99,108,116,128,b65,3000,r'65$^\circ$C双金属')
gather_made_over70(122,110,118,124,138,b70,2000,r'70$^\circ$C双金属')
gather_made_over70(142,126,132,136,140,b75,4000,r'75$^\circ$C双金属')



