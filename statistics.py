'''import pandas as pd
import csv
from multiprocessing import Pool
import time
def getdata(url):
    tb=pd.read_html(url)[0]
    tb.to_csv(r'E:\py_projects\bimetal_vib',mode='a',encoding='utf_8_sig')
    time.sleep(0.5)

def myprocesspool(num=10):
    pool=Pool(num)
    results=pool.map(getdata,urls)
    pool.close()
    pool.join()
    return results
getdata('https://baike.baidu.com/item/%E9%93%9C%E5%BA%B7%E9%93%9C%E7%83%AD%E7%94%B5%E5%81%B6/2160963?fr=aladdin')

'''
import numpy as np
f=open('1标定.txt','r')
lines=f.readlines()
m=len(lines)
print('height=',m)
n=len(lines[1].split(' '))
print('width=',n)
I=np.zeros((m,n),dtype=float)
i=0
for line in lines:
    list=line.strip('\n').split(' ')
    I[i]=list[0:n]
    i+=1
print(I)
aves=np.zeros(n)
aves=(I[0]+I[1])/2
print(aves)
np.savetxt('ave1',aves)
np.savetxt('ave1',I)
f.close()

def UtoT(filename,newname):
    f = open(filename, 'r')
    lines = f.readlines()
    m = len(lines)
    print('height=', m)
    n = len(lines[1].split(' '))
    print('width=', n)
    I = np.zeros((m, n), dtype=float)


    i = 0
    for line in lines:
        list = line.strip('\n').split(' ')
        I[i] = list[0:n]
        i += 1
    if m!=3:
        I=I.T
    #print(I)

    I[0]=I[0]-I[0][0]


    I[1]=(I[1]-0.0005)/0.0425
    I[2]=(I[2]+0.0027)/0.04522
    #print(I)
    np.savetxt(newname,I.T)
    return I
#UtoT('U75C4W.txt','T75C4W.txt')
#UtoT('U75C12W.txt','T75C12W.txt')
#UtoT('U75C30W.txt','T75C30W.txt')
#UtoT('U65C4W.txt','T65C4W.txt')
#UtoT('U65C12W.txt','T65C12W.txt')
#UtoT('U65C30W.txt','T65C30W.txt')
#UtoT('U55C4W.txt','T55C4W.txt')
#UtoT('U55C12W.txt','T55C12W.txt')
#UtoT('U75C40W.txt','T75C40W.txt')
#UtoT('U65C40W.txt','T65C40W.txt')
#UtoT('U55C40W.txt','T55C40W.txt')
#UtoT('U45C4W.txt','T45C4W.txt')
#UtoT('U45C12W.txt','T45C12W.txt')
UtoT('U45C40W.txt','T45C40W.txt')






'''def UtoT_snow(filename,newname):
    f = open(filename, 'r')
    lines = f.readlines()
    m = len(lines)
    print('height=', m)
    n = len(lines[0].split('\t'))
    print('width=', n)
    I = np.zeros((m, n), dtype=float)


    i = 0
    for line in lines:
        list = line.strip('\n').split('\t')
        I[i]=list[0:n]




    I[0]=(I[0]-0.0005)/0.0425

    np.savetxt(newname,I.T)
    return I
#UtoT_snow('23.998.txt','23.998T.txt')
#UtoT_snow('17.txt','17T.txt')
#UtoT_snow('11.txt','11T.txt')
UtoT_snow('8.txt','8T.txt')'''