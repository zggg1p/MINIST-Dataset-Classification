import matplotlib.pyplot as plt
fp=open(r"G:\人工智能2\简单全连接实现MINIST（1个隐藏层，sigmoid激活函数）log.txt","r",encoding='utf8')
fp2=open(r"G:\人工智能2\简单全连接实现MINIST（1个隐藏层，sigmoid激活函数）acc.txt","r",encoding='utf8')
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
while True:
    num=fp.readline()
    if not num:
        break
    temp=num.split("|")
    list1.append(float(temp[0].split()[1]))
    list2.append(float(temp[1].split()[1]))
    list3.append(float(temp[2].split()[1][:-2])/100)
fp.close()
while True:
    num2=fp2.readline()
    if not num2:
        break
    temp=num2.split(",")
    list4.append(float(temp[0][6:]))
    list5.append(float(temp[1].split()[1][:-2])/100)
fp2.close()
plt.rcParams['font.sans-serif']=['SimHei']
fig,axes=plt.subplots(figsize=(10,6))
plt.subplots_adjust(hspace=0.4)
plt.subplot(221)
plt.title("Training Acc")
plt.xlabel("Iterations")
plt.ylabel("Acc",rotation=90)
plt.plot(list1,list3,color='r',label="Training Acc")
plt.legend(loc='lower right')
plt.subplot(222)
plt.title("Training Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss",rotation=90)
plt.plot(list1,list2,color='g',label="Training Loss")
plt.legend(loc='upper right')
plt.subplot(223)
plt.title("Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Acc",rotation=90)
plt.plot(list4,list5,color='b',label="Test Acc")
plt.legend(loc='lower right')
plt.show()
