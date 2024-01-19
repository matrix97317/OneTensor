import numpy as np
import math
np.random.seed(666)
# arr = np.random.randn(256).astype(np.float32)
# print(arr)
arr = [i for i in range(256)]
print(arr)
mean = np.mean(arr)
var = np.mean(np.power((arr-mean),2))
print(f"mean:{mean} , var:{var}")
eps=0.000001
alpha=1.00
beta=0.00
arr2 = ((arr-mean)/math.sqrt(var+eps))*alpha+beta
print(arr2)

def welford(mean,var,count,currValue):
    count += 1
    delta = currValue - mean
    mean += delta / count
    delta2 = currValue - mean
    var += delta * delta2
    return (count, mean, var)

mean_list = []
var_list = []
count_list = []
for i in range(8):
    mean = 0
    var = 0
    count = 0
    for j in range(len(arr)//8):
        newValue = arr[i*len(arr)//8+j]
        count, mean, var= welford(mean,var,count,newValue)
    # var = var / count
    mean_list.append(mean)
    var_list.append(var)
    count_list.append(count)
    # print(f"mean:{mean} , var:{var}")
    
def merge(count_list,var_list,mean_list):
    if len(var_list)==1:
        return count_list,var_list,mean_list
    new_count_list = []
    new_mean_list = []
    new_var_list = []
    for i in range(len(var_list)//2):
        count = count_list[i]+count_list[len(var_list)//2+i]
        delta = mean_list[i]-mean_list[len(var_list)//2+i]
        delta2 = delta * delta
        m = (count_list[i] * mean_list[i] + count_list[len(var_list)//2+i] * mean_list[len(var_list)//2+i]) / count
        s = var_list[i] + var_list[len(var_list)//2+i] + delta2 * ( count_list[i] * count_list[len(var_list)//2+i]) / count
        new_count_list.append(count)
        new_mean_list.append(m)
        new_var_list.append(s)
    return merge(new_count_list,new_var_list,new_mean_list)
count,var,mean = merge(count_list,var_list,mean_list)

print("count: ",count)
print("mean: ",mean)
print("var: ",var[0] / count[0])

