import numpy as np
from scipy import linalg

def laplacian(p1,p2,p3,p4,d_min):
    # print("Executing trajectory")
    d_12 = linalg.norm(p1 - p2)
    d_13 = linalg.norm(p1 - p3)
    d_14 = linalg.norm(p1 - p4)
    d_23 = linalg.norm(p2 - p3)
    d_24 = linalg.norm(p2 - p4)
    d_34 = linalg.norm(p3 - p4)

    d = np.array([d_12, d_13, d_14, d_23, d_24, d_34])
    l = np.array([0,0,0,0,0,0])

    for i in range(6):
        if d[i] < d_min:
            l[i] = -1
        else:
            l[i] = 0

    l_11 = abs(l[0] + l[1] + l[2])
    l_22 = abs(l[0] + l[3] + l[4])
    l_33 = abs(l[1] + l[3] + l[5])
    l_44 = abs(l[2] + l[4] + l[5])

    L_t = np.matrix([[l_11, l[0], l[1], l[2]], [l[0], l_22, l[3], l[4]], [l[1], l[3], l_33, l[5]], [l[2], l[4], l[5], l_44]])


    return L_t , np.asarray(l)

def laplacian2(p1,p2, d_min):
    # print("Executing trajectory")
    d_12 = linalg.norm(p1 - p2)


    d = np.array([d_12])
    l = np.array([0])


    if d[0] < d_min:
        l[0] = -1
    else:
        l[0] = 0

    l_11 = abs(l[0])
    l_22 = abs(l[0])


    L_t = np.matrix([[l_11, l[0]], [l[0], l_22]])


    return L_t , np.asarray(l)

def laplacian_timevarying(p1,p2,p3,p4,l_prev,d_min,d_min_2):
    # print("Executing trajectory")
    d_12 = linalg.norm(p1 - p2)
    d_13 = linalg.norm(p1 - p3)
    d_14 = linalg.norm(p1 - p4)
    d_23 = linalg.norm(p2 - p3)
    d_24 = linalg.norm(p2 - p4)
    d_34 = linalg.norm(p3 - p4)

    d = [d_12, d_13, d_14, d_23, d_24, d_34]
    l = l_prev

    for i in range(6):
        if d[i] < d_min_2:
           l[i] = 1
        elif d[i] >= d_min_2 and d[i] <= d_min:
            l[i] = ((d[i] - d_min)**2)/((d_min_2 - d_min)**2)
        elif d[i] > d_min:
            l[i] = 0    

    l_11 = abs(l[0] + l[1] + l[2])
    l_22 = abs(l[0] + l[3] + l[4])
    l_33 = abs(l[1] + l[3] + l[5])
    l_44 = abs(l[2] + l[4] + l[5])

    L_t = np.matrix([[l_11, -l[0], -l[1], -l[2]], [-l[0], l_22, -l[3], -l[4]], [-l[1], -l[3], l_33, -l[5]], [-l[2], -l[4], -l[5], l_44]])

    return L_t , np.asarray(l)        

def laplacian_timevarying_3(p1,p2,p3,l_prev,d_min,d_min_2):
        # print("Executing trajectory")
    d_12 = linalg.norm(p1 - p2)
    d_13 = linalg.norm(p1 - p3)
    d_23 = linalg.norm(p2 - p3)

    d = [d_12, d_13, d_23]
    l = l_prev

    for i in range(3):
        if d[i] < d_min_2:
           l[i] = 1
        elif d[i] >= d_min_2 and d[i] <= d_min:
            l[i] = ((d[i] - d_min)**2)/((d_min_2 - d_min)**2)    

    l_11 = abs(l[0] + l[1])
    l_22 = abs(l[0] + l[2])
    l_33 = abs(l[1] + l[2])       

    L_t = np.matrix([[l_11, -l[0], -l[1]], [-l[0], l_22, -l[2]], [-l[1], -l[2], l_33]])

    return L_t , np.asarray(l)  

if __name__ == "__main__":

    a,b = laplacian_timevarying_3(np.array([3,0,0]),np.array([1,0,1]),np.array([0,1,0]),np.array([1.0,0.0,0]),3.0,1.0)
    print(a)
    print(b)
    