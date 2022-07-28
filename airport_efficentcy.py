## Returns efficiency score between 0 and 1
import pandas as pd
import numpy as np

def airport_eff(port: str,arr: list):
    #print(port)
    departs = 0
    ports = [arr[3] for x in arr]
    print(ports)
    for ele in ports:
        if ele[3] == port:
            #print(ele)
            departs += 1
    print(port, departs)
    #return(departs)


if __name__ == "__main__":
    df = pd.read_csv('data/train.csv', nrows=200)
    array =df.values.tolist()
    arr = []
    for ele in array:
        arr.append(airport_eff(ele[3], array))
    #print(arr)