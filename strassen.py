
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import sys
import math

def standard (m1, m2):
    if m1.shape[1] != m2.shape[0]: 
        raise Exception("mismatched input shapes")
    res = np.dot(m1,m2)
    return res

def strassen (m1, m2, n0):
    # assume powers of 2
    if m1.shape[1] != m2.shape[0]: 
        raise Exception("mismatched input shapes")
    if m1.shape[0] <= n0:
        res = standard(m1,m2)
    else:
        padding = False
        
        # check if needs padding
        if m1.shape[0] % 2 != 0:
            padding = True
            m1 = np.pad(m1, ((0,1), (0,1)), mode='constant', constant_values=0)
            m2 = np.pad(m2, ((0,1), (0,1)), mode='constant', constant_values=0)
        
        n = int(m1.shape[0]/2)

        a = m1[:n, :n]        
        b = m1[:n, n:]
        c = m1[n:, :n]
        d = m1[n:, n:]
        e = m2[:n, :n]
        f = m2[:n, n:]
        g = m2[n:, :n]
        h = m2[n:, n:]

        p1 = strassen(a,np.subtract(f,h), n0)
        p2 = strassen(np.add(a,b),h, n0)
        p3 = strassen(np.add(c,d),e, n0)
        p4 = strassen(d,np.subtract(g,e), n0)
        p5 = strassen(np.add(a,d),np.add(e,h), n0)
        p6 = strassen(np.subtract(b,d),np.add(g,h), n0)
        p7 = strassen(np.subtract(c,a),np.add(e,f), n0)
        
        q1 = np.add(np.add(np.subtract(p4,p2), p5), p6)
        q2 = np.add(p1, p2)
        q3 = np.add(p3, p4)
        q4 = np.add(np.add(np.subtract(p1, p3), p5), p7)

        top = np.concatenate((q1,q2), axis=1)
        bottom = np.concatenate((q3,q4), axis=1)
        res = np.concatenate((top,bottom), axis=0)

        if padding:
            res = res[:m1.shape[0]-1,:m1.shape[0]-1]

    return res

def generate_bin(n):
    res = np.zeros((n,n),int)
    for i in range(n):
        for j in range(n):
            res[i][j] = random.randint(0,1)
    return res

def generate_two(n):
    two = [0,1,2]
    res = np.zeros((n,n),int)
    for i in range(n):
        for j in range(n):
            res[i][j] = two[random.randrange(len(two))]
    return res

def generate_neg(n):
    neg = [-1,1]
    res = np.zeros((n,n),int)
    for i in range(n):
        for j in range(n):
            res[i][j] = neg[random.randrange(len(neg))]
    return res

def test_standard():
    m1 = np.array([[1, 2, 3,4],[4,5,6,7],[8,9,10,11],[12,13,14,15]])
    m2 = np.array([[10,11,12,13],[20,21,22,23],[30,31,32,33],[40,41,42,43]])
    if np.array_equal(standard(m1,m2),strassen(m1,m2,1)):
        print("standard mult pass")
   
def test_strassen():
    # m1 = generate_bin(9)
    # m2 = generate_bin(9)
    # m3 = generate_neg(9)
    # m4 = generate_neg(9)
    # m5 = generate_two(9)
    # m6 = generate_two(9)
    # if np.array_equal(standard(m1,m2),strassen(m1,m2,1)):
    #     print("strassen bin pass")
    # if np.array_equal(standard(m3,m4), strassen(m3,m4,1)):
    #     print("strassen neg pass")
    # if np.array_equal(standard(m5,m6), strassen(m5,m6,1)):
    #     print("strassen two pass")

    m1 = generate_bin(257)
    m2 = generate_bin(257)
    print(m1,m2)
    conv = standard(m1,m2)
    print(conv)
    stras = strassen(m1,m2,20)
    print(stras)


def generate_txt(filename, n):
    m1 = generate_bin(n)
    m2 = generate_bin(n)
    with open(filename, 'w') as f:
        for i in range(n):
            for j in range(n):
                f.write(str(m1[i][j])+"\n")
        for i in range(n):
            for j in range(n):
                f.write(str(m2[i][j])+"\n")

def main(argv):

    n = int(argv[1])
    inputfile = argv[2]

    m1 = np.zeros((n,n),int)
    m2 = np.zeros((n,n),int)

    with open(inputfile) as f:
        for i in range(n):
            for j in range(n):
                m1[i][j] = f.readline().strip()
        for i in range(n):
            for j in range(n):
                m2[i][j] = f.readline().strip()

    res = strassen(m1,m2,20)
    for i in np.diagonal(res):
        print(i)

def task2(size):
    n = size
    trials = 2

    data = []
    m1 = generate_bin(n)
    m2 = generate_bin(n)
    for n0 in range(2,n,10):
        total = 0
        for t in range(trials):
            start = time.time()
            res = strassen(m1,m2,n0)
            end = time.time()
            total += (end-start)
        avg = total/trials
        data.append((avg,n0))
        print("n0="+str(n0)+ ": "+str(avg))

    opt = min(data)
    print("Optimal n0: ", opt)
    x = [i[1] for i in data]
    y = [i[0] for i in data]
    plt.plot(x,y)
    plt.xlabel("n0")
    plt.ylabel("Average Strassen Runtime Across "+str(trials)+" Trials (s)")
    plt.title("Optimal n0 for Modified Strassen for "+str(n)+" Matrix")
    plt.show()

def gen_matrix(p):
  m = np.random.choice(2, (1024,1024), p=[1-p, p])
  return m

def task3():
  probs = [0.01,0.02,0.03,0.04,0.05]
  expected = [math.comb(1024,3)*pow(p,3) for p in probs]
  data = []
  for p in probs:
    a = gen_matrix(p)
    res = strassen(strassen(a,a,76),a,76)
    triangles = sum(np.diagonal(res))/6
    data.append(triangles)

  error = [(data[i]-expected[i])/expected[i] for i in range(len(probs))]
  print(expected)
  print(data)
  print(error)
  
  plt.plot(probs,expected,'-g',label='Expected')
  plt.plot(probs,data,'r--',label='Experimental')
  plt.legend(loc="upper left")
  plt.xlabel("Edge Inclusion Probability")
  plt.ylabel("Number of Triangles")
  plt.title("Triangles in Random Graphs")
  plt.show()

if __name__ == "__main__":
    # generate_txt("test_1.txt",1)
    main(sys.argv[1:])
    # task2(256)
