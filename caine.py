import time
import random
import numpy as np
import sys

arguments = sys.argv

# if dim = 32, for example, you are multiplying two 32x32 matrices together
type = int(arguments[1])
dim = int(arguments[2])
input_file = arguments[3]

if type == 1:
    from matplotlib import pyplot as plt
# Get the command line arguments

def convAlgo(mat1, mat2):
    n = len(mat1)
    assert len(mat1) == len(mat1[0])
    assert len(mat2) == len(mat2[0])
    if False:
        C = [[0 for i in range(n)] for j in range(n)]

        for i in range(n):
            for j in range(n):
                sum = 0
                for k in range(n):
                    sum = sum + mat1[i][k] * mat2[k][j]

                C[i][j] = sum
    else:
        C = np.dot(mat1, mat2)
    return C

def EfficientAlgo(mat1, mat2, k):
    n = len(mat1)
    #assert len(mat1) == len(mat1[0])
    #assert len(mat2) == len(mat2[0])

    if(n <= k):
        return convAlgo(mat1, mat2)
    else:
        # Get the top left quadrant
        A11 = mat1[:n//2, :n//2]
        B11 = mat2[:n//2, :n//2]

        # Get the top right quadrant
        A12 = mat1[:n//2, n//2:]
        B12 = mat2[:n//2, n//2:]

        # Get the bottom left quadrant
        A21 = mat1[n//2:, :n//2]
        B21 = mat2[n//2:, :n//2]

        # Get the bottom right quadrant
        A22 = mat1[n//2:, n//2:]
        B22 = mat2[n//2:, n//2:]

        M1 = EfficientAlgo(np.add(A11, A22), np.add(B11, B22), k)
        M2 = EfficientAlgo(np.add(A21, A22), B11, k)
        M3 = EfficientAlgo(A11, np.subtract(B12, B22), k)
        M4 = EfficientAlgo(A22, np.subtract(B21, B11), k)
        M5 = EfficientAlgo(np.add(A11, A12), B22, k)
        M6 = EfficientAlgo(np.subtract(A21, A11), np.add(B11, B12), k)
        M7 = EfficientAlgo(np.subtract(A12, A22), np.add(B21, B22), k)

        C11 = np.add(np.subtract(np.add(M1, M4), M5), M7)
        C12 = np.add(M3, M5)
        C21 = np.add(M2, M4)
        C22 = np.add(np.add(np.subtract(M1, M2), M3), M6)

        top = np.concatenate((C11, C12), axis=1)
        bottom = np.concatenate((C21, C22), axis=1)
        matrix = np.concatenate((top, bottom), axis=0)

        return matrix

if False:
    # The inputfile is an ASCII file with 2d**2 integer numbers, one per line, representing two matrices A and B
    MAT1 = []
    MAT2 = []
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            if i < dim**2:
                MAT1.append([int(x) for x in line.split()])
            else:
                MAT2.append([int(x) for x in line.split()])

    # Resiez MAT1 and MAT2 from dim**2 to dim*dim
    MAT1 = np.array(MAT1).reshape(dim, dim)
    MAT2 = np.array(MAT2).reshape(dim, dim)

    assert len(MAT1) == dim
    assert len(MAT2) == dim

    # Add padding 0s to ensure that the matrices are square to a power of 2**n
    if dim & (dim - 1) != 0:
        # Find the next power of 2
        power = 0
        while 2**power < dim:
            power += 1
        # Add padding
        MAT1 = np.pad(MAT1, ((0, 2**power - dim), (0, 2**power - dim)), 'constant')
        MAT2 = np.pad(MAT2, ((0, 2**power - dim), (0, 2**power - dim)), 'constant')

    mat = EfficientAlgo(np.array(MAT1), np.array(MAT2), 15)

    # print diagonal values of mat
    for i in range(dim):
        print(mat[i][i])


if False:
    MAT1 = np.array([[1,2,3,4],[5,6,7,8],[9, 10, 11, 12], [13, 14, 15, 16]])
    MAT2 = np.array([[1,2,3,4],[5,6,7,8],[9, 10, 11, 12], [13, 14, 15, 16]])
    mat = EfficientAlgo(MAT1, MAT2, 1)

# Find the optimal value for n0
if False:
    SIZE = 128
    MAT1 = np.random.randint(0, 2, size=(SIZE, SIZE))
    MAT2 = np.random.randint(0, 2, size=(SIZE, SIZE))

    mat = EfficientAlgo(MAT1, MAT2, 1)

    # Run num_tests tries for each k and get the average time taken
    num_tests = 1
    k_times = []
    for k in range(2, SIZE):
        times = []
        for i in range(num_tests):
            start = time.time()
            mat = EfficientAlgo(MAT1, MAT2, k)
            end = time.time()
            times.append(end - start)
        avg_time = sum(times) / len(times)
        print("k =", k, "Average time taken:", avg_time)
        k_times.append((k, avg_time))

    # Plot the results
    x = [kt[0] for kt in k_times]
    y = [kt[1] for kt in k_times]
    plt.plot(x, y)
    plt.xlabel("k")
    plt.ylabel("Average time taken (s)")
    plt.title("Optimal k for EfficientAlgo")
    plt.show()

# take each of the values of p 5 times, find the average number of trianlges for each p value,
# then plot the actual and expected # of triangles with the y-axis being
# the number of triangles and the x-axis being the value of p.
if False:
    # Define the values of p
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

    num_trials = 5

    num_triangles = np.zeros(len(p_values))

    for i, p in enumerate(p_values):
        for j in range(num_trials):
            # Generate a 1024x1024 matrix of random values between 0 and 1
            rand_matrix = np.random.rand(1024, 1024)

            # Convert the matrix into a binary adjacency matrix using the given probability p
            adj_matrix = np.where(rand_matrix < p, 1, 0)

            # Run the EfficientAlgo on the adjacency matrix
            mat = EfficientAlgo(adj_matrix, adj_matrix, 15)
            mat = EfficientAlgo(mat, adj_matrix, 15)

            # To count the number of triangles in in graph, we can simply add the entries in the diagonal, and divide by 6.
            num_triangles[i] += np.sum(np.diagonal(mat)) / 6

        num_triangles[i] /= num_trials

    # expected number is (1024 choose 3) p^3
    expected_triangles = [(1024 * 1023 * 1022) / (3 * 2 * 1) * p**3 for p in p_values]

    # Plot the results
    x = np.arange(len(p_values))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, num_triangles, width, label='Actual')
    rects2 = ax.bar(x + width/2, expected_triangles, width, label='Expected')

    ax.set_xlabel('Value of p')
    ax.set_ylabel('Number of Triangles')
    ax.set_xticks(x)
    ax.set_xticklabels(p_values)
    ax.legend()

    fig.tight_layout()
    plt.show()


# Remove later
if False:
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

    expected_num_triangles = [178.433, 1427.464, 4817.692, 11419.714, 22304.128]
    actual_num_triangles = [176, 1470.5, 4820.0, 11231.5, 22664.0]

    # Create a bar chart
    x = np.arange(len(p_values))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, expected_num_triangles, width, label='Expected')
    rects2 = ax.bar(x + width/2, actual_num_triangles, width, label='Actual')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Value of p')
    ax.set_ylabel('Number of triangles')
    ax.set_title('Number of triangles for different values of p')
    ax.set_xticks(x)
    ax.set_xticklabels(p_values)
    ax.legend()

    fig.tight_layout()
    plt.show()