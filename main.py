import time
from itertools import product
import numpy as np

def split_to_2x2_blocks(matrix):
	return list(map(
		lambda row: np.hsplit(row, 2),
		np.vsplit(matrix, 2)
	))

def strassen_mul_2x2(lb, rb):
	d = strassen_mul(lb[0][0] + lb[1][1], rb[0][0] + rb[1][1])
	d_1 = strassen_mul(lb[0][1] - lb[1][1], rb[1][0] + rb[1][1])
	d_2 = strassen_mul(lb[1][0] - lb[0][0], rb[0][0] + rb[0][1])

	left = strassen_mul(lb[1][1], rb[1][0] - rb[0][0])
	right = strassen_mul(lb[0][0], rb[0][1] - rb[1][1])
	top = strassen_mul(lb[0][0] + lb[0][1], rb[1][1])
	bottom = strassen_mul(lb[1][0] + lb[1][1], rb[0][0])

	return [[d + d_1 + left - top, right + top],
			[left + bottom, d + d_2 + right - bottom]]
 
def trivial_mul(left, right):
	height, mid_size = left.shape
	mid_size, width = right.shape

	result = np.zeros((height, width))
	for row, col, mid in product(*map(range, [height, width, mid_size])):
		result[row][col] += left[row][mid] * right[mid][col]

	return result

TRIVIAL_MULTIPLICATION_BOUND = 1

def strassen_mul(left, right):
	assert(left.shape == right.shape)
	assert(left.shape[0] == left.shape[1])

	if left.shape[0] <= TRIVIAL_MULTIPLICATION_BOUND:
		return trivial_mul(left, right)

	assert(left.shape[0] % 2 == 0)
	return np.block(
		strassen_mul_2x2(*map(split_to_2x2_blocks, [left, right]))
	)

def matrix_generator(size, low, high):
    return np.random.uniform(size=(size, size), low=low, high=high)

def test_strassen(degree, buffer):
	size = 2**degree
	total_time = 0
	number_of_attempts = 10
	for _ in range(0, number_of_attempts):
		A = matrix_generator(size, -1000, 1000)
		B = matrix_generator(size, -1000, 1000)
		print("Start:", size)
		start = time.time()
		res = strassen_mul(A, B)
		total_time += (time.time() - start) * 1000
		print("Done!")
	
	buffer.append([size, total_time / number_of_attempts])

def experiment():
	buffer_strassen = []
	for degree in range(0, 9):
		test_strassen(degree, buffer_strassen)

	return buffer_strassen
        
strassen = experiment()

