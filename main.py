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
 
TRIVIAL_MULTIPLICATION_BOUND = 1

def strassen_mul(left, right):
	assert(left.shape == right.shape)
	assert(left.shape[0] == left.shape[1])

	if left.shape[0] <= TRIVIAL_MULTIPLICATION_BOUND:
		return left.dot(right)

	assert(left.shape[0] % 2 == 0)
	return np.block(
		strassen_mul_2x2(*map(split_to_2x2_blocks, [left, right]))
	)

def matrix_generator(degree, low, high):
    size = 2**degree
    return np.random.uniform(size=(size, size), low=low, high=high)

A = matrix_generator(2, -100, 100)
B = matrix_generator(2, -100, 100)


print(A)
print(B)
