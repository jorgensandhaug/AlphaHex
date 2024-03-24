# cython: language_level=3
import numpy as np
cimport numpy as np
cimport libc.stdlib



import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
handler = logging.FileHandler('logfile.log')
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(handler)


cdef class UnionFind:
    cdef:
        int *parent
        int *rank
        int size

    def __cinit__(self, int size):
        self.size = size
        self.parent = <int *>libc.stdlib.malloc(size * sizeof(int))
        self.rank = <int *>libc.stdlib.calloc(size, sizeof(int))
        if not self.parent or not self.rank:
            raise MemoryError()
        for i in range(size):
            self.parent[i] = i


    def __dealloc__(self):
        libc.stdlib.free(self.parent)
        libc.stdlib.free(self.rank)

    cdef int find(self, int x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    cdef void union(self, int x, int y):
        cdef int root_x, root_y
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

cdef int get_index(int i, int j, int size):
    return i * size + j



# Use a numpy array for directions
cdef np.ndarray directions = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)

cdef dict memo = {}
#@cython.boundscheck(False)  # Deactivate bounds checking
#@cython.wraparound(False)   # Deactivate negative indexing.
cpdef bint has_winning_path_cython(np.ndarray[int, ndim=2] b, int size, int player):
    #logger.info(f'Inputs: b={b}, size={size}, player={player}, direction={directions}')
    # Add a dictionary to store previous results

    # Create a mask of the board where the player's pieces are
    #cdef np.ndarray[int, ndim=2] mask = (b == player).astype(np.int32)
    
    # Convert the mask to bytes to use as a dictionary key
    #cdef bytes mask_bytes = mask.tobytes()
    
    # Check if the result is already in the memo dictionary
    #if mask_bytes in memo:
        #return memo[mask_bytes]
        

    cdef:
        UnionFind uf = UnionFind(size * size + 2)
        int virtual_node_start = size * size
        int virtual_node_end = size * size + 1
        int i, j, current, ni, nj, di, dj, index

    if player == 1:
        # Connect virtual nodes
        for i in range(size):
            for j in range(size):
                #logger.info(f'{i, j, player, b[i, j]}')
                if b[i, j] == 1:
                    current = get_index(i, j, size)

                    # Connect with virtual nodes if on the corresponding edge
                    if j == 0:
                        uf.union(current, virtual_node_start)
                    if j == size - 1:
                        uf.union(current, virtual_node_end)

                    # Connect with adjacent cells of the same player
                    for index in range(0, len(directions), 2):
                        di, dj = directions[index], directions[index + 1]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size and b[ni, nj] == 1:
                            neighbor = get_index(ni, nj, size)
                            uf.union(current, neighbor)
    else:
        # Connect virtual nodes
        for i in range(size):
            for j in range(size):
                #logger.info(f'{i, j, player, b[i, j]}')
                if b[i, j] == -1:
                    current = get_index(i, j, size)

                    # Connect with virtual nodes if on the corresponding edge
                    if i == 0:
                        uf.union(current, virtual_node_start)
                    if i == size - 1:
                        uf.union(current, virtual_node_end)

                    # Connect with adjacent cells of the same player
                    for index in range(0, len(directions), 2):
                        di, dj = directions[index], directions[index + 1]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size and b[ni, nj] == -1:
                            neighbor = get_index(ni, nj, size)
                            uf.union(current, neighbor)
    
    
    # Store the result in the memo dictionary before returning it
    res = uf.find(virtual_node_start) == uf.find(virtual_node_end)
    #memo[mask_bytes] = res
    return res

