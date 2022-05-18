import numpy as np

#encoding=utf-8

import numpy as np
import random
import time
import math

class Bayes(object):
    def __init__(self, N):
        self.N = N
        self.values = np.zeros((N,N))
        for i in range(N):
            self.values[i,i] = 1
        self.vsum = np.ones((N,1))
    
    def clean(self):
        for i in range(self.N):
            for j in range(self.N):
                self.values[i,j] = math.floor(self.values[i,j]*10.0/self.vsum[i])
            self.vsum[i] = 10
    
    def add_path(self, i,j):
        #assert i>=0 and i < self.N
        #assert j>=0 and j < self.N
        
        self.values[i,j] += 1
        self.vsum[i] += 1
        return True
        
    def get_path(self, i):
        seed = random.randint(0,self.vsum[i])
        tsum = 0
        for j in range(self.N):
            if seed < tsum+self.values[i,j]:
                return j
            tsum += self.values[i,j]
        return self.N-1
        
    def get_add_path(self, i,j):#
        #return i
        #assert i>=0 and i < self.N
        #assert j>=0 and j < self.N
        self.add_path(i,j)
        '''
        for i in range(self.N):
            for j in range(self.N):
                print(self.values[i,j],' ',end='')
            print('')
        print(i,j,self.get_path(i))
        #'''
        return self.get_path(i)

zero_threshold = 0.00000001

class KMNode(object):
    def __init__(self, id, exception=0, match=None, visit=False):
        self.id = id
        self.exception = exception
        self.match = match
        self.visit = visit


class KuhnMunkres(object):
    def __init__(self, N):
        self.N = N
        self.matrix = None
        self.x_nodes = []
        self.y_nodes = []
        self.minz = float('inf')
        self.x_length = 0
        self.y_length = 0
        self.index_x = 0
        self.index_y = 1
        self.x_y_values = np.zeros((N*N,3))
        for i in range(N):
            for j in range(N):
                self.x_y_values[i*N+j,0] = i
                self.x_y_values[i*N+j,1] = j
        self.set_matrix(self.x_y_values)
        self.km()

    def __del__(self):
        pass

    def set_matrix(self, x_y_values):
        xs = set()
        ys = set()
        for x, y, value in x_y_values:
            xs.add(x)
            ys.add(y)

        #选取较小的作为x
        if len(xs) <= len(ys):
            self.index_x = 0
            self.index_y = 1
        else:
            self.index_x = 1
            self.index_y = 0
            xs, ys = ys, xs

        x_dic = {x: i for i, x in enumerate(xs)}
        y_dic = {y: j for j, y in enumerate(ys)}
        self.x_nodes = [KMNode(x) for x in xs]
        self.y_nodes = [KMNode(y) for y in ys]
        self.x_length = len(xs)
        self.y_length = len(ys)

        self.matrix = np.zeros((self.x_length, self.y_length))
        for row in x_y_values:
            x = row[self.index_x]
            y = row[self.index_y]
            value = row[2]
            x_index = x_dic[x]
            y_index = y_dic[y]
            self.matrix[x_index, y_index] = value

        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i, :])


    def km(self):
        for i in range(self.x_length):
            while True:
                self.minz = float('inf')
                self.set_false(self.x_nodes)
                self.set_false(self.y_nodes)

                if self.dfs(i):
                    break

                self.change_exception(self.x_nodes, -self.minz)
                self.change_exception(self.y_nodes, self.minz)

    #"""
    def dfs(self, i):
        x_node = self.x_nodes[i]
        x_node.visit = True
        for j in range(self.y_length):
            y_node = self.y_nodes[j]
            if not y_node.visit:
                t = x_node.exception + y_node.exception - self.matrix[i][j]
                if abs(t) < zero_threshold:
                    y_node.visit = True
                    if y_node.match is None or self.dfs(y_node.match):
                        x_node.match = j
                        y_node.match = i
                        return True
                else:
                    if t >= zero_threshold:
                        self.minz = min(self.minz, t)
        return False
    #"""

    '''
    def dfs(self, i):
        match_list = []
        while True:
            x_node = self.x_nodes[i]
            x_node.visit = True
            for j in range(self.y_length):
                y_node = self.y_nodes[j]
                if not y_node.visit:
                    t = x_node.exception + y_node.exception - self.matrix[i][j]
                    if abs(t) < zero_threshold:
                        y_node.visit = True
                        match_list.append((i, j))
                        if y_node.match is None:
                            self.set_match_list(match_list)
                            return True
                        else:
                            i = y_node.match
                            break
                    else:
                        if t >= zero_threshold:
                            self.minz = min(self.minz, t)
            else:
                return False
    '''
    
    def set_match_list(self, match_list):
        for i, j in match_list:
            x_node = self.x_nodes[i]
            y_node = self.y_nodes[j]
            x_node.match = j
            y_node.match = i

    def set_false(self, nodes):
        for node in nodes:
            node.visit = False

    def change_exception(self, nodes, change):
        for node in nodes:
            if node.visit:
                node.exception += change

    def get_connect_result(self):
        ret = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            y_node = self.y_nodes[j]
            x_id = x_node.id
            y_id = y_node.id
            value = self.matrix[i][j]

            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            ret.append((x_id, y_id, value))

        return ret

    def get_max_value_result(self):
        ret = 0
        for i in range(self.x_length):
            j = self.x_nodes[i].match
            ret += self.matrix[i][j]

        return ret
    
    def clean(self):
        for i in range(self.N):
            for j in range(self.N):
                self.x_y_values[i*self.N+j,2] = 0
        self.set_matrix(self.x_y_values)
        self.km()
    
    def add_path(self, i,j):
        #assert i>=0 and i < self.N
        #assert j>=0 and j < self.N
        
        self.x_y_values[i*self.N+j,2] += 1
        return True
        '''
        value_max = 1000
        if(self.x_y_values[i*self.N+j,2]<value_max):
            self.x_y_values[i*self.N+j,2] += 1
            return True
        return False
        '''
    def get_path(self, i):
        #assert j>=0 and j < self.N
        x_node = self.x_nodes[i]
        j = x_node.match
        y_node = self.y_nodes[j]
        x_id = x_node.id
        y_id = y_node.id
        #assert x_id==i
        return int(y_id)
    def get_add_path(self, i,j):#
        #return i
        #assert i>=0 and i < self.N
        #assert j>=0 and j < self.N
        if(self.add_path(i,j)):
            if(self.get_path(i) == j):
                return j
            self.set_matrix(self.x_y_values)
            self.km()
            #print(self.get_connect_result())
        #print(i)
        #print(self.x_y_values)
        return self.get_path(i)
            


def run_kuhn_munkres(x_y_values):
    process = KuhnMunkres(5)
    process.set_matrix(x_y_values)
    process.km()
    return process.get_connect_result()


def test():
    values = []
    random.seed(0)
    for i in range(500):
        for j in range(1000):
            value = random.random()
            values.append((i, j, value))

    return run_kuhn_munkres(values)

if __name__ == '__main__':
    s_time = time.time()
    #ret = test()
    print("time usage: %s " % str(time.time() - s_time))
    values = [
        (1, 1, 3),
        (1, 3, 4),
        (2, 1, 2),
        (2, 2, 1),
        (2, 3, 3),
        (3, 2, 4),
        (3, 3, 5)
    ]
    values = [
        (1, 1, 10),
        (1, 2, 0),
        (1, 3, 0),
        (2, 1, 10),
        (2, 2, 10),
        (2, 3, 0),
        (3, 1, 10),
        (3, 2, 10),
        (3, 3, 10),
        (4, 1, 5),
        (4, 2, 100),
        (4, 4, 10),
        (5, 5, 10)
    ]
    values = [
        (1, 1, 2),
        (1, 2, 1),
        (1, 3, 1),
        (1, 4, 1),
        (1, 5, 1),
        (2, 1, 1),
        (2, 2, 2),
        (2, 3, 1),
        (2, 4, 1),
        (2, 5, 1),
        (3, 1, 1),
        (3, 2, 1),
        (3, 3, 2),
        (3, 4, 1),
        (3, 5, 1),
        (4, 1, 1),
        (4, 2, 1),
        (4, 3, 1),
        (4, 4, 2),
        (4, 5, 1),
        (5, 1, 2),
        (5, 2, 1),
        (5, 3, 1),
        (5, 4, 1),
        (5, 5, 2),
    ]
    print(run_kuhn_munkres(values))

'''
class KM():
    def __init__(self,N):
        super().__init__()
        # 声明数据结构
        #i:dst j:src
        self.N = N
        self.adj_matrix = np.zeros((N,N))# np array with dimension N*N
     
        # 初始化顶标
        self.label_left = np.max(self.adj_matrix, axis=1)  # init label for the left set
        self.label_right = np.zeros(N)  # init label for the right set
     
        # 初始化匹配结果
        self.match_right = np.empty(N) * np.nan
     
        # 初始化辅助变量
        self.visit_left = np.empty(N) * False
        self.visit_right = np.empty(N) * False
        self.slack_right = np.empty(N) * np.inf
     
    # 寻找增广路，深度优先
    def find_path(self,i):
        self.visit_left[i] = True
        for j, match_weight in enumerate(self.adj_matrix[i]):
            if self.visit_right[j]:
                continue  # 已被匹配（解决递归中的冲突）
            gap = self.label_left[i] + self.label_right[j] - match_weight
            if gap == 0:
                # 找到可行匹配
                self.visit_right[j] = True
                if np.isnan(self.match_right[j]) or self.find_path(self.get_path(j)):  ## j未被匹配，或虽然j已被匹配，但是j的已匹配对象有其他可选备胎
                    self.match_right[j] = i
                    return True
                else:
                # 计算变为可行匹配需要的顶标改变量
                    if self.slack_right[j] < gap:
                        self.slack_right[j] = gap
        return False
    def add_path(self, i,j):
        #assert i>=0 and i < self.N
        #assert j>=0 and j < self.N
        self.adj_matrix[i,j] += 1
    def get_path(self, j):
        #assert j>=0 and j < self.N
        return int(self.match_right[j])
    def get_add_path(self, i,j):
        return i
        assert i>=0 and i < self.N
        assert j>=0 and j < self.N
        self.add_path(i,j)
        if(self.get_path(j) == i):
            return i
        self.KM()
        return self.get_path(j)
        
    # KM主函数
    def KM(self):
        # 初始化顶标
        self.label_left = np.max(self.adj_matrix, axis=1)  # init label for the left set
        self.label_right = np.zeros(self.N)  # init label for the right set
        # 初始化匹配结果
        self.match_right = np.empty(self.N) * np.nan
        
        for i in range(self.N):
            # 重置辅助变量
            self.slack_right = np.empty(self.N) * np.inf
            while True:
                # 重置辅助变量
                self.visit_left = np.empty(self.N) * False
                self.visit_right = np.empty(self.N) * False
                
                  # 能找到可行匹配
                if self.find_path(i):
                    break
                # 不能找到可行匹配，修改顶标
                # (1)将所有在增广路中的X方点的label全部减去一个常数d 
                # (2)将所有在增广路中的Y方点的label全部加上一个常数d
                d = np.inf
                for j, slack in enumerate(self.slack_right):
                    if not self.visit_right[j] and slack < d:
                        d = slack
                for k in range(self.N):
                    if self.visit_left[k]:
                        self.label_left[k] -= d
                for n in range(self.N):
                    if self.visit_right[n]: 
                        self.label_right[n] += d
        res = 0
        for j in range(self.N):
            if self.match_right[j] >=0 and self.match_right[j] < self.N:
                print(self.match_right[j],j)
        #       res += adj_matrix[self.match_right[j]][j]
        #return res
        
km = KM(3)
km.KM()
'''