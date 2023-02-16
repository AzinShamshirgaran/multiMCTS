#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:11:11 2021

@author: shamano
"""
import dataset as ds
import math
import numpy as np
#import scipy.io as sio
#from scipy.stats import expon
#import ongp as op
BUFFER_SIZE=50000

VERTEX_MIN = 10
VERTEX_MAX = 50
MAX_REWARD = 1
MAX_X = 1
MAX_Y = 1
VERBOSE=True



class OrienteeringVertex:
    def __init__(self,x,y,value,idv):
        self.x = x
        self.y = y
        self.value = value
        #self.edges = list()
        self.edges = {}
        self.id = idv
        self.visited = False
        
    def get_coordinates(self):
        return self.x,self.y
        
    def get_value(self):#OrienteeringGraph.list_of_visited_vertices(self),
        return self.value
    
    #def get_ongp_value(self,lst):#OrienteeringGraph.list_of_visited_vertices(self),
       # return op.get_gp( lst, [self.x,self.y])
    

    def set_value(self,value):
        self.value = value
        
    def get_edges(self):
        #return self.edges
        return self.edges.values()
    
    def add_edge(self,edge):
        if edge.source != self:
            raise Exception("Edge does not originate from vertex")
        #self.edges.append(edge)
        self.edges[edge.dest.id] = edge
    
    def set_visited(self):
        self.visited = True
        
    def clear_visited(self):
        self.visited = False
    
    def get_visited(self):
        return self.visited
        
    def add_edge_to_vertex(self,to,alpha):
        if to == self:
            raise Exception("Can't add loop edges")
            
        # avoid adding duplicate edges    
        #for e in self.edges:
        #    if e.dest == to:
        #        return
        #newedge = OrienteeringEdge(self,to,alpha)
        #self.edges.add(newedge)
        if self.edges.has_key(to):
            return
        newedge = OrienteeringEdge(self,to,alpha)
        self.edges[to] = newedge
        
    def get_edge_towards_vertex(self,idx):  # returns the edge towards a vertex with a given id
        if idx not in self.edges:
            raise Exception("Vertex from {} to {} does not exist".format(self.id,idx))
        return self.edges[idx]
        

class OrienteeringEdge:
    def __init__(self,source,dest,alpha):
        self.source = source
        self.dest = dest
        self.alpha = alpha
        self.d = math.sqrt((self.source.x-self.dest.x)**2 + (self.source.y-self.dest.y)**2) #abs( (self.source.x-self.dest.x) )+ abs((self.source.y-self.dest.y) ) #abs( (self.source.x-self.dest.x) )+ abs((self.source.y-self.dest.y) )
        self.offset = self.alpha * self.d
        self.parameter = (1-self.alpha)*self.d
        self.buffer = self.offset + np.random.uniform(0,1, size=BUFFER_SIZE) #np.random.exponential(scale=self.parameter,size=BUFFER_SIZE)
        self.buffer_index = 0


    # check this -- should be fine
    def sample_cost(self):
        if self.buffer_index == BUFFER_SIZE:
            self.regenerate_buffer()
        value = self.buffer[self.buffer_index]
        self.buffer_index += 1
        return value
    
    def regenerate_buffer(self):
        self.buffer = self.offset + np.random.exponential(scale=self.parameter,size=BUFFER_SIZE)
        self.buffer_index = 0
    
    def sample_cost_parallel(self,number=1):
       #return self.offset + np.random.exponential(self.parameter) 
        #3if self.buffer_index == BUFFER_SIZE:
         #   self.regenerate_buffer()
        
        value = np.zeros((number,))
        
        if self.buffer_index + number > BUFFER_SIZE:
            self.regenerate_buffer()

        value[0:number] = self.buffer[self.buffer_index:self.buffer_index+number]
        self.buffer_index += number

        return value
        
def set_verbose():
    global VERBOSE
    VERBOSE=True

def set_silent():
    global VERBOSE
    VERBOSE=False
        
class OrienteeringGraph:
    def __init__(self):
        #if type(fname) == str:
        #if VERBOSE:
           # print('Loading graph ',fname,' ....')
        #content = sio.loadmat('2018-06-21_ripperdan.mat')
        alpha = 0.5#contents['alpha'][0][0]

        #coords = contents['xy']
        #print("coord {}".format(coords))
        #contents = sio.loadmat('graph_test_10.mat')
        #sensor_reading = content['krig_val']
        #print("sensory_reading {}".format(sensor_reading))
        #edge_list = contents['edge_list']
        #print("edges {}".format(edge_list))
        """
        coords= [[0,0],[32, 13], [23, 39], [43, 18], [16, 38], [41, 21],[26, 26],
             [ 7, 42], [ 2, 46], [32, 43], [18, 18], [48,  8],[17, 23], [41, 31], [39, 33],
               [27, 42], [13, 44], [43,3], [46, 42], [12, 16], [23,1], [9,7], [17,  1], [31, 17],
                [36,38],  [13,48], [23, 20], [12, 14], [34, 40], [50,50]]
        value_sensory_reading = [0.25, 0.028, 0.549, 0.435, 0.41, 0.330, 0.204, 0.119,
                                 0.299, 0.0766, 0.25, 0.028, 0.549, 0.435, 0.41, 0.130, 0.204, 0.619,
                                 0.099, 0.1266, 0.45, 0.028, 0.549, 0.435, 0.41, 0.530, 0.04, 0.219,
                                 0.219, 0.16]
        """
        coords= ds.coords

        value_sensory_reading =ds.value_sensory_reading


        #value_sensory_reading=[]
        #for i in coords:
         #   value_sensory_reading.append(sensor_reading[i[0],i[1]])
        #print(value_sensory_reading)
        npoints = len(coords)
        self.vertices = dict()
        self.total_reward = -1

        for i in range(npoints):
            nv = OrienteeringVertex(coords[i][0],coords[i][1],value_sensory_reading[i],i) #sensor_reading[i,1]
            self.vertices[i]= nv


        for i in range(npoints):
            for j in range(npoints):
                if (i != j): #and ( i != self.end_vertex ):  # no loops and end vertex has no outgoing edges
                    edge = OrienteeringEdge(self.vertices[i], self.vertices[j], alpha)
                    self.vertices[i].add_edge(edge)

        #nedges = edge_list.shape[0]
        #for i in range(nedges):
         #   edge = OrienteeringEdge(self.vertices[edge_list[i][0]-1], self.vertices[edge_list[i][1]-1], alpha)
          #  self.vertices[edge_list[i][0]-1].add_edge(edge)

        self.set_start_vertex(0)
        self.end_vertex = npoints - 1
        self.nvertices = npoints
        #self.budget = 2 #contents['t_max'][0][0]

        self.distance_matrix = np.zeros((npoints,npoints),dtype=np.float32)
        for i in range(npoints):
            for j in range(npoints):
                if i!= j:
                    self.distance_matrix[i][j] = self.get_vertex_by_id(i).get_edge_towards_vertex(j).d
        #if VERBOSE:
           # print('Done!')
            

        
        
    def neighbors(self,v):  # returns the list of neighbors of v -> for compatibility with NN
        retval = []
        n = self.vertices[v]
        edges = n.get_edges()
        for i in edges:
            retval.append(i.dest.id)
        return retval
    
    def number_of_nodes(self): # duplicate for code compatibility
        return self.nvertices
    
    def get_number_of_vertices(self):
        return self.nvertices
    
    def set_start_vertex(self,idx):
        self.start_vertex = idx
        self.vertices[idx].set_visited()
        
    def get_vertices(self):
        return self.vertices
    
    def get_total_reward(self):
        if self.total_reward < 0:
            self.total_reward = 0
            for id in self.vertices:
                self.total_reward += self.vertices[id].get_value()
        return self.total_reward
    
    def get_reward(self,id):
        return self.vertices[id].get_value()
    
    def get_vertex_by_id(self,idx):
        return self.vertices[idx]
    
    def get_n_vertices(self):
        return self.nvertices
    
    def get_start_vertex(self):
        return self.start_vertex  # returns the id of the vertex
    
    def set_end_vertex(self,idx):
        self.end_vertex = idx
    
    def get_end_vertex(self):
        return self.end_vertex  # returns the id of the vertex
    
    def get_budget(self):
        return self.budget
    
    def set_budget(self,value):
        self.budget = value
        
    def number_of_unvisited_vertices(self):
        retval = 0
        for i in self.vertices:
            if not self.vertices[i].get_visited():
                retval += 1
        return retval
    
    def list_of_visited_vertices(self):
        retlst = []
        for i in self.vertices:
            if self.vertices[i].get_visited()==True:
                retlst.append([self.vertices[i].x,self.vertices[i].y])
        return retlst
    
    def clear_visits(self):
        for i in self.vertices:
            self.vertices[i].clear_visited()
        self.vertices[self.start_vertex].set_visited()
    
    # # this should be better implemented
    # def shortest_path(self,start,goal):
    #     parent  = [-1]*self.nvertices
    #     g = [float('inf')]* self.nvertices
    #     g[start] = 0
    #     open = [start]
    #     while len(open)>0:
    #         min = g[open[0]]
    #         p = 0
    #         for i in open:
    #             if g[open[i]] < min:
    #                 p = i
    #                 min = g[open[i]]
    #         v = self.vertices[open[p]]
    #         open.pop(p)
    #         for e in v.edges:
    #             if g[e.]
                    
if __name__ == "__main__":
    og = OrienteeringGraph('graph_test.mat')