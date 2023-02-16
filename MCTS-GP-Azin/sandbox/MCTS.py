#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:27:57 2021

@author: shamano
    """


import sys
import cProfile
sys.path.append('/home/azin/PycharmProjects/multiMCTS/OrienteeringGraphAzin')
import pstats
import graph
import math
import numpy as np
#import pickle
from progress.bar import Bar
import configparser
import argparse
import time
# import os.path
import os
import shutil
import copy
import testgp as op
#from collections import defaultdict
FAILURE_RETURN = 0


BUFFER_DISTANCES = {}
SUM_BUFFER_DISTANCE = {}
MEAN_DIST = {}
KNEAR = 15
SAMPLES=1

COMUNICATION_RANGE_DIST=100

class MCTS_node:
    def __init__(self,n,p=None):
        self.node = n
        self.parent = p
        self.children = []
        self.childrenmap = {}
        if p is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1
        self.Q = {}
        self.N = {}
        self.F = {}
      #   self.B = {}
      #  self.T = {}

    def is_leaf(self):
        return len(self.children)==0
    
    def has_child(self,n): # checks if the node has a child with a given id
        if self.N.get(n):
            return True
        else:
            return False

    def set_parent(self,p):
        self.parent = p

    def add_child(self,n):
        if n.node.id == 0:
            print('Trying to add child node with 0 id.')
        self.children.append(n)
        if n.node.id in self.childrenmap:
            print("Adding duplicate child node {} to {}".format(n.node.id,self.node.id))
        self.childrenmap[n.node.id] = n # useful to retrieve a child node by id
        n.set_parent(self)
        self.Q[n.node.id] = 0
        self.N[n.node.id] = 0
        self.F[n.node.id] = 0
        
    def get_child(self,id):
        return self.childrenmap.get(id)
                
    def get_ancestors(self):
        retval = []
        ct = self
        while ct is not None:
            retval.append(ct.node.id)
            ct = ct.parent
        if not ( 0 in retval ):
            pass
        return retval



def get_key_with_max_val(d):   # CANDIDATE FOR BETTER IMPLEMENTATION --> should be faster now

    a = np.argmax(list(d.values()))
    keys = list(d.keys())
    #print(keys[a])
    return keys[a]




def MCTS_prefill_buffer(g,robot):
    end_id =  g.get_end_vertex()
    for i in range(g.get_n_vertices()):
        for j in range(g.get_n_vertices()):
            if ( i!= j) and (i!= end_id):
                v = g.get_vertex_by_id(i)
                dist1 = v.get_edge_towards_vertex(j).sample_cost_parallel(SAMPLES)
                BUFFER_DISTANCES[(v.id,j)]= dist1
                
    for i in range(g.get_n_vertices()):
        if (i!= end_id):
            dist2 = g.vertices[i].get_edge_towards_vertex(end_id).sample_cost_parallel(SAMPLES)
            BUFFER_DISTANCES[(i,end_id)]= dist2

    for i in range(g.get_n_vertices()):
        for j in range(g.get_n_vertices()):
            if (i!=j) and (i!= end_id) and (j!=end_id):
                SUM_BUFFER_DISTANCE[(i,j,j,end_id)] = BUFFER_DISTANCES[(i,j)] + BUFFER_DISTANCES[(j,end_id)] 
                MEAN_DIST[(i,j,j,end_id)] = np.mean(SUM_BUFFER_DISTANCE[(i,j,j,end_id)])

    for i in g.vertices.keys():
        if i!= end_id:
            neighbors = []
            for j in g.vertices.keys():
                if (i != j):
                    neighbors.append((j,g.vertices[j].value/g.vertices[i].get_edge_towards_vertex(j).d))

            neighbors.sort(key=lambda y: y[1],reverse=True)
            subset = [y[0] for y in neighbors]
            children = list(subset[0:KNEAR])
            if len(children) < len(subset):
                subset = subset[KNEAR:2*KNEAR]
                for _ in range(min(4,len(subset))):
                    toadd = np.random.choice(subset)
                    children.append(toadd)
                    subset.remove(toadd)
            # subset = [y[0] for y in neighbors]
            #children = list(subset)
          #  subset = children[0:2*KNEAR]
            if end_id not in children:
                children.append(end_id)
            robot.CHILDREN_MAP[i] = list(children[:])
    #print(robot.CHILDREN_MAP)


def MCTS_sample_traverse_cost_ids(vs,g):

    values = 0
    for source,dest in  vs:
        edge = g.vertices[source].get_edge_towards_vertex(dest)
        values += edge.sample_cost()
    return values



    
def MCTS_pick_action_max(node,og, robot_1):  # only called on a node that is not a leaf
    if node.is_leaf():
        print("node id {}".format(node.node.id))
        print("children map {}".format(robot_1.CHILDREN_MAP[node.node.id]))
        raise Exception("Can't do action selection on a childless node")
    totalN = sum([i for i in node.N.values()])
    # totalReward= og.get_total_reward()
    uct_values = {}
    for i in node.children:
        uct_values[i.node.id] = node.Q[i.node.id]*(1-node.F[i.node.id]) + 3* math.sqrt(math.log(totalN)/node.N[i.node.id])
    return get_key_with_max_val(uct_values)

    
    
def MCTS_pick_root_action(root,robot, og, g2, robot_n):  # picks the best action on the root among those who satisfy the failure constraints
    candidates = {}
    candidates_n={}
    traveresd_n = robot_n.traversed.copy()
    #root_n = MCTS_node(g2.get_vertices()[robot_n.current_vertex])
    if robot_n.root_copy is not None:
        for j in robot_n.root_copy.children:
            if traveresd_n:
                if j.node.id not in traveresd_n:
                    candidates_n[j.node.id]= robot_n.cumulative_reward_list[j.node.id]/ g2.vertices[j.node.id].get_edge_towards_vertex(robot_n.next_location).sample_cost()   #[robot_n.root_copy.Q[j.node.id]]
    if candidates_n:
        action_n=max(candidates_n, key=candidates_n.get)   #get_key_with_max_val(candidates_n)
    else:
        action_n=None
    #print("estimated next location of neighbour: {}".format(action_n))
    #print("root children {}".format(root.children))
    #print("robot n traversed {}".format(robot_n.traversed))

    if 29 in robot_n.traversed:
        traveresd_n=traveresd_n.remove(29)
    if action_n != None and action_n != 29 and traveresd_n != None:
        traveresd_n.append(action_n)

    #print(root)
    children = set(robot.CHILDREN_MAP[robot.current_vertex]) - set(robot.traversed)
    #if root.node.id != robot_n.current_vertex:
        #if og.vertices[root.node.id].get_edge_towards_vertex(robot_n.current_vertex).sample_cost() < COMUNICATION_RANGE_DIST:

    for i in root.children:
        #if traveresd_n:
            #if i.node.id not in traveresd_n:
                #candidates[i.node.id] = root.Q[i.node.id]
        #else:
        candidates[i.node.id] = root.Q[i.node.id]




    #for i in children:
       # if traveresd_n:
        #    if i not in traveresd_n:
        #        candidates[i] = root.Q[i]
        #else:
       #     candidates[i] = root.Q[i]


    if candidates:
        return get_key_with_max_val(candidates)
    else:
        return None
    #totalN = sum([i for i in root.N.values()])
    #uct_values = {}
    #for i in root.children:

       # uct_values[i.node.id] = root.Q[i.node.id]* (1 - root.F[i.node.id]) + 3 * math.sqrt(
       #     math.log(totalN) / root.N[i.node.id])
    #if uct_values:
      #  return get_key_with_max_val(uct_values)
    #else:
      #  return None



def id_sequence_to_node_sequence(vs,root):
    cp = root
    sequence = [cp]
    for i in vs[1:]:
        sequence.append(cp.get_child(i))
        cp = cp.get_child(i)
    return sequence
    


# # Implements tree policy. It will either:
# # pick a child node that has not been tried yet (if it exists)
# # descend into a child using a constrained UCT and apply itself recursively from there
def MCTS_Traverse(current_vertex,g, robot_1):

    bakcup_init_vertex = current_vertex

    visited = {}
    for v in g.vertices.values():
        visited[v.id] = v.visited
    visited[current_vertex.node.id] = True # can't go back to same vertex I'm coming from

    if visited[g.get_end_vertex()]:
        print("We have a problem in MCTS_traverse_random_explore")
        _ = input("Press Enter to continue.")

    traversed = []
    traversed.append(current_vertex.node.id)

    while True:
        # FIRST: search for an unvisited child, if it exists
        candidates = list(robot_1.CHILDREN_MAP[current_vertex.node.id])
        for c in candidates[:]:
            if visited[c] or current_vertex.has_child(c):
                    candidates.remove(c)

        if len(candidates) > 0:   # Unvisited child exists; then pick one  and return
            down_vertex = np.random.choice(candidates)
            traversed.append(down_vertex)
            return traversed
        else:  # all children have been visited; pick one using constrained UCT and recurse down
            bestAction = MCTS_pick_action_max(current_vertex,g,robot_1)
            traversed.append(bestAction)
            # return traversed ######## TO BE REMOVED
            if bestAction == g.get_end_vertex():
                return traversed
            else:
                current_vertex = current_vertex.get_child(bestAction)
                visited[current_vertex.node.id] = True  # this vertex can't be chosen anymore


def MCTS_greedy_rollout(leaf,g,budget,visited, cumulative_reward_list, robot, robot_n):
    end_vertex_idx = g.get_end_vertex()


    current = leaf.node.id
    done = False

    traversed = []
    reward = 0
    
    all = [i for i in range(end_vertex_idx+1)]
    
    unusable = dict(visited)

    while not done:

        children = set(robot.CHILDREN_MAP[current]) - set(unusable.keys())
        #print(current)
        #print(robot_n.current_vertex)
        #if current != robot_n.current_vertex:
         #   if  g.vertices[current].get_edge_towards_vertex(robot_n.current_vertex).sample_cost()  < COMUNICATION_RANGE_DIST:
          #      children = children -set(robot_n.traversed)
        children = list(children)
        if not children:
            children.append(29)
        

        look_for_random = True
        while look_for_random:
            new = np.random.choice(children)
            if new not in unusable:
                look_for_random = False
        """
        if len(traversed) < 1:
            new = np.random.choice(children)
        else:  # len(traversed)<3:
            new = MCTS_get_new_vertex_greedy(g, current, children, budget, unusable)
        """
        reward += cumulative_reward_list[new] / g.vertices[current].get_edge_towards_vertex(new).sample_cost() #g.vertices[new].value

        traversed.append(new)
        time_to_go = g.vertices[current].get_edge_towards_vertex(new).sample_cost()
        if time_to_go > budget:
            return traversed, True,reward# failure
        if new == end_vertex_idx:
            done = True
        else:
            unusable[new] = True
            budget -= time_to_go
        current = new
    return traversed, False,reward





def MCTS_Backup(toadd,reward,failure_rate):

    parent = toadd.parent
    nodebackup = toadd
    

    if parent.N[toadd.node.id] == 0: # adding a new child
        parent.Q[toadd.node.id] = reward
        parent.F[toadd.node.id] = failure_rate
    else:  # update statistic
        parent.Q[toadd.node.id] = (parent.Q[toadd.node.id]*parent.N[toadd.node.id] + reward) / (parent.N[toadd.node.id]+1)
        parent.F[toadd.node.id] = (parent.F[toadd.node.id]*parent.N[toadd.node.id] + failure_rate) / (parent.N[toadd.node.id]+1)
 

    # now propagate upwards
    while parent.parent:
        if (parent.parent.F[parent.node.id] >= failure_rate) and \
            (parent.parent.Q[parent.node.id] <= reward + parent.node.value):   # easy case for update
            parent.parent.F[parent.node.id] = failure_rate
            parent.parent.Q[parent.node.id] = reward + parent.node.value
        elif (parent.parent.F[parent.node.id] <= failure_rate) and \
              (parent.parent.Q[parent.node.id] > reward + parent.node.value):
            break  # easy case for doing nothing
        elif (parent.parent.F[parent.node.id] < failure_rate) and \
              (parent.parent.Q[parent.node.id] < reward + parent.node.value):
            if failure_rate < FAILURE_PROB:     
                parent.parent.F[parent.node.id] = failure_rate
                parent.parent.Q[parent.node.id] = reward + parent.node.value
            else:
                break
        elif (parent.parent.F[parent.node.id] > failure_rate) and \
                (parent.parent.Q[parent.node.id] > reward + parent.node.value):
            if parent.parent.F[parent.node.id] > FAILURE_PROB:
                parent.parent.F[parent.node.id] = failure_rate
                parent.parent.Q[parent.node.id] = reward + parent.node.value
            else:
                break
        else:
            break
        reward += parent.node.value
        parent = parent.parent

    toadd = nodebackup
    parent = toadd.parent
    while parent:  # propagate count upwards
        parent.N[toadd.node.id] += 1
        parent = parent.parent
        toadd=toadd.parent
        
        
    return





def check_early_exit(root):
    s = [i for i in root.N.values()]
    if len(s) == 1:
        return True
    s.sort(reverse=True)
    if s[0]>2*s[1]:
        return True
    else:
        return False


def MCTS_search(g, g2,  robot, robot_n):

    root = MCTS_node(g.get_vertices()[robot.current_vertex])
    robot.root_copy=root
    end_node_id = g.get_end_vertex()

    for itc_counter in range(robot.it):
        #print("itc_counter {}".format(itc_counter))
        current_vertex = root
        vs = MCTS_Traverse(current_vertex,g, robot)

        sequence = id_sequence_to_node_sequence(vs,root)
        parent = sequence[-2]

        if not parent.has_child(vs[-1]):
            toadd = MCTS_node(g.get_vertices()[vs[-1]],parent)
            parent.add_child(toadd)
        else:
            toadd = parent.get_child(vs[-1])

        sequence = id_sequence_to_node_sequence(vs,root)
        leaf = sequence[-1]
        rewards_list = []
        fail_list = []

        unusable = {}
        bv = leaf
        # remove ancestors
        while bv is not None:
            unusable[bv.node.id]=True
            bv = bv.parent
        # remove visited vertices
        for j in list(g.vertices.values()):
            if (j.get_visited()) and (j.id not in unusable):
                unusable[j.id] = True


        if vs[-1] != end_node_id:
            couples = list(zip(vs, vs[1:]))

            for _ in range(int(SAMPLES)):

                cost_to_leaf = MCTS_sample_traverse_cost_ids(couples,g)

                traversed,fail,reward = MCTS_greedy_rollout(leaf,g,robot.current_budget-cost_to_leaf,unusable, robot.cumulative_reward_list, robot, robot_n)
                fail_list.append(fail)
                if not fail:

                    reward += robot.cumulative_reward_list[vs[-1]] / g.vertices[leaf.node.id].get_edge_towards_vertex(vs[-2]).sample_cost() #g.vertices[vs[-1]].value
                    rewards_list.append(reward)

            if not rewards_list:
                rewards_list.append(0)
        else: #leaf is final location
            rewards_list = [g.get_vertex_by_id(end_node_id).get_value()]
            couples = list(zip(vs, vs[1:]))
            for _ in range(int(SAMPLES)):
                a = MCTS_sample_traverse_cost_ids(couples,g)
                fail_list.append(a>robot.budget)

        robot.reward = np.mean(rewards_list)

        robot.failure_rate = sum(fail_list)/len(fail_list)

        MCTS_Backup(toadd,robot.reward,robot.failure_rate)
        
        if (itc_counter+1) % 10 == 0:
            if check_early_exit(root):
                # print("Exiting early: ",itc_counter)
                break

    bestAction = MCTS_pick_root_action(root,robot, g, g2, robot_n) #MCTS_Traverse(root,g) #MCTS_pick_root_action(root,g)
    


    #print(bestAction)
    if bestAction is not None:
        #print(root.node.get_edge_towards_vertex(bestAction).dest.id)
        if root.node.get_edge_towards_vertex(bestAction).dest.id != end_node_id:
            if robot.current_budget - (
                    g.vertices[robot.current_vertex].get_edge_towards_vertex(root.node.get_edge_towards_vertex(bestAction).dest.id).sample_cost() +
                    g.vertices[root.node.get_edge_towards_vertex(bestAction).dest.id].get_edge_towards_vertex(end_node_id).sample_cost()) < 0:
                #print("here")

                return root, root.node.get_edge_towards_vertex(end_node_id)
            else:

                robot.next_location = root.node.get_edge_towards_vertex(bestAction).dest.id
        #robot.root_copy = root.node.get_edge_towards_vertex(bestAction)
                return root, root.node.get_edge_towards_vertex(bestAction)
        else:
            return root, root.node.get_edge_towards_vertex(end_node_id)
    else:
        #robot.root_copy = root.node.get_edge_towards_vertex(end_node_id)
        return root, root.node.get_edge_towards_vertex(end_node_id)


def MCTS_simulate(og1,og2, robot_1, robot_2):


    robot_1.current_budget = og1.get_budget()
    robot_1.current_vertex = og1.get_start_vertex()
    robot_1.goal_vertex = og1.get_end_vertex()
    og1.vertices[robot_1.current_vertex].set_visited()
    #robot_1.cumulative_reward_list = op.get_gp([robot_1.current_vertex])
    #cumulative_reward1 = robot_1.cumulative_reward_list[0]
    #print(cumulative_reward)
    robot_1.traversed = [robot_1.current_vertex]


    robot_2.current_budget = og2.get_budget()
    robot_2.current_vertex = og2.get_start_vertex()
    robot_2.goal_vertex = og2.get_end_vertex()
    og2.vertices[robot_2.current_vertex].set_visited()
    #robot_2.cumulative_reward_list = op.get_gp([robot_2.current_vertex])
    #cumulative_reward2 = robot_2.cumulative_reward_list[0]
    #print(cumulative_reward)
    robot_2.traversed = [robot_2.current_vertex]

    done = False
    done1 = False
    done2 = False
    while not done:
        if not done1:

            # THIS WAS USED FOR ICRA
            #tree,action = MCTS_search_with_constraints_multiple_samples(og,current_vertex,current_budget,rollout_policy,k,max_iterations,expansion_probability=0.1)
            start1 = time.time()
            tree1,action1 = MCTS_search(og1, og2, robot_1, robot_2)
            #print(tree1)
            end1 = time.time()
            robot_1.times += (end1 - start1)
            robot_1.traversed.append(action1.dest.id)
            print("Robot 1 visited locations {}".format(robot_1.traversed))

            if action1 is None:
                cumulative_reward1=0
                robot_1.current_budget=-1
                tree1=None  # failure
            else:




                robot_1.current_budget = robot_1.current_budget - og1.vertices[robot_1.current_vertex].get_edge_towards_vertex(action1.dest.id).sample_cost()  # action1.sample_cost()
                print("Robot 1 current budget {}".format(robot_1.current_budget))
                robot_1.current_vertex = action1.dest.id

                list_visited_loc = og1.list_of_visited_vertices()
                og1.vertices[robot_1.current_vertex].set_visited()

                #cumulative_reward1 = op.get_gp(robot_1.traversed)[robot_1.current_vertex]
                cumulative_reward1 = robot_1.cumulative_reward_list[robot_1.current_vertex] #og.vertices[current_vertex].get_value()  # list_visite_loc
                #print(robot_1.current_vertex)
                #print(robot_1.goal_vertex)
                if (robot_1.current_budget < 0) or (robot_1.current_vertex == robot_1.goal_vertex):
                    done1 = True

        if not done2:
            start2 = time.time()
            tree2,action2 = MCTS_search(og2, og1, robot_2, robot_1)
            end2 = time.time()
            robot_2.times += (end2 - start2)
            robot_2.traversed.append(action2.dest.id)
            print("Robot 2 visited locations {}".format(robot_2.traversed))
            if action2 is None:
                cumulative_reward2=0
                robot_2.current_budget=-1
                tree2=None  # failure
            else:

                robot_2.current_budget = robot_2.current_budget - og2.vertices[robot_2.current_vertex].get_edge_towards_vertex(action2.dest.id).sample_cost()           #action2.sample_cost()
                print("Robot 2 current budget {}".format(robot_2.current_budget))
                robot_2.current_vertex = action2.dest.id

                list_visited_loc2 = og2.list_of_visited_vertices()
                og2.vertices[robot_2.current_vertex].set_visited()

                #cumulative_reward2 = op.get_gp(robot_2.traversed)[robot_2.current_vertex]
                cumulative_reward2 =  robot_2.cumulative_reward_list[robot_2.current_vertex] #og.vertices[current_vertex].get_value()  # list_visite_loc

                if (robot_2.current_budget < 0) or (robot_2.current_vertex == robot_2.goal_vertex):
                    done2 = True

        done = done1 and done2

    return cumulative_reward1,robot_1.current_budget,tree1,robot_1.traversed,cumulative_reward2,robot_2.current_budget,tree2,robot_2.traversed




def read_configuration(fname):
    config = configparser.ConfigParser()
    print("Reading configuration file ",fname)
    if os.path.exists(fname):
        config.read(fname)
    else:
        raise Exception("Can't read configuration file {}".format(fname))
    global NVERTICES,DEPTHLIMIT,NTRIALS, RETRIES, REPETITIONS,CONTRACTION,LIMITEXPLOREDEPTH
    global EPSMIN,EPSINC,EPSN,ITERMIN,ITERINC,ITERN,MINFREQ,VERBOSE,BIAS,BUDGET
    global SAMPLESMIN,SAMPLESINC,SAMPLESN,FAILURE_PROB,GREEDY_THRESHOLD, COMUNICATION_RANGE_DIST
    
    if config['MAIN']['NTRIALS'] is None:
        print('Missing configuration parameter ',NTRIALS)
    else:
          NTRIALS = int(config['MAIN']['NTRIALS'])
         
    if config['MAIN']['NVERTICES'] is None:
        print('Missing configuration parameter ',NVERTICES)
    else:
          NVERTICES = int(config['MAIN']['NVERTICES'])
         
    if config['MAIN']['DEPTHLIMIT'] is None:
        print('Missing configuration parameter ',DEPTHLIMIT)
    else:
          DEPTHLIMIT = int(config['MAIN']['DEPTHLIMIT'])   
         
    if config['MAIN']['RETRIES'] is None:
        print('Missing configuration parameter ',RETRIES)
    else:
          RETRIES = int(config['MAIN']['RETRIES'])     
    
    if config['MAIN']['REPETITIONS'] is None:
        print('Missing configuration parameter ',REPETITIONS)
    else:
          REPETITIONS = int(config['MAIN']['REPETITIONS']) 
         
    if config['MAIN']['EPSMIN'] is None:
        print('Missing configuration parameter ',EPSMIN)
    else:
          EPSMIN = float(config['MAIN']['EPSMIN']) 
         
    if config['MAIN']['EPSN'] is None:
        print('Missing configuration parameter ',EPSN)
    else:
          EPSN = int(config['MAIN']['EPSN']) 
        
    if config['MAIN']['EPSINC'] is None:
        print('Missing configuration parameter ',EPSINC)
    else:
          EPSINC = float(config['MAIN']['EPSINC']) 
         
    if config['MAIN']['ITERMIN'] is None:
        print('Missing configuration parameter ',ITERMIN)
    else:
          ITERMIN = int(config['MAIN']['ITERMIN']) 
         
    if config['MAIN']['ITERINC'] is None:
        print('Missing configuration parameter ',ITERINC)
    else:
          ITERINC = int(config['MAIN']['ITERINC']) 
        
    if config['MAIN']['ITERN'] is None:
        print('Missing configuration parameter ',ITERN)
    else:
          ITERN = int(config['MAIN']['ITERN']) 
         
    if config['MAIN']['MINFREQ'] is None:
        print('Missing configuration parameter ',MINFREQ)
    else:
          MINFREQ = int(config['MAIN']['MINFREQ']) 
         
    if config['MAIN']['SAMPLESMIN'] is None:
        print('Missing configuration parameter ',SAMPLESMIN)
    else:
          SAMPLESMIN = int(config['MAIN']['SAMPLESMIN'])   
         
    if config['MAIN']['SAMPLESINC'] is None:
        print('Missing configuration parameter ',SAMPLESINC)
    else:
          SAMPLESINC = int(config['MAIN']['SAMPLESINC'])   
         
    if config['MAIN']['SAMPLESN'] is None:
        print('Missing configuration parameter ',SAMPLESN)
    else:
          SAMPLESN = int(config['MAIN']['SAMPLESN'])   
         
    if config['MAIN']['BIAS'] is None:
        print('Missing configuration parameter ',BIAS)
    else:
          BIAS = float(config['MAIN']['BIAS']) 
    
    if config['MAIN']['CONTRACTION'] is None:
        print('Missing configuration parameter ',CONTRACTION)
    else:
          CONTRACTION = float(config['MAIN']['CONTRACTION'])         

    if config['MAIN']['VERBOSE'] is None:
        print('Missing configuration parameter ',VERBOSE)
    else:
          VERBOSE = (config['MAIN']['VERBOSE'] == "True")
         
    if config['MAIN']['LIMITEXPLOREDEPTH'] is None:
        print('Missing configuration parameter ',LIMITEXPLOREDEPTH)
    else:
          LIMITEXPLOREDEPTH = (config['MAIN']['LIMITEXPLOREDEPTH'] == "True") 
         
    if config['MAIN']['BUDGET'] is None:
        print('Missing configuration parameter ',BUDGET)
    else:
          BUDGET = float(config['MAIN']['BUDGET'])  
         
    if config['MAIN']['FAILURE_PROB'] is None:
        print('Missing configuration parameter ',FAILURE_PROB)
    else:
          FAILURE_PROB = float(config['MAIN']['FAILURE_PROB']) 
         
    if config['MAIN']['GREEDY_THRESHOLD'] is None:
        print('Missing configuration parameter ',GREEDY_THRESHOLD)
    else:
          GREEDY_THRESHOLD = float(config['MAIN']['GREEDY_THRESHOLD'])      
         
    print('Done reading configuration')
    
    
parser = argparse.ArgumentParser(description='Process parameters.')
parser.add_argument(
    '--logdir', 
    type = str, 
    default = 'sandbox',
    help = 'Directory where data will be saved')

parser.add_argument(
    '--conf', 
    type = str, 
    default = 'config.txt',
    help = 'Config file to use')

args = parser.parse_args()

def create_nested_path(path):
    # first break down all intermediate folders
    l = []
    done = False
    while not done:
        a,b = os.path.split(path)
        l.insert(0,b)
        if len(a)==0:
            done = True
        else:
            path = a
    partial_path = ''
    for i in l:
        partial_path = os.path.join(partial_path,i)
        if not os.path.isdir(partial_path):
            os.mkdir(partial_path)


class Robot:
    def __init__(self ):

        self.budget =500
        self.energy_T =0
        self.reward_T =0
        self.indicator =0
        self.rewards = []
        self.budgets = []
        self.times = 0
        self.totalRewards = 0
        self.failures = 0
        self.policyFailure = 0
        self.CHILDREN_MAP = {}
        self.current_budget=0
        self.current_vertex=0
        self.goal_vertex=0
        self.traversed=[]
        self.it=100 #No of rollout before it chooses the next locations
        self.root_copy=None
        self.cumulative_reward_list= [0]*100
        self.next_location=0
        self.len_traveresed=0



    def reset_children_map(self):
        self.CHILDREN_MAP = {}
    def reset(self ,params):
        # print("I'm here")
        self.current_state = 0
        self.visited_states_list =[self.start_state]
        self.visited_states_indexes_list =[0]
        self.unvisited_states_list =copy.deepcopy \
            (params.Univisited_locations)
        self.energy_T =0
        self.reward_T =0




def main(profiler):
    
    print('Starting...')
    
    read_configuration(args.conf)

    og1 = graph.OrienteeringGraph() #('graph_test_{}.mat'.format(NVERTICES)) #'2018-06-21_ripperdan.mat'
    og2 = graph.OrienteeringGraph() #'2018-06-21_ripperdan.mat'
    if not os.path.isdir(args.logdir):
        print("{} does not exist and will be created".format(args.logdir))
        # create all intermediate folders if needed
        create_nested_path(args.logdir)
    
    #backup configuration file and the version of the code used
    shutil.copyfile(args.conf,os.path.join(args.logdir,"config.txt"))
    shutil.copyfile("MCTS.py",os.path.join(args.logdir,"MCTS.py"))
    
    print('Processing graph with {} vertices'.format(NVERTICES))
    print('Budget is ',BUDGET)
    og1.set_budget(BUDGET)
    og2.set_budget(BUDGET)
   
    print('Starting simulation')
    ntrials = NTRIALS #No of trials

    count_f = 0
    count_l = 0
    count_r = 0
    count_t=0
    iterations_list=[100]
    for iterations in range(iterations_list[0]):
        #rewards = []
        #budgets = []
        #times = []
        #totalRewards = 0
        #failures = 0
        #policyFailure = 0
        robot_1=Robot()
        robot_2=Robot()

        print("SAMPLES=",SAMPLES) #No of samples to compute the cost and number of simulations
        print("Iterations=",robot_1.it) #No of rollout before it chooses the next locations



        bar = Bar('Processing', max=ntrials)
        residual = 0
        complete_time_series=[]
        complete_reward_series=[]
        complete_budget_series=[]
        best_residual_budget=[]
        best_failure_rate=[]
        best_policy_failure=[]
        best_time=[]
        best_bias=[]
        best_iterations=[]
        best_reward=[]
        visited_1=[]
        visited_2=[]
        com_visited=[]

        for _ in range(ntrials):
            #robot_1 = Robot()
            #robot_2 = Robot()
            robot_1.reset_children_map()#CHILDREN_MAP = {}
            robot_2.reset_children_map()
            MCTS_prefill_buffer(og1,robot_1)
            MCTS_prefill_buffer(og2, robot_2)
            profiler.enable()
            reward1,budget1,tree1,traversed1, reward2,budget2,tree2,traversed2 = MCTS_simulate(og1, og2,robot_1, robot_2)
            profiler.disable()
            #op.get_gp(traversed1+traversed2)
            #reward1, budget1, tree1, traversed1 = MCTS_simulate(og, robot_2, robot_1)

            robot_1.rewards.append(reward1)
            robot_1.budgets.append(budget1)


            complete_reward_series.append(reward1)
            complete_budget_series.append(budget1)
            visited_1.append(len(traversed1))
            visited_2.append(len(traversed2))


            print("\nReward1: ",reward1)
            print("Budget1: ",budget1)

            print("Path1: ",traversed1)
            print("len Path1: ", len(traversed1))
            if tree1 is None:
                robot_1.policyFailure += 1
            if budget1 < 0:
                robot_1.failures = robot_1.failures + 1
            else:
                robot_1.totalRewards += reward1
                robot_1.len_traveresed=len(robot_1.traversed)


            print("\nReward2: ",reward2)
            print("Budget2: ",budget2)
            print("Path2: ",traversed2)
            print("len Path2: ", len(traversed2))
            common_numb=[]

            for numb in robot_1.traversed:
                if numb in robot_2.traversed:
                    common_numb.append(numb)
            print("common visited locations: ", common_numb)
            com_visited.append(len(common_numb))
            if tree2 is None:
                robot_2.policyFailure += 1
            if budget2 < 0:
                robot_2.failures = robot_2.failures + 1
            else:
                robot_2.totalRewards += reward2
                robot_2.len_traveresed = len(robot_2.traversed)

            count_f+= (robot_1.failures+robot_2.failures)/2
            count_l+=(len(robot_1.traversed)+len(robot_2.traversed))/2
            if budget1 > 0 and budget2 > 0:
                count_r =+ len(common_numb)
                count_t += (robot_1.times + robot_2.times ) / 2
            # reset flags to start a new iteration
            for i in og1.vertices.keys():
                og1.vertices[i].clear_visited()
            og1.vertices[0].set_visited()
            for i in og2.vertices.keys():
                og2.vertices[i].clear_visited()
            og2.vertices[0].set_visited()
            bar.next()
        bar.finish()

        if ntrials-robot_1.failures > 0:
            av_rev = robot_1.totalRewards/(ntrials-robot_1.failures)

            #print("Average reward: ",av_rev)
            best_residual_budget.append(residual / (ntrials - robot_1.failures))
            best_reward.append(av_rev)
            best_failure_rate.append(robot_1.failures/ntrials)
            best_policy_failure.append(robot_1.policyFailure/ntrials)

            best_bias.append(BIAS)
            best_iterations.append(iterations)

            #if  (robot_1.failures/ntrials <= FAILURE_PROB):
            #    found_absolute_best = True
             #   absolute_best = av_rev
             #   best_iterations_val = iterations
                #best_epsilon_val = EPSILON
              #  absolute_best_time = av_time

    print("percentage of failure {}".format(count_f/iterations_list[0]))
    print("percentage of visiting locs {}".format(count_l / iterations_list[0]))
    print("percentage of revisited locations {}".format(count_r/iterations_list[0]))
    print("Avg planning time {}".format(count_t / iterations_list[0]))

            
    print("----------------")
    print("Comprehensive Results")
    print("Average reward:",best_reward)
    print("Average remaining budget:",best_residual_budget)

    #print("Failure rate (rans out of energy):",best_failure_rate)
    #print("Policy failure rate (No tree found):",best_policy_failure)
    print("average number of visited by robot 1:",np.mean(visited_1))
    print("average number of visited by robot 2:", np.mean(visited_2))
    print("average number of common visited locations:", np.mean(com_visited))
    
    print("----------------")

        
        
    with open(os.path.join(args.logdir,'results.txt'),"w") as f:
        f.write("Comprehensive Results\n")
        f.write("Vertices:{}\n".format(NVERTICES))
        f.write("Budget:{}\n".format(BUDGET))
        f.write("Iterations:{}\n".format(ITERMIN))
        f.write("Independent Runs:{}\n".format(NTRIALS))
        f.write("Average reward: {}\n".format(best_reward))
        f.write("Average residual budget: {}\n".format(best_residual_budget))
    return profiler


if __name__ == "__main__":
    profiler= cProfile.Profile()

    profiler=main(profiler)

    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
    #main()

