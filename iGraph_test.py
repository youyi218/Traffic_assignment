# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:43:38 2021

@author: yyou
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:09:46 2017

@author: Pramesh Kumar
"""
import math
import time
import numpy as np
import os
import csv
from scipy.optimize import fsolve
from igraph import *
from multiprocessing import Pool
from collections import Counter
import psutil
import os

if os.name == "nt":
    inputLocation = os.path.dirname(os.path.realpath(__file__)) + "\\ATMO\\"
else:
    inputLocation = os.path.dirname(os.path.realpath(__file__)) + "/ATMO/"
#print(inputLocation)

indexA = 0
indexB = 0
indexSpeed = 0
indexDistance = 0
indexLinktype = 0
indexCapacity = 0
indexPeage = 0
HORIZON = 2017
TCAM1530 = 1.32
TCAM3050 = -0.86
#n_processor= psutil.cpu_count(logical = False)
# n_processor= 2
class Zone:
    def __init__(self, _tmpIn):
        self.zoneId = _tmpIn[0]
        self.lat = 0
        self.lon = 0
        self.destList = []

# class Demand:
def initDemand(_tmpIn):
    """
    return a dict of demand object
    """
    demand = dict()
    demand['fromZone'] = int(float(_tmpIn[0]))
    demand['toNode'] = int(float(_tmpIn[1]))
    demand['demand'] = float(_tmpIn[2])
    demand['sequence'] = []
    demand['sptt'] = 0.0
    return demand

def readDemand():
    inFile = open(inputLocation+ "demand.csv")
    for x in inFile:
        tmpIn = [t.strip() for t in x.strip().split(",")]
        A, B = int(float(tmpIn[0])), int(float(tmpIn[1]))
        # if A > 300:
        #     break
        tripSet[A, B] = initDemand(tmpIn)
        if A not in zoneSet:
            zoneSet[A] = Zone([A])
        if B not in zoneSet:
            zoneSet[B] = Zone([B])
        if B not in zoneSet[A].destList:
            zoneSet[A].destList.append(B)

    inFile.close()
    print(len(tripSet), "OD pairs")
    print(len(zoneSet), "zones")


def readNetwork(g):
    print("Lire les noeuds")
    with open(inputLocation + "nodes.csv" ) as inFile:
        fileReader = csv.reader(inFile, delimiter = ';')
        attributs = next(fileReader)
        indexId = attributs.index('N')
        listNodeId = []
        for tmpIn in fileReader:
            listNodeId.append(str(tmpIn[indexId]))
        g.add_vertices(listNodeId) 
        # g.vs['id'] = listNodeId
    # g.add_vertices(800000)
        
    print("Lire les arcs")
    with open(inputLocation + "network.csv" ) as inFile:
        fileReader = csv.reader(inFile, delimiter = ';')
        attributs = next(fileReader)
        indexA = attributs.index('A')
        indexB = attributs.index('B')
        indexSpeed = attributs.index('VIT_MOY_VL')
        indexDistance = attributs.index('DISTANCE')
        indexLinktype = attributs.index('LINKTYPE')
        indexCapacity = attributs.index('CAPACITE')
        indexPeage = attributs.index('PEAGE_KM')
        indexNature = attributs.index('NATURE')
        indexNbVoie = attributs.index('NB_VOIES')
        indexUrbain = attributs.index('URBAIN')
        indexClass = attributs.index('CL_ADMIN')
        # initialize all edges:
        list_arcs = []
        for tmpIn in fileReader:
            list_arcs.append((str(tmpIn[indexA]), str(tmpIn[indexB])))
        g.add_edges(list_arcs)
        inFile.close()
        
    with open(inputLocation + "network.csv" ) as inFile:
        fileReader = csv.reader(inFile, delimiter = ';')
        next(fileReader)
        counter = 0
        for tmpIn in fileReader:
            if counter % 50000 == 0:
                print(counter)
            counter += 1
            A, B = str(tmpIn[indexA]), str(tmpIn[indexB])
            edge_id = g.get_eid(A,B)
            linktype = int(float(tmpIn[indexLinktype]))
            distance = round(float(tmpIn[indexDistance]),2)
            speed = int(float(tmpIn[indexSpeed]))
            peage = float(tmpIn[indexPeage])
            # calculate cost
            if linktype == 100:
                cost_hp = 0.21 * max(distance - 0.3, 0) * math.pow(1 + TCAM1530/100, min(HORIZON, 2030)-2017) *\
                    math.pow(1 + TCAM3050/100, max(HORIZON, 2030) - 2030)
            else:
                cost_hp = 0.21 * distance * math.pow(1 + TCAM1530/100, min(HORIZON, 2030)-2017) *\
                    math.pow(1 + TCAM3050/100, max(HORIZON, 2030) - 2030)
            peage = peage * distance * math.pow(1 - 0.5/100, HORIZON - 2017)
            g.es[edge_id]['capacity'] = int(tmpIn[indexCapacity]) if int(tmpIn[indexCapacity]) != 0 else 1800
            g.es[edge_id]['linktype'] = linktype
            g.es[edge_id]['peage'] = peage
            g.es[edge_id]['distance'] = distance
            g.es[edge_id]['cost_p'] = round(cost_hp + peage, 4)
            g.es[edge_id]['cost_t'] = round(60 * distance / speed, 4)
            g.es[edge_id]['cost_total'] = g.es[edge_id]['cost_t'] + 4 * g.es[edge_id]['cost_p']
            g.es[edge_id]['nature'] = str(tmpIn[indexNature])
            g.es[edge_id]['nb_voies'] = str(tmpIn[indexNbVoie])
            g.es[edge_id]['urbain'] = str(tmpIn[indexUrbain])
            g.es[edge_id]['class'] = str(tmpIn[indexClass])
            g.es[edge_id]['flow'] = 0.0
            # g.vs.find(id=A)['connected'] = True
            # g.vs.find(id=B)['connected'] = True
            # try:
            #     if B not in g.vs.find(id=A)['outLinks']:
            #         g.vs.find(id=B)['outLinks'].append(B)
            # except (KeyError, TypeError):
            #     g.vs[A]['outLinks'] = [B]
            # try: 
            #     if A not in g.vs[B]['inLinks']:
            #         g.vs[B]['inLinks'].append(A)  
            # except (KeyError, TypeError):
            #     g.vs[B]['inLinks'] = [A]
        inFile.close()

###########################################################################################################################


#%%        
def updateTravelTime():
    '''
    This method updates the travel time ['cost_t'] on the links with the current flow
    '''
    gamma_v = 0.41
    alpha_v = 4
    chi_v = 2
    for edge in g.es:
        if edge['class'] == 'Autoroute':
            if edge['urbain'] == 'Non':
                gamma_v = 0.34
                if edge['nb_voies'] in [3,4]:
                    chi_v = 2.6
                    alpha_v = 6
                elif edge['nb_voies'] == 2:
                    chi_v = 2.7
                    alpha_v = 4
            elif edge['urbain'] == 'Oui':
                gamma_v = 0.41
                chi_v = 2
                if edge['nb_voies'] in [3,4]:
                    alpha_v = 6
                elif edge['nb_voies'] == 2:
                    alpha_v = 4
        elif edge['class'] == 'Nationale':
            if edge['urbain'] == 'Non':
                gamma_v = 0.41
                if edge['nature'] == 'Route a 2 chaussees':
                    chi_v = 2.6
                    alpha_v = 4
                if edge['nb_voies'] == 1:
                    chi_v = 1.6
                    alpha_v = 2.6
                elif edge['nature'] == 'Route a 1 chaussee':
                    chi_v = 2.2
                    alpha_v = 4
                    gamma_v = 0.41
            # urbain
            elif edge['urbain'] == 'Oui':
                chi_v = 1.6
                if edge['nature'] == 'Route a 1 chaussee' and edge['nb_voies'] == 1:
                    alpha_v = 3.4
                    gamma_v = 5.6
                else:
                    alpha_v = 3.5
                    gamma_v = 2.7
            
        elif edge['class'] == 'Departementale':
            if edge['urbain'] == 'Non':
                gamma_v = 0.41
                if edge['nature'] == 'Route a 2 chaussees':
                    if edge['nb_voies'] in [2,3]:
                        chi_v = 2.6
                        alpha_v = 4
                    elif edge['nb_voies'] == 1:
                        chi_v = 1.6
                        alpha_v = 2.6
                if edge['nature'] == 'Route a 1 chaussee':
                    if edge['nb_voies'] in [2,3]:
                        chi_v = 2.2
                        alpha_v = 4
                    elif edge['nb_voies'] == 1:
                        chi_v = 2.2
                        alpha_v = 4
            # urbain
            elif edge['urbain'] == 'Oui':
                chi_v = 1.6
                if edge['nature'] == 'Route a 1 chaussee' and edge['nb_voies'] == 1:
                    alpha_v = 3.4
                    gamma_v = 5.6
                else:
                    alpha_v = 3.5
                    gamma_v = 2.7
        else:
            if edge['nature'] == 'Route a 2 chaussees':
                e = 3
                chi_v = 2
                gamma_v = 0.41
                if edge['nb_voies'] in [1,2]:
                    alpha_v = 4
                elif edge['nb_voies'] in [3,4]:
                    alpha_v = 6
            elif edge['nature'] == 'Route a 1 chaussee':
                chi_v = 1.6
                if edge['nb_voies'] == 1:
                    alpha_v = 3.4
                    gamma_v = 5.6
                elif edge['nb_voies'] == 2:
                    alpha_v = 3.5
                    gamma_v = 2.7
        edge['cost_t'] = edge['cost_t'] * (1 + gamma_v * math.pow((edge['flow']/24*chi_v/edge['capacity']),alpha_v))
        edge['cost_total'] = edge['cost_t'] + 4 * edge['cost_p']


def findAlpha(x_bar):
    '''
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm
    '''
    print("calculate alpha for FW algo")
    def df(alpha):
        sum_derivative = 0 ## this line is the derivative of the objective function.
        for i in range(len(g.es)):
            sum_derivative = sum_derivative + (x_bar[i] - g.es[i]['flow'])*g.es[i]['cost_total']
        return sum_derivative
    sol2 = fsolve(df, np.array([0.1]))
    return max(0.1, min(1, sol2[0]))

def loadAON():
    '''
    This method produces auxiliary flows for all or nothing loading.
    
    Returns:
        SPTT: sum of the cost and demand of all links
        x_bar: a dictionary of flows of all links
    '''
    x_bar = {l: 0.0 for l in range(len(g.es))}
    SPTT = 0.0
    for r in originZones:
        timeDijkstra= time.time()
        paths = g.get_shortest_paths(str(r), [str(dest) for dest in zoneSet[r].destList], "cost", "out", "epath")
        for index, s in enumerate(zoneSet[r].destList):
            # tripSet[(r, s)]['sequence'] = paths[index]
            try:
                dem = tripSet[r, s]['demand']
            except KeyError:
                dem = 0.0
            try:
                SPTT = SPTT + int(sum(g.es[paths[index]]['cost_total']) * dem)
            except KeyError:
                pass
            if r != s:
                tripSet[(r, s)]['sptt'] = str(SPTT)
                for edge in paths[index]:
                    x_bar[edge] = x_bar[edge] + dem
        if int(r)%500 == 0:
            print("Origin: {}, Dijkstra time used: {} sedonds ".format(r, round(time.time() - timeDijkstra, 2)))
    return SPTT, x_bar

# def loadAON():
#     """
#     This function start multiprocesses for calculation of shortest paths
#     """
#     print('Start all or nothing loading')
#     SPTT_list, x_bar_list = [], []
#     originZones_list = [list(originZones)[i::n_processor] for i in range(n_processor)]
#     pool = Pool(n_processor)
#     process_results = []
#     for p in range(n_processor):
#         print("Start process: ",str(p))
#         process_results.append(pool.apply_async(loadAON_process, args=(originZones_list[p], g, zoneSet, tripSet, p)))
#     pool.close()
#     pool.join()
#     results = [result.get() for result in process_results]
#     SPTT_list, x_bar_list = zip(*results)
#     SPTT = sum(SPTT_list)
#     # print(x_bar_list)
#     x_bar = Counter()
#     for i in range(n_processor):
#         x_bar += Counter(x_bar_list[i])
#     # print(x_bar)
#     print("Loading finished")
#     return SPTT, x_bar

def assignment(algorithm, accuracy = 0.01, maxIter=100):
    '''
    * Performs traffic assignment
    * Algorithm to calculate alpha can be MSA or FW
    * Accuracy to be given for convergence
    * maxIter to stop if not converged
    '''
    it = 1 # iteration
    gap = float("inf")
    x_bar = {l: 0.0 for l in range(len(g.es))}
    startP = time.time()
    while gap > accuracy:
        startI  = time.time()
        print("iteration start :" + str(it))
        # Calculate the alpha
        if algorithm == "MSA" or it < 2:
            alpha = (1.0/it)
        elif algorithm == "FW":
            alpha = findAlpha(x_bar)
        else:
            print("Terminating the program.....")
            print("The solution algorithm ", algorithm, " does not exist!")
        
        timeCost = time.time()
        updateTravelTime()
        print("Update travel time used: ", round(time.time() - timeCost, 2), "secs")

        # All or nothing load
        SPTT, x_bar = loadAON()
        
        # Update flow for every link with old_flow and flow of current iteration
        timeFlow = time.time()
        for i in range(len(g.es)):
            g.es[i]['flow'] = alpha*x_bar[i] + (1-alpha)*g.es[i]['flow']
        print("Modify attributs flow time used: ", round(time.time() - timeFlow, 2), "secs")
        
        # total demand assigned
        TSTT = round(sum([a['flow'] * a['cost_total'] for a in g.es]), 3)
        # total demand in reality
        SPTT = round(SPTT, 3)
        gap = round(abs((TSTT / SPTT) - 1), 5)
        print("Iteeration {} takes {} seconds, gap is {} with alpha {}.".format(it, round(time.time() - startI, 2) gap, alpha))
        print(TSTT, SPTT, gap)
        it = it + 1
        if it > maxIter:
            print("The assignment did not converge with the desired gap and max iterations are reached")
            print("current gap ", gap)
            break
    print("Assignment took", round(time.time() - startP, 2), " seconds")
    print("assignment converged in ", it-1, " iterations")
    print("current gap ", gap)


###########################################################################################################################


#%%

def writeUEresults(sep=';'):
    outFile = open("UE_results2.csv", "w")                                                                                                                                                                                                                                                                # IVT, WT, WK, TR
    tmpOut = sep.join(["B","A","Capacite","Distance","Cout","Volume"])
    outFile.write(tmpOut+"\n")
    
    for edge in g.es:
        tmpOut = str(g.vs[edge.target]['name']) + sep + str(g.vs[edge.source]['name']) + sep + str(edge['capacity']) + sep + str(edge['distance']) + sep + str(edge['cost_total']) + sep + str(edge['flow'])
        outFile.write(tmpOut + "\n")
    
    outFile.close()

def writePathResults(sep=';'):
    outFile = open("OD_paths2.csv", "w")                                                                                                                                                                                                                                                                # IVT, WT, WK, TR
    tmpOut = sep.join(["Origine","Destination","sequence","sptt"])
    outFile.write(tmpOut+"\n")
    for trip_key in tripSet:
        tmpOut = str(trip_key[0]) + sep + str(trip_key[1]) + sep + str(tripSet[trip_key]['sequence']) + sep + str(tripSet[trip_key]['sptt'])
        outFile.write(tmpOut + "\n")
    outFile.close()
    
#%%
    
#%%
if __name__ == '__main__':
    
    #%%
    readStart = time.time()
    g = Graph(directed = True)
    
    tripSet = {}
    zoneSet = {}
    
    print("reading")
    readDemand()
    readNetwork(g)
    
    originZones = set([k['fromZone'] for k in tripSet.values()])
    print("Reading the network data took", round(time.time() - readStart, 2), "secs")
    
    print("start assignement: ")
    assignment("FW", accuracy = 0.01, maxIter=4)

#%%
    writeUEresults()
    #assignment("stochastic", "MSA", accuracy = 0.01, maxIter=100)"" 
    # writePathResults()

