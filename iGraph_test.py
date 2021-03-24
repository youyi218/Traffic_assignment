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
from scipy import optimize
import os
import csv
from igraph import *

inputLocation = os.path.dirname(os.path.realpath(__file__)) + "\\ATMO\\"
print(inputLocation)

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
        if A > 2000:
            break
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
    # print("Lire les noeuds")
    # with open(inputLocation + "nodes.csv" ) as inFile:
    #     fileReader = csv.reader(inFile, delimiter = ';')
    #     attributs = next(fileReader)
    #     indexId = attributs.index('N')
    #     listNodeId = []
    #     for tmpIn in fileReader:
    #         listNodeId.append(int(tmpIn[indexId]))
    #     g.add_vertices(len(listNodeId)) 
    #     g.vs['id'] = listNodeId
    g.add_vertices(800000)
        
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
            A, B = int(tmpIn[indexA]), int(tmpIn[indexB])
            # list_arcs.append((g.vs.find(id=A),g.vs.find(id=B)))
            list_arcs.append((A,B))
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
            A, B = int(tmpIn[indexA]), int(tmpIn[indexB])
            # g.add_edges([(A, B)])
            edge_id = g.get_eid(A,B)
            # calculate cost
            linktype = int(float(tmpIn[indexLinktype]))
            distance = round(float(tmpIn[indexDistance]),2)
            speed = int(float(tmpIn[indexSpeed]))
            peage = float(tmpIn[indexPeage])
            ftt = 60 * distance / speed
            if linktype == 100:
                cost_hp = 0.21 * max(distance - 0.3, 0) * math.pow(1 + TCAM1530/100, min(HORIZON, 2030)-2017) *\
                    math.pow(1 + TCAM3050/100, max(HORIZON, 2030) - 2030)
            else:
                cost_hp = 0.21 * distance * math.pow(1 + TCAM1530/100, min(HORIZON, 2030)-2017) *\
                    math.pow(1 + TCAM3050/100, max(HORIZON, 2030) - 2030)
            peage = peage* distance * math.pow(1 - 0.5/100, HORIZON - 2017)
            cost = int(4 * (cost_hp + peage) + ftt)
            g.es[edge_id]['capacity'] = int(tmpIn[indexCapacity]) if int(tmpIn[indexCapacity]) != 0 else 1800
            g.es[edge_id]['linktype'] = linktype
            g.es[edge_id]['peage'] = peage
            g.es[edge_id]['distance'] = distance
            g.es[edge_id]['t'] = round(ftt,2)
            g.es[edge_id]['p'] = round(cost_hp + peage, 2)
            g.es[edge_id]['cost'] = cost
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
readStart = time.time()
g = Graph(directed = True)

tripSet = {}
zoneSet = {}

print("reading")
readDemand()
readNetwork(g)

originZones = set([k['fromZone'] for k in tripSet.values()])
print("Reading the network data took", round(time.time() - readStart, 2), "secs")

#%%
def DijkstraHeap(origin):
    '''
    Calcualtes shortest path from an origin to all other destinations.
    The labels and preds are stored in node instances.
    '''
    # time1 = time.time()
    # labels = g.shortest_paths([origin], zoneSet[origin].destList, "cost", "out")
    # print("Time 1: ", round(time.time() - time1, 2), "secs")
    # print(labels[0][:20])
    # time2 = time.time()
    paths = g.get_shortest_paths(origin, zoneSet[origin].destList, "cost", "out", "epath")
    # print(paths[:20])
    # print("Time 2: ", round(time.time() - time2, 2), "secs")
    
    # time3 = time.time()
    # somme = [int(sum(g.es[c]['cost'])) for c in paths]
    # print(somme[:20])
    # print("Time 3: ", round(time.time() - time3, 2), "secs")
    return paths
        
def updateTravelTime():
    '''
    This method updates the travel time ['t'] on the links with the current flow
    '''
    gamma_v = 0.41
    alpha_v = 4
    chi_v = 2
    e = 3
    for edge in g.es:
        if edge['class'] == 'Autoroute':
            if edge['urbain'] == 'Non':
                gamma_v = 0.34
                e = 2.5
                if edge['nb_voies'] in [3,4]:
                    chi_v = 2.6
                    alpha_v = 6
                elif edge['nb_voies'] == 2:
                    chi_v = 2.7
                    alpha_v = 4
            elif edge['urbain'] == 'Oui':
                gamma_v = 0.41
                chi_v = 2
                e = 3
                if edge['nb_voies'] in [3,4]:
                    alpha_v = 6
                elif edge['nb_voies'] == 2:
                    alpha_v = 4
        elif edge['class'] == 'Nationale':
            if edge['urbain'] == 'Non':
                gamma_v = 0.41
                e = 2.5
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
                e = 3
                if edge['nature'] == 'Route a 1 chaussee' and edge['nb_voies'] == 1:
                    alpha_v = 3.4
                    gamma_v = 5.6
                else:
                    alpha_v = 3.5
                    gamma_v = 2.7
            
        elif edge['class'] == 'Departementale':
            e = 3
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
        edge['t'] = edge['t'] * (1 + gamma_v * math.pow(((e+1)*edge['flow']/24*chi_v/edge['capacity']),alpha_v))
        edge['cost'] = edge['t'] + edge['p']

from scipy.optimize import fsolve
def findAlpha(x_bar):
    '''
    This uses unconstrained optimization to calculate the optimal step size required
    for Frank-Wolfe Algorithm

    ******************* Need to be revised: Currently not working.**********************************************
    '''
    #alpha = 0.0

    print("calculate alpha for FW algo")
    def df(alpha):
        sum_derivative = 0 ## this line is the derivative of the objective function.
        for edge in g.es:
            key = g.get_eid(edge.source, edge.target)
            sum_derivative = sum_derivative + (x_bar[key] - edge['flow'])*edge['cost']
        return sum_derivative
    # sol = optimize.root(df, np.array([0.1]))
    sol2 = fsolve(df, np.array([0.1]))
    #print(sol.x[0], sol2[0])
    return max(0.1, min(1, sol2[0]))
    '''
    def int(alpha):
        tmpSum = 0
        for l in linkSet:
            tmpFlow = (linkSet[l].flow + alpha*(x_bar[l] - linkSet[l].flow))
            tmpSum = tmpSum + linkSet[l].fft*(tmpFlow + linkSet[l].alpha * (math.pow(tmpFlow, 5) / math.pow(linkSet[l].capacity, 4)))
        return tmpSum

    bounds = ((0, 1),)
    init = np.array([0.7])
    sol = optimize.minimize(int, x0=init, method='SLSQP', bounds = bounds)

    print(sol.x, sol.success)
    if sol.success == True:
        return sol.x[0]#max(0, min(1, sol[0]))
    else:
        return 0.2
    '''

def loadAON():
    '''
    This method produces auxiliary flows for all or nothing loading.
    '''
    x_bar = {l: 0.0 for l in range(len(g.es))}
    SPTT = 0.0
    for r in originZones:
        print("start charge AON: " + str(r))
        timeDijkstra= time.time()
        paths = DijkstraHeap(r)
        for index, s in enumerate(zoneSet[r].destList):
            tripSet[(r, s)]['sequence'] = paths[index]
            try:
                dem = tripSet[r, s]['demand']
            except KeyError:
                dem = 0.0
            try:
                SPTT = SPTT + int(sum(g.es[paths[index]]['cost']) * dem)
            except KeyError:
                SPTT = 0.0
            if r != s:
                tripSet[(r, s)]['sptt'] = str(SPTT)
                for path in paths[index]:
                    x_bar[path] = x_bar[path] + dem
        print("Dijkstra time used: ", round(time.time() - timeDijkstra, 2), "secs")
    return SPTT, x_bar

def assignment(loading, algorithm, accuracy = 0.01, maxIter=100):
    '''
    * Performs traffic assignment
    * Type is either deterministic or stochastic (for now only deterministic)
    * Algorithm to calculate alpha can be MSA or FW
    * Accuracy to be given for convergence
    * maxIter to stop if not converged
    '''
    it = 1 # iteration
    gap = float("inf")
    x_bar = {l: 0.0 for l in range(len(g.es))}
    startP = time.time()
    while gap > accuracy:
        print("iteration start")
        if algorithm == "MSA" or it < 2:
            alpha = (1.0/it)
        elif algorithm == "FW":
            alpha = findAlpha(x_bar)
            #print("alpha", alpha)
        else:
            print("Terminating the program.....")
            print("The solution algorithm ", algorithm, " does not exist!")
        # prevLinkFlow = np.array([g.es['flow']])
        timeFlow = time.time()
        for i in range(len(g.es)):
            g.es[i]['flow'] = alpha*x_bar[i] + (1-alpha)*g.es[i]['flow']
        print("Modify attribut flow time used: ", round(time.time() - timeFlow, 2), "secs")
        timeCost = time.time()
        updateTravelTime()
        print("Update travel time used: ", round(time.time() - timeCost, 2), "secs")
        #printXbar(x_bar)
        if loading == "deterministic":
            SPTT, x_bar = loadAON()
            #print([linkSet[a].flow * linkSet[a].cost for a in linkSet])
            TSTT = round(sum([a['flow'] * a['cost'] for a in g.es]), 3)
            SPTT = round(SPTT, 3)
            gap = round(abs((TSTT / SPTT) - 1), 5)
            # print(TSTT, SPTT, gap)
            if it == 1:
                gap = gap + float("inf")
                
        else:
            print("Terminating the program.....")
            print("The loading ", loading, " is unknown")

        it = it + 1
        if it > maxIter:
            print("The assignment did not converge with the desired gap and max iterations are reached")
            print("current gap ", gap)
            break
    print("Assignment took", time.time() - startP, " seconds")
    print("assignment converged in ", it-1, " iterations")


###########################################################################################################################

#%%

print("start assignement: ")
assignment("deterministic", "MSA", accuracy = 0.1, maxIter=3)
#%%

def writeUEresults(sep=';'):
    outFile = open("UE_results2.csv", "w")                                                                                                                                                                                                                                                                # IVT, WT, WK, TR
    tmpOut = sep.join(["B","A","Capacite","Distance","Cout","Volume"])
    outFile.write(tmpOut+"\n")
    
    for edge in g.es:
        tmpOut = str(edge.target) + sep + str(edge.source) + sep + str(edge['capacity']) + sep + str(edge['distance']) + sep + str(edge['cost']) + sep + str(edge['flow'])
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
writeUEresults()
#assignment("stochastic", "MSA", accuracy = 0.01, maxIter=100)"" 
writePathResults()

