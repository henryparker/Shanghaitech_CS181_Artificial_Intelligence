import os,sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
import pickle

with open('checkpoints/naive_qlearning_dict.pkl', 'rb') as f:
    weight = pickle.load(f)

def Qvalue(features:list,action:int)->list:
    global weight
    return np.sum(np.array(features)*weight[action])

def get_input()->dict:
    waiting_queue = {"E2TL_0":0,"E2TL_1":0,"E2TL_2":0,"E2TL_3":0,
                     "N2TL_0":0,"N2TL_1":0,"N2TL_2":0,"N2TL_3":0,
                     "W2TL_0":0,"W2TL_1":0,"W2TL_2":0,"W2TL_3":0,
                     "S2TL_0":0,"S2TL_1":0,"S2TL_2":0,"S2TL_3":0}
    cars = traci.vehicle.getIDList()
    for car in cars:
        road_id = traci.vehicle.getLaneID(car)
        if road_id in waiting_queue:
            waiting_queue[road_id] += 1
    return waiting_queue

def get_argmax_action(features:list):
    result = [Qvalue(features,action) for action in {0,1,2,3}]
    return np.argmax(result)

def naive_qlearning_get_action():
    features = list(get_input().values())
    return get_argmax_action(features)

