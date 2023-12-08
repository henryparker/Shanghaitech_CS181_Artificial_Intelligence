import os,sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
import torch
import random
from torch import tensor
from deep_qlearning_model import MyModel

device = torch.device("cuda")

model = MyModel().to(device)
# state_dict=torch.load('checkpoints/deep_qlearning_model_79.pth')
state_dict=torch.load('checkpoints/deep_qlearning_model_new_99.pth')
model.load_state_dict(state_dict)
model.eval()

def Qvalue(features:tensor,action:int)->float:
    global model
    features = features.to(device).type(torch.float)
    action = tensor([action]).type(torch.int).to(device)
    return model(features, action).detach().cpu().numpy()

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

def get_argmax_action(features:tensor)->int: # {0,1,2,3}
    result = [Qvalue(features,action) for action in {0,1,2,3}]
    return np.argmax(result)

def greedy_get_action(): # let the load with most cars be green
    waiting_queue = {"E2TL_0":0,"E2TL_1":0,"E2TL_2":0,"E2TL_3":0,
                     "N2TL_0":0,"N2TL_1":0,"N2TL_2":0,"N2TL_3":0,
                     "W2TL_0":0,"W2TL_1":0,"W2TL_2":0,"W2TL_3":0,
                     "S2TL_0":0,"S2TL_1":0,"S2TL_2":0,"S2TL_3":0}
    cars = traci.vehicle.getIDList()
    for car in cars:
        road_id = traci.vehicle.getLaneID(car)
        if road_id in waiting_queue:
            waiting_queue[road_id] += 1
    result = [0,0,0,0]
    result[0] = waiting_queue["N2TL_0"]+waiting_queue["N2TL_1"]+waiting_queue["N2TL_2"]+\
                waiting_queue["S2TL_0"]+waiting_queue["S2TL_1"]+waiting_queue["S2TL_2"]
    result[1] = waiting_queue["N2TL_3"]+waiting_queue["S2TL_3"]
    result[2] = waiting_queue["W2TL_0"]+waiting_queue["W2TL_1"]+waiting_queue["W2TL_2"]+\
                waiting_queue["E2TL_0"]+waiting_queue["E2TL_1"]+waiting_queue["E2TL_2"]
    result[3] = waiting_queue["W2TL_3"]+waiting_queue["E2TL_3"]
    return np.argmax(result)

def deep_qlearning_get_action()->int: # {0,1,2,3}
    if random.random() < 0.3: return greedy_get_action()
    features = tensor(list(get_input().values())).to(device)
    old_act = get_argmax_action(features)
    return old_act
