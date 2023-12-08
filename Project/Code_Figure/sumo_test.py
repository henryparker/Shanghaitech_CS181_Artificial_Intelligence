import os,sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# from naive_qlearning import naive_qlearning_get_action
from deep_qlearning import deep_qlearning_get_action

# figure setting
waitingtimelist = []
fig=plt.figure(figsize=(4, 4), dpi=300)
#----------------------------------------------------------------
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
seed_it(181)
#----------------------------------------------------------------
# phase codes based on environment.net.xml
# PHASE_NS_GREEN = 0
# PHASE_NS_YELLOW = 1
# PHASE_NSL_GREEN = 2
# PHASE_NSL_YELLOW = 3
# PHASE_EW_GREEN = 4
# PHASE_EW_YELLOW = 5
# PHASE_EWL_GREEN = 6
# PHASE_EWL_YELLOW = 7
#----------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--method", default="random", type = str)
parser.add_argument("-g","--gui", default=False, type = bool)
args = parser.parse_args()
#----------------------------------------------------------------
def waiting_time(wait_time:dict): # loss_1
    waiting_queue = ["E2TL", "N2TL", "W2TL", "S2TL"]
    cars = traci.vehicle.getIDList()
    for car in cars:
        road_id = traci.vehicle.getRoadID(car)
        if road_id in waiting_queue:
            wait_time[car] = traci.vehicle.getAccumulatedWaitingTime(car)
    return

def do_action(old_action:int,new_action:int): # action in {0,1,2,3}
    if old_action == new_action:
        traci.trafficlight.setPhase("TL", 2*new_action)
    else:
        traci.trafficlight.setPhase("TL", 2*old_action+1)

def random_get_action():
    return random.randint(0,3)

period_seq = [0]*80+[1]*40+[2]*80+[3]*40
period_count = -1
def period_get_action():
    global period_count,period_seq
    period_count+=1
    if period_count==len(period_seq):
        period_count = 0
    return period_seq[period_count]

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

def original_get_action(*args): return

def test_get_action(*args): return 0
#----------------------------------------------------------------
if args.method==None:
    raise("You didn't specify the method to use.")
elif args.method=="original":
    Get_Action = original_get_action
    do_action = original_get_action
elif args.method=="random":
    Get_Action = random_get_action
elif args.method=="period":
    Get_Action = period_get_action
elif args.method=="greedy":
    Get_Action = greedy_get_action
elif args.method=="test":
    Get_Action = test_get_action
# elif args.method=="naive_qlearning":
#     Get_Action = naive_qlearning_get_action
elif args.method=="deep_qlearning":
    Get_Action = deep_qlearning_get_action
#----------------------------------------------------------------
if __name__ == "__main__":
    gui_str = "sumo"
    if args.gui: gui_str = "sumo-gui"
    traci.start([gui_str,"-c","intersection/sumo_config.sumocfg","-W","--no-step-log","true"])
    
    loss = dict()
    oldAction = 0
    
    count = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        newAction = Get_Action()
        do_action(oldAction,newAction)
        waiting_time(loss)
        traci.simulationStep()
        oldAction = newAction
        count += 1
        if count%1000 == 0: print(count)
        waitingtimelist += [sum(loss.values())]

    # tmp_list1 = []
    # loss_1 = dict()
    # while traci.simulation.getMinExpectedNumber() > 0:
    #     newAction = deep_qlearning_get_action()
    #     do_action(oldAction,newAction)
    #     waiting_time(loss_1)
    #     traci.simulationStep()
    #     oldAction = newAction
    #     count += 1
    #     if count%1000 == 0: print(count)
    #     tmp_list1 += [np.var(list(loss_1.values()))]
    # waitingtimelist += [tmp_list1]
    # traci.close()

    # traci.start([gui_str,"-c","intersection/sumo_config.sumocfg","-W","--no-step-log","true"])
    # loss_2 = dict()
    # tmp_list2 = []
    # while traci.simulation.getMinExpectedNumber() > 0:
    #     newAction = greedy_get_action()
    #     do_action(oldAction,newAction)
    #     waiting_time(loss_2)
    #     traci.simulationStep()
    #     oldAction = newAction
    #     count += 1
    #     if count%1000 == 0: print(count)
    #     tmp_list2 += [np.var(list(loss_2.values()))]
    # waitingtimelist += [tmp_list2]
    # traci.close()

    # traci.start([gui_str,"-c","intersection/sumo_config.sumocfg","-W","--no-step-log","true"])
    # loss_3 = dict()
    # tmp_list3 = []
    # while traci.simulation.getMinExpectedNumber() > 0:
    #     newAction = period_get_action()
    #     do_action(oldAction,newAction)
    #     waiting_time(loss_3)
    #     traci.simulationStep()
    #     oldAction = newAction
    #     count += 1
    #     if count%1000 == 0: print(count)
    #     tmp_list3 += [np.var(list(loss_3.values()))]
    # waitingtimelist += [tmp_list3]
    # traci.close()
    # traci.close()
    # traci.start([gui_str,"-c","intersection/sumo_config.sumocfg","-W","--no-step-log","true"])
    # loss_4 = dict()
    # tmp_list4 = []
    # while traci.simulation.getMinExpectedNumber() > 0:
    #     newAction = random_get_action()
    #     do_action(oldAction,newAction)
    #     waiting_time(loss_4)
    #     traci.simulationStep()
    #     oldAction = newAction
    #     count += 1
    #     if count%1000 == 0: print(count)
    #     tmp_list4 += [np.var(list(loss_4.values()))]
    # waitingtimelist += [tmp_list4]
    # traci.close()
    

    # plot
    for i in waitingtimelist:
        plt.plot(i)
    plt.xlabel("iteration")
    plt.ylabel("variance")
    plt.savefig("waiting_time.png")
    plt.show()


    print("The total waiting time is",sum(loss.values()))
    print("The variance of waiting time is",np.var(list(loss.values())))

    traci.close()
    sys.exit()