import os,sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import torch
import random
from torch import tensor
from torch.optim import Adam
from deep_qlearning_model import MyModel
from tqdm import trange

device = torch.device("cuda")
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
def waiting_time():
    wait_time = 0
    waiting_queue = ["E2TL", "N2TL", "W2TL", "S2TL"]
    cars = traci.vehicle.getIDList()
    for car in cars:
        road_id = traci.vehicle.getRoadID(car)
        if road_id in waiting_queue:
            wait_time += traci.vehicle.getAccumulatedWaitingTime(car)
    return wait_time

def do_action(old_action:int,new_action:int): # action in {0,1,2,3}
    if old_action == new_action:
        traci.trafficlight.setPhase("TL", 2*new_action)
    else:
        traci.trafficlight.setPhase("TL", 2*old_action+1)

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

def get_loss(scores, rewards, qvalues):
    f = torch.nn.MSELoss()
    target = rewards+0.8*torch.max(qvalues,\
        dim=1,keepdim=True).values.to(device)
    return f(scores,target.reshape(-1))
#----------------------------------------------------------------
if __name__ == "__main__":
    model = MyModel().to(device)
    model.train()
    # state_dict=torch.load('checkpoints/deep_qlearning_model_new_159.pth')
    # model.load_state_dict(state_dict)
    optimizer = Adam(model.parameters(),\
        lr=1e-3, weight_decay=5e-4)
    for epoch in range(160):
        with torch.no_grad():
            Actions = [tensor([i],device=device)\
                .type(torch.int) for i in range(4)]
            oldAction = Actions[0]
            oldwait = tensor([0],device=device).type(torch.float).reshape(1,1)
            traci.start(["sumo","-c","intersection/sumo_config.sumocfg",\
                                            "-W","--no-step-log","true"])
            features = tensor(list(get_input().values()))\
                .to(device).type(torch.float).reshape(1,16)
            actions = oldAction
            rewards = tensor([0],device=device).type(torch.float).reshape(1,1)
            qvalues = torch.zeros(size=(1,4)).to(device)
            # while traci.simulation.getMinExpectedNumber() > 0:
            for _ in trange(5400): #5000
                qvalue = model(features[-1],oldAction).reshape(1,4)
                # epsilon-greedy
                if random.random() < 0.3:
                    qrandom = torch.randint(0,4,(1,4)).to(device)
                    newAction = torch.argmax(qrandom).to(device).reshape(1,)
                else:
                    newAction = torch.argmax(qvalue).to(device).reshape(1,)
                do_action(oldAction, newAction)
                traci.simulationStep()
                feature = tensor(list(get_input().values()))\
                    .to(device).type(torch.float).reshape(1,16)
                features = torch.cat((features,feature),dim=0)
                actions = torch.cat((actions,newAction),dim=0)
                qvalues = torch.cat((qvalues,qvalue),dim=0)
                oldAction = newAction
                newait = tensor([waiting_time()],device=device).type(torch.float).reshape(1,1)
                reward = oldwait-newait
                rewards = torch.cat((rewards,reward),dim=0)
            traci.close()
            qvalues = qvalues[1:]
            rewards = rewards[1:]
            features = features[:-1]
            actions = actions[:-1]
        print("training for epoch",epoch)
        batch_size = 256;total_loss = 0
        for _ in trange(800): #800
            total_loss = 0; ind = 0
            for j in range(0,5400-batch_size,batch_size):
                ind = j+batch_size
                optimizer.zero_grad()
                scores = model(features[j:ind], actions[j:ind])
                loss = get_loss(scores[torch.arange(batch_size),\
                    actions[j:ind]],rewards[j:ind], qvalues[j:ind])
                loss.backward()
                optimizer.step()
                with torch.no_grad(): total_loss += loss
            optimizer.zero_grad()
            scores = model(features[ind:], actions[ind:])
            loss = get_loss(scores[torch.arange(scores.size(0)),\
                    actions[ind:]], rewards[ind:], qvalues[ind:])
            loss.backward()
            optimizer.step()
            with torch.no_grad(): total_loss += loss
        print("For",epoch,"epoch, the loss is",total_loss)
        if epoch%10==9:
            torch.save(model.state_dict(), 'checkpoints/deep_qlearning_model_new_'+str(epoch)+'.pth')
