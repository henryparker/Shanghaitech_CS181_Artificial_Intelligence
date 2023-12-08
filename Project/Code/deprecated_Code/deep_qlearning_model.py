from torch import nn
from torch import tensor
import torch

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.emb = nn.Embedding(4, 32)
#         self.t_emb = nn.Embedding(2000, 16)
#         self.act_net = nn.Sequential(
#             nn.Linear(32, 300),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             nn.ReLU(),
#             nn.Linear(300, 128),
#         )
#         self.f_net = nn.Sequential(
#             nn.Linear(16, 300),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             nn.ReLU(),
#             nn.Linear(300, 128),
#         )
#         self.time_net = nn.Sequential(
#             nn.Linear(16, 300),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             nn.ReLU(),
#             nn.Linear(300, 128),
#         )
#         self.emb_net = nn.Sequential(
#             nn.Linear(256, 300),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             nn.ReLU(),
#             nn.Linear(300, 300),
#             nn.ReLU(),
#             nn.Linear(300, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )

#     def forward(self, features:tensor, action:tensor, time:tensor):
#         act_emb = self.emb(action)
#         # act_emb: (32, )
#         act_info = self.act_net(act_emb)
#         # act_info: (128, )
#         f_info = self.f_net(features)
#         # f_info: (128, )
#         t_emb = self.t_emb(time)
#         # t_emb: (16, )
#         t_info = self.time_net(t_emb)
#         # t_info: (128, )
#         emb_info = act_info*f_info
#         # emb_info: (128, )
#         t_emb_info = torch.cat((emb_info, t_info), dim=1)
#         # t_emb_info: (256, )
#         Qscore = self.emb_net(t_emb_info)
#         # Qscore: (1, )
#         return Qscore

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 4)
        )

    def forward(self, features:tensor, *args):
        Qscore = self.net(features)
        return Qscore