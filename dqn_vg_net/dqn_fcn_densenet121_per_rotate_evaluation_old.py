# --------------------------------------------------------
# Deep Q-Learning Algorithm for learning graspable points
# Input: only depth image
# Version: pytorch code
# !! With Prioritized Experience Replay (PER)
# !! Using FCN(DenseNet121) output single-channel q-value affordance map,
#    and input 10 rotated depth images at same time, then find the max q-value in all 10 output q-value map
# --------------------------------------------------------

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time
import shutil
from vrep_env import ArmEnv
import random
from configs import Configs
import my_utils
import shutil
from tensorboardX import SummaryWriter
from prioritized_memory import Memory
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#set hyper parameters
Train_Configs = Configs()
env = ArmEnv()
# env = gym.make('MountainCar-v0')
# env = env.unwrapped
# NUM_STATES = env.observation_space.shape[0] # 2
# DIM_ACTIONS = env.action_space.n
DIM_ACTIONS = Train_Configs.DIM_ACTIONS
DIM_STATES = Train_Configs.DIM_STATES
CHANNELS = Train_Configs.ANGLE_CHANNELS

#--------------------------------------
# Build network for q-value prediction
# Input: depth image, dim:[227,227] =>> change to [64,64]
# Output: dim[DIM_ACTIONS](64*64*10), every element represents a q-value,
#--------------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base_model = torchvision.models.densenet121(pretrained=True)
        self.get_feature = self.base_model.features  # shape:([batch_size, 1024, 7, 7]),input[batch_size,3,224,224]

        self.conv_feat = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, CHANNELS, kernel_size=1, stride=1, bias=False)
        )#out:[batch_size,CHANNELS,7,7]

        # Lateral convolutional layer
        self.lateral_layer = nn.Conv2d(3, CHANNELS, kernel_size=1, stride=1, padding=0)# 512
        # Bilinear Upsampling
        self.up = nn.Upsample(scale_factor=32, mode='bilinear')

    def forward(self, x):
        out1 = self.get_feature(x) # out:[batch_size,1024,7,7]
        out2 = self.conv_feat(out1) # out:[batch_size,CHANNELS,7,7]
        out_up =  self.up(out2) + self.lateral_layer(x)
        return out_up#dim:[batch_size,CHANNELS,224,224]

class Dqn():
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net.cuda()
        self.target_net.cuda()

        # create prioritized replay memory using SumTree
        self.memory = Memory(Train_Configs.MEMORY_CAPACITY)

        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), Train_Configs.LR)
        # self.loss = nn.MSELoss(size_average=True)
        self.loss = nn.MSELoss(reduce=False, size_average=False)

        self.fig, self.ax = plt.subplots()
        self.discount_factor = Train_Configs.GAMMA

    def store_trans(self, state_path, action, reward, next_state_path,done):
        ## action type: id
        x, y, c = my_utils.translate_actionID_to_XY_and_channel(action)
        trans = state_path+'#'+str(action)+'#'+str(reward)+'#'+next_state_path#np.hstack((state, [action], [reward], next_state))
        #------ calculate TD errors from (s,a,r,s'), only from the first depth image, without considering other 9 rotated depth images
        if c > 0:
            state_path = my_utils.get_rotate_depth(c,state_path)
        state = my_utils.copy_depth_to_3_channel(state_path).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        if c > 0:
            next_state_path = my_utils.get_rotate_depth(c,next_state_path)
        next_state = my_utils.copy_depth_to_3_channel(next_state_path).reshape(1, 3, DIM_STATES[0], DIM_STATES[1])
        # normlize
        state = (state - np.min(state)) / (np.max(state) - np.min(state))
        next_state = (next_state - np.min(next_state)) / (np.max(next_state) - np.min(next_state))
        # numpy to tensor
        state = torch.cuda.FloatTensor(state)
        next_state = torch.cuda.FloatTensor(next_state)

        target_singleChannel_q_map = self.eval_net.forward(state)#dim:[1,1,224,224],CHANNEL=1
        # x,y,c = my_utils.translate_actionID_to_XY_and_channel(action)
        old_val = target_singleChannel_q_map[0][0][x][y]
        # old_val = target[0][action]
        target_val_singleChannel_q_map = self.target_net.forward(next_state)#dim:[1,1,224,224]

        if done == 1:
            target_q = reward # target[0][action] = reward
        else:
            target_q = reward + self.discount_factor * torch.max(target_val_singleChannel_q_map) # target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target_q)
        # self.memory.add(error.cpu().detach().numpy(), trans)
        self.memory.add(float(error), trans)
        # self.memory[index] = trans + '#' + str(error)
        # self.memory_counter += 1
        '''--------------------- reference code for per --------------------
        target = self.model(Variable(torch.FloatTensor(state))).data
        old_val = target[0][action]
        target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))
        ---------------------------------------------------------------------'''

    def choose_action(self, state_path,EPSILON):
        # notation that the function return the action's index nor the real action
        # EPSILON
        state = []
        state.append(my_utils.copy_depth_to_3_channel(state_path))#dim:[3, DIM_STATES[0], DIM_STATES[1]]#.reshape(1, 3, DIM_STATES[0], DIM_STATES[1]))
        for i in range(1,Train_Configs.INPUT_IMAGES):
            state_rotate = my_utils.get_rotate_depth(i,state_path)
            state_rotate_3 = my_utils.copy_depth_to_3_channel(state_rotate)
            state.append(state_rotate_3)
        state = np.array(state)
        # normlize
        state = (state - np.min(state)) / (np.max(state) - np.min(state))
        # numpy to tensor
        state = torch.cuda.FloatTensor(state) #dim:[INPUT_IMAGE,3,224,224]

        # state = torch.cuda.FloatTensor(my_utils.copy_depth_to_3_channel(state_path).reshape(1,3,DIM_STATES[0],DIM_STATES[1]))#torch.unsqueeze(torch.FloatTensor(state) ,0)
        prob = np.min((EPSILON,1))
        p_select = np.array([prob, 1 - prob])
        # p_select = np.array([0, 1])
        selected_ac_type = np.random.choice([0, 1], p=p_select.ravel())

        if selected_ac_type == 0:#np.random.randn() <= Train_Configs.EPSILON:
            target_multiChannel_q_map = self.eval_net.forward(state)  # dim:[INPUT_IMAGES,1,224,224]
            # target_multiChannel_q_map = target_multiChannel_q_map[0]  # dim:[CHANNEL,224,224]
            action = my_utils.find_maxQ_in_qmap(target_multiChannel_q_map.cpu().detach().numpy())
            # action_value = self.eval_net.forward(state)
            # action = torch.max(action_value.cpu(), 1)[1].data.numpy() # get action whose q is max
            # action = action[0] #get the action index
            # add noise #
            ac_ty = '0'
        else:
            if np.random.randn() <= 0.5:
                action = my_utils.select_randpID_from_mask(state_path)
                ac_ty = '1'
            else:
                action = np.random.randint(0,DIM_ACTIONS)
                ac_ty = '2'
        #### change the action (id) to robot execution action type

        return ac_ty,action # the id of action

    def choose_action_for_eval(self, state_path):
        # for evaluation
        state = []
        state.append(my_utils.copy_depth_to_3_channel(state_path))  # dim:[3, DIM_STATES[0], DIM_STATES[1]]#.reshape(1, 3, DIM_STATES[0], DIM_STATES[1]))
        for i in range(1, Train_Configs.INPUT_IMAGES):
            state_rotate = my_utils.get_rotate_depth(i, state_path)
            state_rotate_3 = my_utils.copy_depth_to_3_channel(state_rotate)
            state.append(state_rotate_3)
        state = np.array(state)
        # normlize
        state = (state - np.min(state)) / (np.max(state) - np.min(state))
        # numpy to tensor
        state = torch.cuda.FloatTensor(state)  # dim:[INPUT_IMAGE,3,224,224]

        target_multiChannel_q_map = self.eval_net.forward(state)  # dim:[INPUT_IMAGES,1,224,224]
        action = my_utils.find_maxQ_in_qmap(target_multiChannel_q_map.cpu().detach().numpy())

        return action # the id of action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def load_batch_data(self,batch_list):#batch_list.dim:[batch_size]
        # print(batch_list)
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        for item in batch_list:
            # print (item)
            data = item.split('#')#state+'#'+str(action)+'#'+str(reward)+'#'+next_state
            action_id = int(data[1])
            batch_state.append(my_utils.copy_depth_to_3_channel(my_utils.get_rotate_depth(action_id,data[0])).reshape((3,DIM_STATES[0],DIM_STATES[1])))
            batch_action.append([int(data[1])])
            batch_reward.append([float(data[2])])
            batch_next_state.append(my_utils.copy_depth_to_3_channel(my_utils.get_rotate_depth(action_id,data[3])).reshape((3,DIM_STATES[0],DIM_STATES[1])))

        # normlize
        batch_state = (batch_state - np.min(batch_state)) / (np.max(batch_state) - np.min(batch_state))
        batch_next_state = (batch_next_state - np.min(batch_next_state)) / (np.max(batch_next_state) - np.min(batch_next_state))

        return torch.cuda.FloatTensor(batch_state),torch.cuda.LongTensor(batch_action),torch.cuda.FloatTensor(batch_reward),torch.cuda.FloatTensor(batch_next_state)
        # batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        # #note that the action must be a int
        # batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        # batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1: NUM_STATES+2])
        # batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % Train_Configs.Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter+=1

        mini_batch, idxs, is_weights = self.memory.sample(Train_Configs.BATCH_SIZE)#
        batch_state, batch_action, batch_reward, batch_next_state = self.load_batch_data(mini_batch)#dim:[1]

        eval_singleChannel_q_map = self.eval_net(batch_state)  # dim:[BATCH_SIZE,1,224,224]
        x_y_c_list = my_utils.translate_actionID_to_XY_and_channel_batch(batch_action)
        # old_val = target_multiChannel_q_map[0][c][x][y]
        batch_q = []
        # for xyc in x_y_c_list:
        for i in range(len(x_y_c_list)):
            xyc = x_y_c_list[i]
            batch_q.append([eval_singleChannel_q_map[i][0][xyc[0]][xyc[1]]])
        q_eval = torch.cuda.FloatTensor(batch_q)#self.eval_net(batch_state).gather(1, batch_action)#action: a value in range [0,DIM_ACTIONS-1]
        q_eval = Variable(q_eval.cuda(), requires_grad=True)
        target_singleChannel_q_map = self.target_net(batch_next_state).cpu().detach().numpy()#q_next,dim:[BATCH_SIZE,1,224,224]
        batch_q_next = []
        for b_item in target_singleChannel_q_map:#dim:[1,224,224]
            batch_q_next.append([np.max(b_item)])
        q_next = torch.cuda.FloatTensor(batch_q_next)
        # q_next = Variable(q_next.cuda(), requires_grad=True)

        q_target = batch_reward + Train_Configs.GAMMA*q_next
        q_target = Variable(q_target.cuda(), requires_grad=True)
        # self.average_q = q_eval.mean()
        weight_tensor = torch.cuda.FloatTensor(is_weights)#
        weight_tensor = weight_tensor.reshape((Train_Configs.BATCH_SIZE,1))
        weight_tensor = Variable(weight_tensor.cuda(), requires_grad=False)

        # loss_w_no = self.loss(q_eval, q_target)
        # print('loss_no.shape:',loss_w_no.shape)
        # print(loss_w_no)
        # loss_cc = self.loss(q_eval, q_target).mean()
        # print('batch_reward.shape:',batch_reward.shape)
        # print('q_next.shape:',q_next.shape)
        #
        # print('q_eval.shape:',q_eval.shape)
        # print('q_target.shape:',q_target.shape)
        # loss = self.loss(q_eval, q_target)
        # print('w.shape:',weight_tensor.shape)
        loss = (weight_tensor * self.loss(q_eval, q_target)).mean()##(torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        # print('loss.shape:',loss.shape)
        # print(loss)
        # print('loss_no_mean:',float(loss_cc_no),',loss_cc:',float(loss_cc),',loss:',float(loss))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss),float(q_eval.mean())

def main():
    net = Dqn()
    # net.cuda()
    EPSILON = Train_Configs.EPSILON
    print("The DQN is collecting experience...")
    step_counter_list = []
    log_writer = SummaryWriter('logs/')  # ('logs/')
    log_txt_writer = open('logs/train_log.txt','a')
    step_counter = 0
    time_start = time.time()
    for episode in range(Train_Configs.EPISODES):
        state = env.reset()
        sum_reward = 0
        while True:
            step_counter += 1
            ac_type, action = net.choose_action(state,EPSILON)
            EPSILON = Train_Configs.EPSILON + step_counter*1e-6
            # print('action_id:',action)
            next_state, reward, done = env.step(ac_type,action,state,step_counter)
            if done == -1:
                print('Env error occured, restart simulation...')
                break
            # reward = reward * 100 if reward >0 else reward * 5
            net.store_trans(state, action, reward, next_state,done)
            sum_reward += reward

            if net.memory.tree.n_entries >= 1000:#Train_Configs.MEMORY_CAPACITY:
                l, mean_q = net.learn()
                if step_counter >= 1100:
                    if step_counter == 1100:
                        time_start = time.time()
                    log_writer.add_scalar('loss', float(l), global_step=step_counter)
                    log_writer.add_scalar('mean_q', float(mean_q), global_step=step_counter)
                    log_txt_writer.write('used time:'+str((time.time()-time_start)/60)+',step:'+str(step_counter)+',loss:'+str(float(l))+',mean_q:'+str(float(mean_q)))
                    log_txt_writer.write('\n')
                    #time format,hour
                # print('train network, episode:',episode,', step:',step_counter,', loss:',l)
            if done == 1 or step_counter == 200:
                print("episode {}, the sum reward is {}".format(episode, round(sum_reward, 4)))
                # step_counter_list.append(step_counter)
                # net.plot(net.ax, step_counter_list)
                break

            state = next_state

        if (episode+1) % 200 == 0 and step_counter >= 1200:
            torch.save(net.eval_net.state_dict(), 'models/ep_' + str(episode+1) + '_params.pkl')
            print('#####################   save model   #####################')

    torch.save(net.eval_net.state_dict(), 'models/final_params.pkl')
    log_txt_writer.close()

def evaluation(model_path):
    net = Dqn()#net.eval_net.state_dict()
    net.eval_net.load_state_dict(torch.load('models/'+model_path))# (torch.load('models/model_D12_M5/step_35000_params.pkl'))

    EVALUATION_EP = 50
    MAX_STEP_FOR_EVERY_EP = 50

    log_txt_result = open('evaluation_result/'+model_path.replace('pkl','txt'),'a')

    for episode in range(EVALUATION_EP):
        print ('-----------------  EP ', str(episode + 1), ' start  ------------------')
        obj_num, state = env.reset_eval()
        step_count = 0
        success_grasp_obj_count = 0
        while step_count <= MAX_STEP_FOR_EVERY_EP:
            step_count += 1
            action = net.choose_action_for_eval(state)
            next_state, reward, done = env.step_eval(action,state,step_count)
            if done == -1:
                print('Env error occured, restart simulation...')
                break

            if done == 1:
                success_grasp_obj_count += 1
                if success_grasp_obj_count == obj_num:
                    break

            state = next_state

        log_txt_result.write('EP'+str(episode+1)+',gen_obj_num:'+str(obj_num)+',grasp_obj_num:'+str(success_grasp_obj_count)+',used_step:'+str(step_count)+'\n')

    log_txt_result.close()

if __name__ == '__main__':
    # main()
    # myN = Net()
    # input = torch.randn(1,1,64,64)
    #
    # out = myN.forward(input)
    # print(out.shape)
    # print(out)
    evaluation('ep_1800_params.pkl')