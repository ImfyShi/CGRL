import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torch
import copy
import numpy as np

from agent.model import Actor, Critic
import utilities as utilities

MAX_GRAD_NORM = 1.
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   
    dict(name='clip', epsilon=0.02),            
][1]     

A_LR = 1e-4
C_LR = 5e-4
UPDATE_STEPS = 1

class PPO(nn.Module):

    def __init__(self, device, model_state=None):
        super(PPO, self).__init__()

        self.criticNet = Critic()
        self.actorNet = Actor()
        self.criticNet.to(device)
        self.actorNet.to(device)
        self.c_optimizer = opt.Adam(self.criticNet.parameters(), lr=C_LR)
        self.a_optimizer = opt.Adam(self.actorNet.parameters(), lr=A_LR)

        if model_state is not None:
            self.criticNet.load_state_dict(model_state['criticNet'])
            self.actorNet.load_state_dict(model_state['actorNet'])
            self.c_optimizer.load_state_dict(model_state['c_optimizer'])
            self.a_optimizer.load_state_dict(model_state['a_optimizer'])


    def _forward_actor(self, states):
        return self.actorNet(states)

    def _forward_critic(self, states):
        return self.criticNet(states)

    def _update_critic(self, item_features, edge_features, edge_indices, column_features, rewards, item_nums):
        result_closs = 0
        for _ in range(UPDATE_STEPS):
            v = self._get_v(trainable=True, item_features=item_features, edge_indices=edge_indices, edge_features=edge_features, column_features=column_features, item_nums=item_nums)
            advantage = rewards - v
            c_loss = torch.mean(torch.square(advantage))
            self.c_optimizer.zero_grad()
            c_loss.backward()
            self.c_optimizer.step()
            # result_closs = c_loss.detach().numpy()
            result_closs = c_loss
        return result_closs

    def _update_actor(self, item_features, edge_features, edge_indices, column_features, actions, rewards, column_nums, item_nums):
        # params = list(self.actorNet.item_embedding.named_parameters())
        # print(params.__len__())
        # print(params[0])
        # print(params[1])

        self.oldactorNet = copy.deepcopy(self.actorNet)
        # self.oldactorNet.load_state_dict(self.actorNet.state_dict())
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        v = self._get_v(trainable=False, item_features=item_features, edge_indices=edge_indices, edge_features=edge_features, column_features=column_features, item_nums=item_nums)
        advantage = rewards - v

        # update actor
        result_aloss = 0
        for _ in range(UPDATE_STEPS):
            pi = self._get_probability_distribution_of_states(actor=self.actorNet, item_features=item_features, edge_indices=edge_indices, edge_features=edge_features, column_features=column_features, column_nums=column_nums)
            oldpi = self._get_probability_distribution_of_states(actor=self.oldactorNet, item_features=item_features, edge_indices=edge_indices, edge_features=edge_features, column_features=column_features, column_nums=column_nums)
            pi_action = pi.gather(1,actions)
            oldpi_action = oldpi.gather(1,actions)
            ratio = pi_action/oldpi_action
            surr = ratio * advantage
            # mean
            aloss = -torch.mean(torch.minimum(
                surr,
                torch.clamp(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*advantage)
            )
            # batch_aloss = self._get_aloss(advs=advantage, states=states, actions=actions, batch_ratio=ratio, batch_surr=surr, batch_aloss=aloss)
            # print("batch_aloss-aloss={}".format(batch_aloss-aloss))
            self.a_optimizer.zero_grad()
            aloss.backward(retain_graph=True)
            self.a_optimizer.step()

            # result_aloss = aloss.detach().numpy()
            result_aloss = aloss
        return result_aloss

    def _get_aloss(self, advs, states, actions,batch_ratio,batch_surr,batch_aloss):
        aloss = None
        ele_num = 0
        for adv, state, action in zip(advs, states, actions):
            ratio = self._get_prob_of_actions(self.actorNet, state, action) / (
                        self._get_prob_of_actions(self.oldactorNet, state, action) + 1e-5)
            surr = ratio * adv
            # mean
            ele_num += len(ratio)
            if aloss is None:
                aloss = -torch.sum(torch.minimum(
                    surr,
                    torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv)
                )
            else:
                aloss += -torch.sum(torch.minimum(
                    surr,
                    torch.clamp(ratio, 1. - METHOD['epsilon'], 1. + METHOD['epsilon']) * adv)
                )
        aloss /= ele_num
        return aloss

    def update(self, item_features, edge_features, edge_indices, column_features, actions, rewards, column_nums, item_nums):
        '''
        # states = torch.tensor(states, dtype=torch.float32)
        # big_ele_states, small_ele_states = states[:, 0:self.big_ele_num], states[:, self.big_ele_num:-1]
        big_ele_states = big_ele_states.to(torch.float32)
        small_ele_states = small_ele_states.to(torch.float32)
        # actions = torch.tensor(np.array(actions)[:, np.newaxis], dtype=torch.int64)
        # rewards = torch.tensor(np.array(rewards)[:, np.newaxis], dtype=torch.float32)
        actions = actions[:, np.newaxis]
        rewards = rewards[:, np.newaxis]
        '''
        closs = self._update_critic(item_features, edge_features, edge_indices, column_features, rewards, item_nums)

        aloss = self._update_actor(item_features, edge_features, edge_indices, column_features, actions, rewards, column_nums, item_nums)
        return closs, aloss

    def _get_prob_of_actions(self, actor, states, actions):
        pi = actor(states)
        prob = pi.log_prob(actions).exp()
        return prob

    def choose_action(self, item_features, edge_features, edge_indices, column_features, is_stochastic=False):
        '''
        states = states[np.newaxis,:]
        big_ele_states, small_ele_states = states[:,0:self.big_ele_num], states[:,self.big_ele_num:-1]
        torch.tensor(big_ele_states, dtype=torch.float32), torch.tensor(small_ele_states, dtype=torch.float32)
        '''
        prob_weights = self._get_probability_distribution_of_states(actor=self.actorNet, item_features=item_features, edge_indices=edge_indices, edge_features=edge_features, column_features=column_features, column_nums=[len(column_features)])
        if is_stochastic:
            action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.cpu().ravel().detach().numpy())
            return action
        else:
            action = prob_weights.argmax(dim=1)
            return action[0]
        # return action[0]

    def _get_probability_distribution_of_states(self, actor, item_features, edge_indices, edge_features, column_features, column_nums):
        output = actor(item_features, edge_indices, edge_features, column_features)
        output = utilities.pad_tensor(input_=output, pad_sizes=column_nums, pad_value=-10000000)
        probability_distribution = torch.softmax(input=output, dim=1)
        return probability_distribution

    def _get_v(self, trainable, item_features, edge_indices, edge_features, column_features, item_nums):
        output = self.criticNet(item_features, edge_indices, edge_features, column_features)
        output = utilities.pad_tensor(input_=output, pad_sizes=item_nums, pad_value=0)
        v = output.sum(1).unsqueeze(dim=-1)
        if trainable:
            return v
        else:
            # return v.detach().numpy()
            return v
