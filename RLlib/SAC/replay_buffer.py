import numpy as np
import random
from pynvml import *
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, rnn_state, labels, done):
        state_cpu = state.cpu().numpy().flatten()
        action_cpu = action.flatten()
        next_state_cpu = next_state.cpu().numpy().flatten()
        rnn_state_cpu = torch.cat((rnn_state[0].unsqueeze(0), rnn_state[1].unsqueeze(0)), dim=0).cpu().numpy().reshape(-1, rnn_state[0].size(-1))

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state_cpu, action_cpu, reward, next_state_cpu, rnn_state_cpu, labels, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, rnn_state, labels, mask = map(np.stack, zip(*batch))
        # move data to GPU
        if rnn_state[:, 0] is not None:
            rnn_state = (torch.from_numpy(rnn_state[:, 0]).to(device), torch.from_numpy(rnn_state[:, 1]).to(device))
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(device)
        labels = torch.FloatTensor(labels).to(device)
        mask = torch.FloatTensor(mask).to(device).unsqueeze(1)
        return state, action, reward, next_state, rnn_state, labels, mask

    def __len__(self):
        return len(self.buffer)


class ReplayMemoryGPU:
    def __init__(self, cfg):
        self.capacity = cfg.SAC.replay_size
        self.device = cfg.device
        self.dim_state = cfg.ENV.dim_state
        self.dim_action = cfg.ENV.dim_action
        self.dim_hidden = cfg.SAC.hidden_size
        self.dim_labels = 5
        # determine the dimension of experience replay
        # (state, action, reward, next_state, rnn_state, labels, done)
        self.dim_mem = self.dim_state + self.dim_action + 1 + self.dim_state + 2 * self.dim_hidden + self.dim_labels + 1
        self.position, self.buffer = self.create_buffer()

    def create_buffer(self):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(self.device.index)
        info = nvmlDeviceGetMemoryInfo(h)
        freeGPUMem = info.free * 1e-9  # with Gb unit
        reqGPUMem = self.dim_mem * self.capacity * 4 * 1e-9 # 4 bytes per float32 element
        if freeGPUMem >= reqGPUMem + 1: # add an extra Gb requirement for memory safety
            buffer = torch.from_numpy(np.empty((self.capacity, self.dim_mem), dtype=np.float32)).to(self.device)
            return 0, buffer
        else:
            print("At least %d GB GPU memory are requried!"%(reqGPUMem + 1))
            raise MemoryError

    def push(self, state, action, reward, next_state, rnn_state, labels, done):
        """ state: GPU(1, dim_state)
            action: CPU(dim_action,)
            reward: scalar
            next_state: GPU(1, dim_state)
            rnn_state: list with 2 items, for each: GPU(1, hidden_size)
            labels: CPU(5,)
            done: scalar
        """
        action_gpu = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        reward_gpu = torch.FloatTensor([[reward]]).to(self.device)
        rnn_state_gpu = torch.cat(rnn_state, dim=-1)  # here we assume the batchsize must be 1!!
        labels_gpu = torch.FloatTensor(labels).unsqueeze(0).to(self.device)
        done_gpu = torch.FloatTensor([[done]]).to(self.device)
        transition = torch.cat((state, action_gpu, reward_gpu, next_state, rnn_state_gpu, labels_gpu, done_gpu), dim=-1)
        # insert the transition into buffer
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device):
        """Sampling on GPU device"""
        indices = torch.LongTensor(np.random.randint(0, self.position, size=batch_size)).to(device)
        data_batch = torch.index_select(self.buffer, 0, indices)
        # parse the results
        start, end = 0, self.dim_state
        state = data_batch[:, start: end]
        start, end = start + self.dim_state, end + self.dim_action
        action = data_batch[:, start: end]
        start, end = start + self.dim_action, end + 1
        reward = data_batch[:, start: end].unsqueeze(1)
        start, end = start + 1, end + self.dim_state
        next_state = data_batch[:, start: end]
        start, end = start + self.dim_state, end + 2 * self.dim_hidden
        rnn_state = (data_batch[:, start: start + self.dim_hidden], data_batch[:, start + self.dim_hidden: end])
        start, end = start + 2 * self.dim_hidden, end + self.dim_labels
        labels = data_batch[:, start: end]
        start, end = start + self.dim_labels, end + 1
        mask = data_batch[:, start: end].unsqueeze(1)
        return state, action, reward, next_state, rnn_state, labels, mask