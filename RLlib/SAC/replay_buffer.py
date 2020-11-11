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
    def __init__(self, cfg, batchsize, gpu_id, device):
        self.capacity = cfg.replay_size
        self.batch_size = batchsize
        self.device = device
        self.dim_state = cfg.dim_state  # 124
        self.dim_action = cfg.dim_action_acc + cfg.dim_action_fix  # 61
        self.dim_hidden = cfg.hidden_size
        self.dim_labels = 5
        # determine the dimension of experience replay
        # (state, action, reward, next_state, rnn_state, labels, done)
        self.dim_mem = self.dim_state + self.dim_action + 1 + self.dim_state + 2 * self.dim_hidden + self.dim_labels + 1
        self.position, self.buffer = self.create_buffer(gpu_id)
        self.length = 0

    def create_buffer(self, gpu_id):
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(h)
        freeGPUMem = info.free * 1e-9  # with Gb unit
        reqGPUMem = self.dim_mem * self.batch_size * self.capacity * 4 * 1e-9 # 4 bytes per float32 element
        if freeGPUMem >= reqGPUMem + 1: # add an extra Gb requirement for memory safety
            buffer = torch.from_numpy(np.empty((self.capacity, self.batch_size, self.dim_mem), dtype=np.float32)).to(self.device)
            return 0, buffer
        else:
            print("At least %d GB GPU memory are requried!"%(reqGPUMem + 1))
            raise MemoryError

    def push(self, state, actions, reward, next_state, rnn_state, labels, done):
        """ state: GPU(B, dim_state)
            actions: GPU(B, dim_action)
            reward: GPU(B, 1)
            next_state: GPU(B, dim_state)
            rnn_state: list with 2 items, for each: GPU(B, hidden_size)
            labels: GPU(B, dim_label)
            done: GPU (B, 1)
        """
        rnn_state_gpu = torch.cat(rnn_state, dim=-1)
        transition = torch.cat((state, actions, reward, next_state, rnn_state_gpu, labels, done), dim=-1)  # (B, 444)
        # insert the transition into buffer
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.length += 1

    def sample(self, batch_size, device):
        """Sampling on GPU device"""
        assert self.length > batch_size, "Not enough transitions!"
        indices = np.random.randint(0, self.length, size=batch_size) % self.capacity
        indices = torch.LongTensor(indices).to(device)
        data_batch = torch.index_select(self.buffer, 0, indices)
        # parse the results
        # state
        start, end = 0, self.dim_state
        state = data_batch[:, :, start: end].view(-1, self.dim_state)
        # action
        start, end = start + self.dim_state, end + self.dim_action
        action = data_batch[:, :, start: end].view(-1, self.dim_action)
        # reward
        start, end = start + self.dim_action, end + 1
        reward = data_batch[:, :, start: end].view(-1, 1)
        # next_state
        start, end = start + 1, end + self.dim_state
        next_state = data_batch[:, :, start: end].view(-1, self.dim_state)
        # rnn_state
        start, end = start + self.dim_state, end + 2 * self.dim_hidden
        rnn_state = (data_batch[:, :, start: start + self.dim_hidden].view(-1, self.dim_hidden), 
                     data_batch[:, :, start + self.dim_hidden: end].view(-1, self.dim_hidden))
        # labels
        start, end = start + 2 * self.dim_hidden, end + self.dim_labels
        labels = data_batch[:, :, start: end].view(-1, self.dim_labels)
        # done
        start, end = start + self.dim_labels, end + 1
        mask = data_batch[:, :, start: end].view(-1, 1)
        return state, action, reward, next_state, rnn_state, labels, mask

    def __len__(self):
        return self.length