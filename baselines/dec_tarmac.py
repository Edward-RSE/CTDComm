# Cleaned up and rearranged from the version by Yaru Niu for MAGIC
# Itself slightly adapted from a version from Dr. Abhishek Das

import math

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from models import MLP
from tar_comm import TarCommNetMLP
from dec_agent import DecAgent
from action_utils import select_action, translate_action

class DecTarMAC(TarCommNetMLP):
    """
    Complete module combining all decentralised agents during centralised training.
    Mostly replicates TarMAC (TarCommNetMLP class based on "TarMAC: Targeted Multi-Agent Communication"
        by Das et al. 2019) whilst allowing for decentralised behaviour and communication.
    """

    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        
        Returns: None
        """

        # Inherit init_weights and get_agent_mask from parent
        super(TarCommNetMLP, self).__init__()
        
        # Set up all the stuff from TarComm
        self.num_inputs = num_inputs
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes

        # Set up individual agents
        self.agents = nn.ModuleList([DecAgent(args, num_inputs, a_id) for a_id in range(args.nagents)])

        # Centralised value estimation
        self.cave = args.cave
        self.message_augment = args.message_augment
        self.v_augment = args.v_augment
        if args.cave and args.message_augment:
            # Include both the adjacency data and the message content (attention value) as input to the value head
            value_input_size = 2 * self.hid_size + (args.comm_passes * args.nagents * args.nagents)
        elif args.cave:
            # Include the adjacency data as input to the value head
            value_input_size = self.hid_size + (args.comm_passes * args.nagents * args.nagents)
        elif args.message_augment or args.v_augment:
            # Include the message content (aggregated message or attention value) as input to the value head
            value_input_size = 2 * self.hid_size
        else:
            value_input_size = self.hid_size
        self.value_head = nn.Linear(value_input_size, 1)

        return None

    def init_hidden(self, batch_size):
        """
        Initialise the LSTM hidden state and cell state for each agent
            then stack them
        
        Arguments:
            batch_size {number}
        
        Returns:
            hidden_states {tensor} -- LSTM hidden states for all agents over all batches (N x batch_size x hid_size)
            cell_states {tensor} -- LSTM cell states for all agents over all batches (N x batch_size x hid_size)
        """

        hidden_states, cell_states = [], []
        for agent in self.agents:
            hidden_state, cell_state = agent.init_hidden(batch_size)
            hidden_states.append(hidden_state)
            cell_states.append(cell_state)
        hidden_states = torch.stack(hidden_states)
        cell_states = torch.stack(cell_states)

        # Transpose for backwards compatibility
        hidden_states = torch.transpose(hidden_states, 0, 1)
        cell_states = torch.transpose(cell_states, 0, 1)

        return hidden_states, cell_states
    
    def separate_input(self, x):
        # TODO: confirm the expected shapes of the hidden and cell states then transpose them if necessary
        """
        Separate the input to the forward function so that it can be sent to decentralised agents.
        Also parses the hidden and cell states where applicable.
        Similar to the 'forward_state_encoder' function in TarComm except that the encoder is
            removed to be executed by the decentralised agents.
        
        Arguments:
            x -- Concatenated observations (states) from the environment, may also
                contain the hidden and cell states
        
        Returns:
            x {tensor} -- Concatenated observations (states) from the environment,
                transposed for decentralisation (N x batch_size x num_inputs)
            hidden_state {tensor} -- Concatenated agent hidden states,
                transposed for decentralisation (N x batch_size x hid_size)
            cell_state {tensor} -- Concatenated agent cell states,
                transposed for decentralisation (N x batch_size x hid_size)
        """
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x
            
            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras

                # Transpose to separate by agent
                cell_state = torch.transpose(cell_state, 0, 1)
            else:
                hidden_state = extras
        else:
            hidden_state = x

        # Transpose to shape (n, batch_size, num_inputs)
        hidden_state = torch.transpose(hidden_state, 0, 1)
        assert len(x.shape) == 3, f"input has {len(x.shape)} dimensions, not 3. Shape is {x.shape}"
        x = torch.transpose(x, 0, 1)

        return x, hidden_state, cell_state

    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Centralised forward function (including critic) for training Dec-TarMAC.
        B: Batch Size: Normally 1 in case of episode
        N: number of agents
        H: size of the hidden state for recurrent architectures (args.hid_size)
        Note: Need to reshape things as (N x B x ...) rather than (B x N x ...)
            to pass things to decentralised agents

        Arguments:
            x {tensor} -- State of the agents (N x num_inputs)
            info {dict} -- Information passed between this module
                and the trainer/executor function running it

        Returns:
            action {tensor} --
                If continuous: actions chosen by each agent (dims are env dependent)
                If discrete: probability distribution representing action preferences
                    for each agent (dims are env dependent)
            value_head {tensor} -- critics' value estimates (N)
            hidden_state {tuple} (if args.recurrent): Contains
                hidden_state: Recurrent hidden state (H)
                cell_state: Recurrent cell state (H)
            adjacency_data {tensor} (if args.save_adjacency) -- Adjacency matrices
                for all communication rounds, including all agents (N x N)
        """

        # Process and separate observations for each agent (maintaining batch structure)
        x, hidden_state, cell_state = self.separate_input(x)
        batch_size = x.size()[1]
        n = self.nagents

        # Encode inputs
        # if self.args.recurrent:
        # #     x = torch.stack([self.agents[a_id].forward_state_encoder(x[a_id], hidden_state[a_id], batch_size) for a_id in range(n)])
        # # else:
        # #     hidden_state = torch.stack([self.agents[a_id].forward_state_encoder(x[a_id], hidden_state[a_id]) for a_id in range(n)])
        for a_id in range(n): self.agents[a_id].forward_state_encoder(x[a_id], hidden_state[a_id], cell_state[a_id], batch_size)

        if self.args.save_adjacency:
            adjacency_data = torch.zeros([self.comm_passes, n, n])
        
        # For decentralised environments, this mask should be dealt with by individual agents
        #   rather than through 'info' passed through this central process but that
        #   isn't compatible with the environments used here
        _, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask_alive = agent_mask.clone()

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn:
            comm_action = torch.tensor(info['comm_action'])
            # Masks have shape (batch_size x N x N x 1)
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()
        
        # #### Check the shapes of x, hidden state and cell_state to make sure that I'm actually sending the correct info to each agent!
        # assert False, "Nothing broken up to the first comm, is everything the correct shape?"

        for i in range(self.comm_passes):
            # Pass communications to each agent and get the attention vectors out
            keys, values = [], []
            for agent in self.agents:
                key, value = agent.get_attn()
                keys.append(key)
                values.append(value)
            keys = torch.stack(keys)
            values = torch.stack(values)

            # #### Check the shapes of x, hidden state and cell_state to make sure that I'm actually sending the correct info to each agent!
            # assert False, "Nothing broken up to the attention vector generation, is everything the correct shape?"

            # Send attention vectors to each agent and get new communication out
            attn, hidden_state, cell_state, comm = [], [], [], []
            for a_id in range(n):
                agent_attn, agent_hidden_state, agent_cell_state, agent_comm = self.agents[a_id].attn2hid(keys, values, agent_mask[:, :, a_id], agent_mask_alive[:, :, a_id], i)
                attn.append(agent_attn)
                hidden_state.append(agent_hidden_state)
                cell_state.append(agent_cell_state)
                comm.append(agent_comm)
            attn = torch.stack(attn).transpose(0,1)
            hidden_state = torch.stack(hidden_state).transpose(0,1)
            cell_state = torch.stack(cell_state).transpose(0,1)
            if self.message_augment:
                comm_stack = torch.stack(comm).transpose(0,1)
            else:
                comm_stack = None

            # Save attentions as a proxy for an adjacency matrix
            if self.args.save_adjacency:
                adjacency_data[i] = attn

            # #### Check the shapes of x, hidden state and cell_state to make sure that I'm actually sending the correct info to each agent!
            # assert False, "Nothing broken up to the communication, is everything the correct shape?"
        
        # Agents pick actions decentrally
        if self.args.continuous:
            action_mean, action_log_std, action_std = [], [], []
            for agent in self.agents:
                agent_action_mean, agent_action_log_std, agent_action_std = agent.get_action()
                action_mean.append(agent_action_mean)
                action_log_std.append(agent_action_log_std)
                action_std.append(agent_action_std)
            action = (torch.stack(action_mean).transpose(0,1), torch.stack(action_log_std).transpose(0,1), torch.stack(action_std).transpose(0,1))
        else:
            # I can't just do a transpose because we're working with lists
            #   From (N x heads x action_space) to (heads x N x action_space)
            agentwise_action = [agent.get_action() for agent in self.agents]
            action = [torch.stack([agentwise_action[a_id][head] for a_id in range(n)]).transpose(0,1) for head in range(len(agentwise_action[0]))]
        
        # #### Check the shapes of x, hidden state and cell_state to make sure that I'm actually sending the correct info to each agent!
        # assert False, "Nothing broken up to action selection, is everything the correct shape?"

        # Centralised critic
        if self.cave and self.message_augment:
            # Add the adjacency data and the aggregated messages and add them to the hidden state as inputs to the critic
            value_input = torch.cat([hidden_state.view(batch_size, n, self.hid_size),
                                     comm_stack.view(batch_size, n, self.hid_size),
                                     torch.flatten(adjacency_data).expand((batch_size, n, -1))], dim=2)
        elif self.cave and self.v_augment:
            # Add the adjacency data and the attention values and add them to the hidden state as inputs to the critic
            value_input = torch.cat([hidden_state.view(batch_size, n, self.hid_size),
                                     values.transpose(0,1).view(batch_size, n, self.hid_size),
                                     torch.flatten(adjacency_data).expand((batch_size, n, -1))], dim=2)
        elif self.cave:
            # Flatten the adjacency data and add it to the hidden state to go into the critic
            value_input = torch.cat([hidden_state.view(batch_size, n, self.hid_size),
                                     torch.flatten(adjacency_data).expand((batch_size, n, -1))], dim=2)
            # stacked_adjacency = torch.stack([torch.stack([torch.flatten(adjacency_data) for _ in range(n)]) for _ in range(batch_size)])
            # value_input = torch.cat([hidden_state.view(batch_size, n, self.hid_size), stacked_adjacency], dim=2)
        elif self.message_augment:
            value_input = torch.cat([hidden_state.view(batch_size, n, self.hid_size),
                                     comm_stack.view(batch_size, n, self.hid_size)], dim=2)
        elif self.v_augment:
            # Add the adjacency data and the attention values and add them to the hidden state as inputs to the critic
            value_input = torch.cat([hidden_state.view(batch_size, n, self.hid_size),
                                     values.transpose(0,1).view(batch_size, n, self.hid_size)], dim=2)
        else:
            value_input = hidden_state
        value_head = self.value_head(value_input)
        
        if self.args.save_adjacency and self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone()), adjacency_data.detach().numpy()
        elif self.args.save_adjacency:
            return action, value_head, adjacency_data
        elif self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head