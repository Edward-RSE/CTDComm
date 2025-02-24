# Cleaned up and rearranged from the version by Yaru Niu for MAGIC
# Itself slightly adapted from a version from Dr. Abhishek Das

import math

import torch
import torch.nn.functional as F
from torch import nn

from ctdcomm.policy_nets.models import MLP
from ctdcomm.policy_nets.tar_comm import TarCommNetMLP


class DecAgent(TarCommNetMLP):
    """
    The code for an individual agent in the Dec-TarMAC framework.
    Set up to fit into the larger forward function of the DecTarMAC
        module whilst remaining separable during execution.
    Separated into multiple functions since the communication between
        agents must be handled outside of this class.
    """

    def __init__(self, args, num_inputs, agent_id):
        """
        Initialization method for the decentralised agents to be used in Dec-TarMAC.
        Sets up various internal neural network layers and weights
        Note: batch_size dimension is often covered

        Arguments:
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
            agent_id {number} -- Unique id for this decentralised agent, use None if the agents share weights
        """

        super(TarCommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent

        # New attributes
        self.agent_id = agent_id

        # Input processing
        self.encoder = nn.Linear(num_inputs, args.hid_size)
        self.tanh = nn.Tanh()

        # This is commented out of the TarMAC forward function without explanation
        # if args.recurrent:
        #     self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        # Attentional communication modules
        self.state2query = nn.Linear(args.hid_size, args.qk_hid_size)
        self.state2key = nn.Linear(args.hid_size, args.qk_hid_size)
        self.state2value = nn.Linear(args.hid_size, args.value_hid_size)

        # Function for processing communication
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(args.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(args.comm_passes)])

        # initialise weights as 0
        if args.comm_init == 'zeros':
            for i in range(args.comm_passes):
                self.C_modules[i].weight.data.zero_()

        # The main function for converting current hidden state to next state
        if args.recurrent:
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)
        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(args.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(args.comm_passes)])

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        return None

    def init_hidden(self, batch_size):
        """
        Initialise the LSTM hidden state and cell state for this agent

        Arguments:
            batch_size {number}

        Returns:
            hidden state {tensor} -- (batch_size x hid_size)
            cell state {tensor} -- (batch_size x hid_size)
        """
        # dim 0 = num of layers * num of direction
        return (torch.zeros(batch_size, self.hid_size, requires_grad=True),
                torch.zeros(batch_size, self.hid_size, requires_grad=True))

    def forward_state_encoder(self, x, hidden_state, cell_state, batch_size):
        """
        Encode the input from the environment and store internally
        Also, store the hidden state and batch size

        Arguments:
            x {tensor} -- Agent state (batch_size x num_inputs)
            hidden_state {tensor} -- Previous hidden state for the networks (batch_size x hid_size)
            cell_state {tensor} -- Previous cell state for the networks (batch_size x hid_size)
            batch_size {number} -- Number of batches being processed in parallel

        Returns: None (everything is stored internally)
            (self.hidden_state {tensor} -- Previous hidden state for the networks (batch_size x hid_size))
            (self.cell_state {tensor} -- Previous cell state for the networks (batch_size x hid_size))
            (self.x {tensor} -- Encoded agent state (batch_size x hid_size))
        """
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        self.batch_size = batch_size

        if self.args.recurrent:
            self.x = self.encoder(x)
            # TODO: Test whether the below improves performance (commented out of the original TarMAC code)
            # hidden_state = self.tanh(self.hidd_encoder(prev_hidden_state) + x)
        else:
            self.x = self.encoder(x)
            self.x = self.tanh(x)
            self.hidden_state = x

        return None

    def get_attn(self):
        """
        Calculate the vectors for determining attention based on the hidden state.
        Note that only the keys and values get passed to other agents. The queries are stored
            internally to calculate the attention coefficients in Attn2Hid.

        Arguments:
            None (Updates based on the hidden state updated elsewhere)

        Returns:
            key - attention key vector, k (batch_size x 16)
            value - attention value vector, v. The learned message passed between agents (batch_size x hid_size)
            (self.query - attention query vector, q. Stored locally and used later (batch_size x 16))
        """
        comm = self.hidden_state.view(self.batch_size, self.hid_size) if self.args.recurrent else self.hidden_state

        if self.args.comm_mask_zero:
            comm_mask = torch.zeros_like(comm)
            comm = comm * comm_mask

        self.query = self.state2query(comm)
        key = self.state2key(comm)
        value = self.state2value(comm)

        return key, value

    def attn2hid(self, keys, values, agent_mask, agent_mask_alive, comm_pass):
        """
        Calculate attention coefficients then updates the hidden state and cell state based on
            the input and communication from other agents.

        Arguments:
            keys {tensor} -- stack of attention key vectors from all agents (N x batch_size x 16)
            values {tensor} -- stack of attention value vectors from all agents (N x batch_size x hid_size)
            agent_mask {tensor} -- alive mask from the environment, filtered through the hard attention mask if using hard attention (IC3Net) (batch_size x N x 1)
            agent_mask_alive {tensor} -- alive mask from the environment (batch_size x N x 1)

        Returns:
            attn {tensor} -- attention coefficients (N x batch_size)
            hidden_state {tensor} -- agent hidden state (batch_size x hid_size)
            cell_state {tensor} -- agent cell state (batch_size x hid_size)
        """

        # Calculate attention coefficients
        #   Need batch_size on the first dimension for broadcasting purposes
        #   The unsqueeze is to maintain the matrix shapes of (num_receivers=1 x 16) x (16 x num_senders=nagents)
        scores = torch.matmul(self.query.unsqueeze(-2), keys.transpose(0,1).transpose(-2, -1))
        scores = scores.squeeze(-2) / math.sqrt(self.hid_size) #Shouldn't this be sqrt(16) since that's the size of the key/value?
        scores = scores.masked_fill(agent_mask.squeeze(-1) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = attn * agent_mask.squeeze(-1)
        comm = torch.matmul(attn.unsqueeze(-2), values.transpose(0,1)).squeeze(-2)
        # We're using [:, agent_id] rather than [:, 0], but they should be equivalent (or more correct if it ever matters)
        comm  = comm * agent_mask_alive[:, self.agent_id].expand(self.batch_size, self.hid_size)
        c = self.C_modules[comm_pass](comm)

        if self.args.recurrent:
            # skip connection - combine comm. matrix and encoded input for all agents
            inp = self.x + c

            inp = inp.view(self.batch_size, self.hid_size)

            self.hidden_state, self.cell_state = self.f_module(inp, (self.hidden_state, self.cell_state))
        else:
            # Get next hidden state from f node
            # and Add skip connection from start and sum them
            # Python's built-in sum is meant to concatenate these, torch.sum() would actually add element-wise
            self.hidden_state = sum([self.x, self.f_modules[comm_pass](self.hidden_state), c])
            self.hidden_state = self.tanh(self.hidden_state)

        if self.args.message_augment:
            return attn, self.hidden_state, self.cell_state, comm
        else:
            return attn, self.hidden_state, self.cell_state, None

    def get_action(self):
        """
        Get information necessary for selecting an action, based on the hidden state updated during
            the communication rounds.

        Arguments: None

        Returns:
            For a continuous action space:
                action_mean {tensor} -- The mean of the probability distribution over the action space (batch_size x dim_actions)
                action_log_std {tensor} -- The log standard deviation of the probability distribution over the action space (batch_size x dim_actions)
                action_std {tensor} -- The standard deviation of the probability distribution over the action space (batch_size x dim_actions)
            For a discrete action space:
                actions {list of tensors} -- List of actions (shape varies based on the environment)
        """

        if self.continuous:
            action_mean = self.action_mean(self.hidden_state)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # Return the action as a distributino tobe sampled later
            return (action_mean, action_log_std, action_std)
        else:
            # Return discrete actions
            return [F.log_softmax(head(self.hidden_state), dim=-1) for head in self.heads]