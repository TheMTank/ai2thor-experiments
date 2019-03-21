"""
ActorCritic Adapted from: https://github.com/ikostrikov/pytorch-a3c/blob/master/model.py
A3C_LSTM_GA adapted from: https://github.com/devendrachaplot/DeepRL-Grounding/blob/master/models.py

Main A3C model which outputs predicted value, action logits and hidden state.
Includes helper functions too for weight initialisation and dynamically computing LSTM/flatten input
size.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    # todo can still explain more
    """
    Weights are normalized over their column. Also, allows control over std which is useful for
    initialising action logit output so that all actions have similar likelihood
    """

    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m):
    # todo add docstring on why magic numbers like 6 and name of normalisation
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    """
    Ikostrikov's implementation of A3C (https://arxiv.org/abs/1602.01783).

____________________________________________________________________________________________________

                            A3C policy model architecture

   Image Processing module -> Flattened output ->     Policy Learning Module    --->  Final output
           ______________________                ___________________________________
          |     _______    ____ |     __       |      ___________                 |
image ->  | 4x |conv2d| + |ELU| |    |__|      |     |   LSTM   | --> Critic FC-> | -> value
          |    |______|   |___| | -> |__| -->  | --> |__________| --> Actor FC -> | -> policy logits
          |                     |    |__|      |        ^   ^                     | -> (hx, cx)
          |                     |    |__|      |        |   |                     |
          |_____________________|              |  prev cx  hx                     |
                                               |__________________________________|
____________________________________________________________________________________________________

    Processes an input image (with num_input_channels) with 4 conv layers,
    interspersed with 4 elu activation functions. The output of the final layer is then flattened
    and passed to an LSTM (with previous or initial hidden and cell states (hx and cx)).
    The new hidden state is used as an input to the critic and value nn.Linear layer heads,
    The final output is then the predicted value, action logits, hx and cx.
    """

    def __init__(self, num_input_channels, num_outputs, resolution):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm_cell_size = self.calculate_lstm_input_size_for_A3C(resolution)

        self.lstm = nn.LSTMCell(self.lstm_cell_size, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
                                            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
                                            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, self.lstm_cell_size)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    @staticmethod
    def calculate_lstm_input_size_for_A3C(resolution, stride=2, kernel_size=3, padding=1,
                                          num_filters=32):
        """
        Find LSTM size after 4 conv layers below in A3C using regular
        convolution math. For example:
        42x42 -> (42 − 3 + 2)÷ 2 + 1 = 21x21 after 1 layer
        11x11 after 2 layers -> 6x6 after 3 -> and finally 3x3 after 4 layers
        Therefore lstm input size after flattening would be (3 * 3 * num_filters)
        We assume that the same kernel_size, padding and stride is used in all convolutional layers
        """
        width = (resolution[0] - kernel_size + 2 * padding) // stride + 1
        width = (width - kernel_size + 2 * padding) // stride + 1
        width = (width - kernel_size + 2 * padding) // stride + 1
        width = (width - kernel_size + 2 * padding) // stride + 1

        height = (resolution[1] - kernel_size + 2 * padding) // stride + 1
        height = (height - kernel_size + 2 * padding) // stride + 1
        height = (height - kernel_size + 2 * padding) // stride + 1
        height = (height - kernel_size + 2 * padding) // stride + 1

        return width * height * num_filters


class A3C_LSTM_GA(torch.nn.Module):
    """
    ASCII charts on architecture, Check original paper and check original charts at:
    Gated-Attention Architectures for Task-Oriented Language Grounding
    https://arxiv.org/abs/1706.07230
____________________________________________________________________________________________________

                            Figure 2. Model Architecture and state processing

                    Image Processing (f_theta_image)         Image Repr.
                  ___________________________________       __
                 |  _______     _______     _______ |      |__|
      image ->   | |conv2d| -> |conv2d| -> |conv2d| | ---> |__|   ---------|
                 | |______|    |______|    |______| |      |__|            |
                 |__________________________________|      |__|            |
                                                                           |   ________     _______
              Instruction Processing (f_theta_language) Instruction Repr.  |  | multi |    |      |
                 _____________________________              __             -->| modal | -> |policy|
instruction ->  |           ________         |             |__|            |  |fusion |    |      |
word indices    |          |GRU RNN|         |        ---> |__|            |  |_______|    |______|
                |          |_______|         |             |__|   ---------|
                |____________________________|             |__|

____________________________________________________________________________________________________

                                Figure 3: Gated-Attention unit architecture.

Image Representation                          Gated-Attention Multi-modal Fusion unit
      _______
     |  ____|__                                                       _______
     | |  ____|__            ________________________                |  ____|__
     |_| |  ____|__   ----> | element-wise multiply |  --------->    | |  ____|__    to policy
       |_| |conv2d|         |_______________________|                |_| |  ____|__   ------>
         |_|      |                                                    |_| |conv2d|
           |______|                            ^                         |_|      |
                                               |                           |______|
                                               |
                                               |

Instruction Repr. -> Attention vector (a_L) -> Gated-Attention filters
  __                    __                    _______
 |__|                  |__|                  |  ____|__
 |__|     ---->        |__|    ---->         | |  ____|__
 |__|                  |__|                  |_| |  ____|__
 |__|                  |__|                    |_| |conv2d|
                                                 |_|      |
                                                   |______|

____________________________________________________________________________________________________

                            Figure 4: A3C policy model architecture

   Flattened GA fusion output        --->           Policy Learning Module    --->   Final output
                               ____________________________________________________
       __                     |     _________         ___________                 |
      |__|                    |    |FC layer|  --->  |   LSTM   | --> Critic FC-> | -> value
      |__|      ------>       | -> |________|        |__________| --> Actor FC -> | -> policy logits
      |__|                    |                         ^   ^           ^         | -> (hx, cx)
      |__|                    |                         |   |           |         |
                              |                   prev cx  hx        tx emb       |
                              |___________________________________________________|
____________________________________________________________________________________________________

    Very similar to the above ActorCritic but has Gated Attention (GA) and processes an instruction
    which is a part of the input state using a GRU. There is also a time embedding layer to help
    stabilize value prediction and only 3 conv layers compared to ActorCritic's 4 layers.
    Originally ran on the ViZDoom environment. The above ASCII art figures and commented code
    below should make the flow quite clear.

    The gated attention multi-modal fusion module enables the policy to focus on certain parts
    of the input image given the instruction e.g. for the instruction "Go to the red cup", a
    specific attention filter could be learned which would enable the agent to language ground
    itself in the meaning of both "red", "cup" and "go to". Language grounding is the ability to
    map meaning within symbols+language into real world objects and goals.
    """

    def __init__(self, num_input_channels, num_outputs, resolution, vocab_size, episode_length):
        super(A3C_LSTM_GA, self).__init__()

        self.output_width, self.output_height = self.\
            calculate_input_width_height_for_A3C_LSTM_GA(resolution)
        self.num_output_filters = 64
        self.lstm_cell_size = self.output_width * self.output_height * self.num_output_filters

        # Image Processing
        self.conv1 = nn.Conv2d(num_input_channels, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, self.num_output_filters, kernel_size=4, stride=2)

        # Natural Language Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = vocab_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)

        # Gated-Attention layers
        self.attn_linear = nn.Linear(self.gru_hidden_size, self.num_output_filters)

        # Time embedding layer, helps in stabilizing value prediction e.g. if only 1 step is left
        # in episode not much reward can be earned vs the same image state with 500 steps left
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(episode_length, self.time_emb_dim)

        # A3C-LSTM layers (extra self.time_emb_dim input to critic+actor)
        self.linear = nn.Linear(self.lstm_cell_size, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, num_outputs)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))

        # Get the instruction representation
        encoder_hidden = torch.zeros(1, 1, self.gru_hidden_size)
        for i in range(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[0, i]).view(1, 1, -1)  # 1x1x32
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

        # Get the attention vector from the instruction representation
        x_attention = F.sigmoid(self.attn_linear(x_instr_rep))

        # Gated-Attention expansion from vector into self.num_output_filters depth slices
        # so each depth slice has the same attention value spread over the entire width and height
        x_attention = x_attention.unsqueeze(2).unsqueeze(3)
        x_attention = x_attention.expand(1, self.num_output_filters, self.output_height,
                                         self.output_width)  # BCHW -> 1x64xHxW
        assert x_image_rep.size() == x_attention.size()  # BCHW must be the same for element-wise
        x = x_image_rep * x_attention  # element-wise multiplication between attention and filters
        x = x.view(x.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)

    @staticmethod
    def calculate_input_width_height_for_A3C_LSTM_GA(resolution):
        """
        Similar to the calculate_lstm_input_size_for_A3C function except that there are only
        3 conv layers and there is variation among the kernel_size, stride, the number of channels
        and there is no padding. Therefore these are hardcoded. Check A3C_LSTM_GA class for these
        numbers. Also, returns tuple representing (width, height) instead of size
        """
        width = (resolution[0] - 8) // 4 + 1
        width = (width - 4) // 2 + 1
        width = (width - 4) // 2 + 1

        height = (resolution[1] - 8) // 4 + 1
        height = (height - 4) // 2 + 1
        height = (height - 4) // 2 + 1

        return width, height
