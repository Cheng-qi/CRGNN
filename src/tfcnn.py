import torch 
import torch.nn as nn  

class CNN(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_size, dropout, image_size, **kward):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.interactive_fliters = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel[0], (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(hidden_channel[0], hidden_channel[1], (5, 5)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            # nn.Dropout(dropout)
        )
        with torch.no_grad():
            test_input = torch.rand(1, in_channel, *image_size)
            out_dims = self.interactive_fliters(test_input).view(1,-1).shape
        self.output_layer = nn.Sequential(
            nn.Linear(out_dims[-1], out_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        log = {}
        # input = input.suqeeze
        input = input.unsqueeze(1)
        x_base = self.interactive_fliters(input).view(input.shape[0], -1)
        # merge = x_base.view(input.shape[0], -1)
        out = self.output_layer(x_base)
        return out, log



class TFCNN(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_size, dropout, image_size, **kward):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.time_fliters = nn.Sequential(
            nn.LayerNorm([in_channel, *image_size]),
            nn.Conv2d(in_channel, hidden_channel, (2, 6), 2),
            nn.ReLU(),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU(),
            nn.MaxPool2d((2,2)),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU(),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, (3, 3), 1),
            nn.ReLU(),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, (3, 3), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU()
        )
        self.interactive_fliters = nn.Sequential(
            nn.LayerNorm([in_channel, *image_size]),
            nn.Conv2d(in_channel, hidden_channel, (5, 5), 1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(hidden_channel, hidden_channel, (3, 3), 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(dropout)
        )
        self.frequency_fliters = nn.Sequential(
            nn.LayerNorm([in_channel, *image_size]),
            nn.Conv2d(in_channel, hidden_channel, (6, 2), 2),
            nn.ReLU(),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(hidden_channel, hidden_channel, (3, 3), 1),
            nn.ReLU(),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU(),
            nn.Conv2d(hidden_channel, hidden_channel, (3, 3), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(hidden_channel, hidden_channel, (1, 1)),
            # nn.ReLU()
        )
        with torch.no_grad():
            test_input = torch.rand(1, in_channel, *image_size)
            out_dims = torch.cat([self.time_fliters(test_input).view(1,-1,hidden_channel),
                                self.frequency_fliters(test_input).view(1,-1,hidden_channel),
                                self.interactive_fliters(test_input).view(1,-1,hidden_channel)
                                ], dim=-2).shape
        
        # self.MLP = nn.Linear(1335, )

        self.output_layer = nn.Sequential(
            # nn.Linear(hidden_channel * hidden_channel * 3 * 4, 1024),
            nn.Linear(out_dims[-1]*out_dims[-2], out_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        log = {}
        # input = input.suqeeze
        input = input.unsqueeze(1)
        x_time = self.time_fliters(input).view(input.shape[0], self.hidden_channel,-1)
        x_base = self.interactive_fliters(input).view(input.shape[0], self.hidden_channel, -1)
        x_frequency = self.frequency_fliters(input).view(input.shape[0], self.hidden_channel, -1)
        merge = torch.cat([x_time, x_frequency, x_base], dim=-1)
        
        merge = merge.view(input.shape[0], -1)
        out = self.output_layer(merge)
        return out, log

