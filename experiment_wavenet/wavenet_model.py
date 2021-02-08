from torch import nn


class WaveNetModel(nn.Module):
    def __init__(
            self,
            feature_length,
            layers=10,
            blocks=4,
            dilation_channels=32,
            residual_channels=32,
            skip_channels=256,
            end_channels=256,
            kernel_size=2,
            bias=False
    ):
        super(WaveNetModel, self).__init__()

        self.feature_length = feature_length
        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size

        receptive_field = 1
        init_dilation = 1

        self.dilations = []

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.start_conv = nn.Conv1d(
            in_channels=feature_length,
            out_channels=residual_channels,
            kernel_siz=1,
            bias=bias
        )

        for b in range(blocks):
            new_dilation = 1

            for i in range(layers):
                self.dilations.append((new_dilation, init_dilation))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=1,
                                    bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
                                    out_channels=1,
                                    kernel_size=1,
                                    bias=True)
