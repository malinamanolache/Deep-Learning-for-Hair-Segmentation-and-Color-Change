# https://gist.github.com/KushajveerSingh/7773052dfb6d8adedc53d0544dedaf60
import torch
import torch.nn as nn
class VGG(nn.Module):
    """
    Standard PyTorch implementation of VGG. Pretrained imagenet model is used.
    """
    def __init__(self):
        super().__init__()
        self.skip_connections=[]
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # We need these for MaxUnpool operation
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        # self.feature_maps = OrderedDict()
        # self.pool_locs = OrderedDict()
    
    def forward(self, x):
        skip_connections=[]
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
                skip_connections.append(x)
            else:
                x = layer(x)
        
        return skip_connections


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (2,2), padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3,3), padding='same')
        self.conv3 = nn.Conv2d(out_channels, out_channels, (3,3), padding='same')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, enc):
        up = nn.Upsample((enc.shape[2], enc.shape[3]), mode='bilinear', align_corners=True)
        x = up(x)
        x = self.conv1(x)
        x = torch.cat([x, enc], dim = 1)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.batch(x)
        return x

class Unet_VGG(nn.Module):
    def __init__(self, vgg):
        super(Unet_VGG, self).__init__()
    
        #expanding path
        self.dec_5 = DecoderBlock(1024, 512)
        self.dec_4 = DecoderBlock(512, 256)
        self.dec_3 = DecoderBlock(256, 128)
        self.dec_2 = DecoderBlock(128, 64)     
        self.last_conv = nn.Conv2d(64, 1, (1,1), padding='same')
        self.conv = nn.Conv2d(512, 1024, (3, 3), padding='same')
        self.sigmoid = nn.Sigmoid()
        self.vgg=vgg

    def forward(self, x):
        skip_connections = self.vgg(x)
        out_vgg = self.conv(skip_connections[4])
        x_dec_5 = self.dec_5(out_vgg, skip_connections[3])
        x_dec_4 = self.dec_4(x_dec_5, skip_connections[2])
        x_dec_3  =self.dec_3(x_dec_4, skip_connections[1])
        x_dec_2 = self.dec_2(x_dec_3, skip_connections[0])

        self.upsample = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        x_dec_1 = self.upsample(x_dec_2)
        x_dec_1 = self.last_conv(x_dec_1)
        x_dec_1 = self.sigmoid(x_dec_1)

        return x_dec_1

