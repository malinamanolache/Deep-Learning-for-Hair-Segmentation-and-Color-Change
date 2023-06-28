import torch
import torch.nn as nn
 
class EncoderBlock(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(EncoderBlock, self).__init__()
       self.conv1 = nn.Conv2d(in_channels, out_channels, (3,3), padding='same')
       self.conv2 = nn.Conv2d(out_channels, out_channels, (3,3), padding='same')
       self.relu = nn.ReLU()
       self.maxpool = nn.MaxPool2d(2, stride=2)
       self.dropout = nn.Dropout(0.3)
       self.batch = nn.BatchNorm2d(out_channels)
 
   def forward(self, x):
       x = self.conv1(x)
       x = self.relu(x)
       x = self.conv2(x)
       x = self.relu(x)
       x = self.maxpool(x)
       x = self.dropout(x)
       x = self.batch(x)
       return x
 
 
class DecoderBlock(nn.Module):
   def __init__(self, in_channels, out_channels):
       super(DecoderBlock, self).__init__()
       self.conv1 = nn.Conv2d(in_channels, out_channels, (2,2), padding='same')
       self.conv2 = nn.Conv2d(in_channels, out_channels, (3,3), padding='same')
       self.conv3 = nn.Conv2d(out_channels, out_channels, (3,3), padding='same')
       self.relu = nn.ReLU()
       self.dropout = nn.Dropout(0.3)
       self.batch = nn.BatchNorm2d(out_channels)
 
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
 
class Unet_augmentation(nn.Module):
   def __init__(self):
       super(Unet_augmentation, self).__init__()
       #contracting path
       self.enc_1 = EncoderBlock(3, 64)
       self.enc_2 = EncoderBlock(64, 128)
       self.enc_3 = EncoderBlock(128, 256)
       self.enc_4 = EncoderBlock(256, 512)
       self.enc_5 = EncoderBlock(512, 1024)
       #expanding path
       self.dec_5 = DecoderBlock(1024, 512)
       self.dec_4 = DecoderBlock(512, 256)
       self.dec_3 = DecoderBlock(256, 128)
       self.dec_2 = DecoderBlock(128, 64)    
       self.last_conv = nn.Conv2d(64, 1, (1,1), padding='same')
       self.sigmoid = nn.Sigmoid()
 
   def forward(self, x):
       x_enc_1 = self.enc_1(x)
       x_enc_2 = self.enc_2(x_enc_1)
       x_enc_3 = self.enc_3(x_enc_2)
       x_enc_4 = self.enc_4(x_enc_3)
       x_enc_5 = self.enc_5(x_enc_4)
       x_dec_5 = self.dec_5(x_enc_5, x_enc_4)
 
       x_dec_4 = self.dec_4(x_dec_5, x_enc_3)
       x_dec_3  =self.dec_3(x_dec_4, x_enc_2)
       x_dec_2 = self.dec_2(x_dec_3, x_enc_1)
       self.upsample = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
       x_dec_1 = self.upsample(x_dec_2)
       x_dec_1 = self.last_conv(x_dec_1)
       x_dec_1 = self.sigmoid(x_dec_1)
 
       return x_dec_1
