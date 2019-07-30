import torch

def getMoments(tens):
    tens_size = tens.size()
    batch_size = tens_size[0]
    num_channels = tens_size[1]
    tens = tens.view(batch_size,num_channels,-1)
    mu = tens.mean(-1,keepdim=False).view(batch_size,num_channels,1,1)
    std = tens.std(-1,unbiased=False).view(batch_size,num_channels,1,1)
    return mu, std

def adaIn(content,style,eps=1e-6):
    content = content.float()
    style = style.float()
    content_mu,content_std = getMoments(content)
    style_mu,style_std = getMoments(style)
    norm = ((content - content_mu)/ (content_std + eps)) #stability
    norm_style = style_std*(norm) + style_mu
    return norm_style.view_as(content)

def adaInV2(content,style_mu,style_std,eps=1e-6):
    content = content.float()
    content_mu,content_std = getMoments(content)
    norm = ((content - content_mu)/ (content_std + eps)) #stability
    norm_style = style_std*(norm) + style_mu
    return norm_style.view_as(content)


def convOutput(in_size, kernel_size, stride, padding = 0):
    output = int((in_size - kernel_size + 2*(padding)) / stride + 1)
    return output 

def deconvOutput(in_size, kernel_size, stride, padding = 0):
    output = int(stride*(in_size-1) + kernel_size - (2*padding))
    return output
    
def getPaddingConv(in_size,output_size,kernel_size,stride,):
    padding = int((stride*(output_size - 1) + kernel_size - in_size) / 2)
    return padding

def getPaddingDeConv(in_size,output_size,kernel_size,stride):
    padding = int((stride*(in_size - 1) + kernel_size - output_size) / 2)
    return np.max((padding,0))

def getStrideConv(in_size,output_size,kernel_size,padding):
    stride = int(((2*padding) - kernel_size + in_size) / (output_size - 1))
    return stride

def getStrideDeConv(in_size,output_size,kernel_size,padding):
    stride = int(((2*padding) - kernel_size + output_size) / (in_size - 1))
    return np.max((stride,1))

