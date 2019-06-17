from layers import *

class Projection(nn.Module):
    def __init__(self, out_dims):
        super(Projection, self).__init__()
        """
           Projection matrix on embedding, implemented as a 1d convolutional layer
           N x C_p x B Tensor, N is batch size, B is the dim of embedding vector 
        """
        self.projection = conv1d(in_channels = 1, out_channels = out_dims, 
                       bias = False, kernel_size = 1, stride = 1, padding = 0, 
                       spectral = True, init_zero_weights = False, activation = None,
                       pool = None,norm = None) 
        # N x out_dims x B
    def forward(self,embedding):
        return self.projection(embedding)

class GeneratorV1(torch.nn.Module):
    def __init__(self):
        super(GeneratorV1,self).__init__()
        """
        V1 of Generator, with adaptive instance norm applied on upsampling and middle residual layers,
        with same number of channels
        Inputs:
            y_i_t = landmark image of time step t of sequence i
            e_i = embedding
        """
        # embedding shape N x 1 x B
        self.projection = Projection(128)
        # projection N x 64 x B
        # ***downsampling part*** #
        # N x 3 x H x W landmark image (224x224)
        self.downres1 = ResDownIn(3,[32,32],2) # N x 16 x H // 2 x W // 2 (112x112)
        self.downres2 = ResDownIn(32,[64,64],2) # N x 32 x H // 4 x W // 4 (56x56)
        self.downres3 = ResDownIn(64,[64,64],2) # N x 64 x H // 8 x W // 8 (28x28)
        self.attn1 = SelfAttn(64) # N x 64 x H // 8 x W // 8 (28 x 28)
        self.downres4 = ResDownIn(64,[64,64],2) # N x 64 x H // 16 x W // 16 (14x14)
        # ***middle Res block (w / adaptive instance norm)*** #
        self.res1 = ResAdaIn(64,64)
        self.res2 = ResAdaIn(64,64)
        self.res3 = ResAdaIn(64,64)
        self.res4 = ResAdaIn(64,64)
        self.res5 = ResAdaIn(64,64)
        # ***upsampling part*** #
        self.upres1 = ResUpAdaIn(64,[64,64],2) # N x 64 x H // 8 x W // 8 (28x28)
        self.upres2 = ResUpAdaIn(64,[64,64],2) # N x 64 x H // 4 x W // 4 (56x56)
        self.attn2 = SelfAttn(64) # N x 64 x H // 8 x W // 8 (56x56)
        self.upres3 = ResUpAdaIn(64,[64,64],2) # N x 64 x H // 2 x W // 2 (112x112)
        self.upres4 = ResUpAdaIn(64,[64,64],2) # N x 64 x H x W (224x224)
        self.conv = conv2d(in_channels = 64, out_channels = 3, 
                       bias = True, kernel_size = 1, stride = 1, padding = 0, 
                       spectral = False, init_zero_weights = False,activation = nn.Tanh(),
                       pool = None,norm = None) # N x 3 x H x W
    
    def forward(self,y,embedding):
        projection = self.projection(embedding)
        # down
        down = self.downres1(y)
        down = self.downres2(down)
        down = self.downres3(down)
        down = self.attn1(down)
        down = self.downres4(down)
        # middle
        mid = self.res1(down,projection)
        mid = self.res2(mid,projection)
        mid = self.res3(mid,projection)
        mid = self.res4(mid,projection)
        mid = self.res5(mid,projection)
        # up
        up = self.upres1(mid,projection)
        up = self.upres2(up,projection)
        up = self.attn2(up)
        up = self.upres3(up,projection)
        up = self.upres4(up,projection)
        out = self.conv(up)
        return out

class GeneratorV2(torch.nn.Module):
    def __init__(self):
        super(GeneratorV2,self).__init__()
        """
        V2 of Generator, with adaptive instance norm applied on upsampling and middle residual layers, 
        but with varying number of channels
        Inputs:
            y_i_t = landmark image of time step t of sequence i
            e_i = embedding
        """
        # embedding shape N x 1 x B
        self.projection1 = Projection(128)
        self.projection2 = Projection(64)
        self.projection3 = Projection(64)
        self.projection4 = Projection(32)
        self.projection5 = Projection(3)
        # projection N x out_dims x B
        # ***downsampling part*** #
        # N x 3 x H x W landmark image (224x224)
        self.downres1 = ResDownIn(3,[32,32],2) # N x 16 x H // 2 x W // 2 (112x112)
        self.downres2 = ResDownIn(32,[64,64],2) # N x 32 x H // 4 x W // 4 (56x56)
        self.downres3 = ResDownIn(64,[64,64],2) # N x 64 x H // 8 x W // 8 (28x28)
        self.attn1 = SelfAttn(64) # N x 64 x H // 8 x W // 8 (28 x 28)
        self.downres4 = ResDownIn(64,[128,128],2) # N x 128 x H // 16 x W // 16 (14x14)
        # ***middle Res block (w / adaptive instance norm)*** #
        self.res1 = ResAdaIn(128,128)
        self.res2 = ResAdaIn(128,128)
        self.res3 = ResAdaIn(128,128)
        self.res4 = ResAdaIn(128,128)
        self.res5 = ResAdaIn(128,128)
        # ***upsampling part*** #
        self.upres1 = ResUpAdaIn(128,[64,64],2) # N x 64 x H // 8 x W // 8 (28x28)
        self.upres2 = ResUpAdaIn(64,[64,64],2) # N x 64 x H // 4 x W // 4 (56x56)
        self.attn2 = SelfAttn(64) # N x 64 x H // 4 x W // 4 (56x56)
        self.upres3 = ResUpAdaIn(64,[32,32],2) # N x 32 x H // 2 x W // 2 (112x112)
        self.upres4 = ResUpAdaIn(32,[3,3],2) # N x 3 x H x W (224x224)

    def forward(self,y,embedding):
        projection1 = self.projection1(embedding)
        projection2 = self.projection2(embedding)
        projection3 = self.projection3(embedding)
        projection4 = self.projection4(embedding)
        projection5 = self.projection5(embedding)
        
        # down
        down = self.downres1(y)
        down = self.downres2(down)
        down = self.downres3(down)
        down = self.attn1(down)
        down = self.downres4(down)
        # middle
        mid = self.res1(down,projection1)
        mid = self.res2(mid,projection1)
        mid = self.res3(mid,projection1)
        mid = self.res4(mid,projection1)
        mid = self.res5(mid,projection1)
        # up
        up = self.upres1(mid,projection2)
        up = self.upres2(up,projection3)
        up = self.attn2(up)
        up = self.upres3(up,projection4)
        out = self.upres4(up,projection5)
        return out

class GeneratorV3(torch.nn.Module):
    def __init__(self,projection_dims=128):
        super(GeneratorV3,self).__init__()
        """
        V3 of Generator, with adaptive instance norm applied on upsampling layers,
        with linear in residual
        Inputs:
            y_i_t = landmark image of time step t of sequence i
            e_i = embedding
        """
        # embedding shape N x 1 x B
        self.projection = Projection(projection_dims)
        # projection N x projection_dims x B
        # ***downsampling part*** #
        # N x 3 x H x W landmark image (224x224)
        self.downres1 = ResDownIn(3,[32,32],2) # N x 16 x H // 2 x W // 2 (112x112)
        self.downres2 = ResDownIn(32,[64,64],2) # N x 32 x H // 4 x W // 4 (56x56)
        self.downres3 = ResDownIn(64,[128,128],2) # N x 64 x H // 8 x W // 8 (28x28)
        self.attn1 = SelfAttn(128) # N x 64 x H // 8 x W // 8 (28 x 28)
        self.downres4 = ResDownIn(128,[256,256],2) # N x 64 x H // 16 x W // 16 (14x14)
        # ***middle Res block (w / adaptive instance norm)*** #
        self.res1 = ResIn(256,256)
        self.res2 = ResIn(256,256)
        self.res3 = ResIn(256,256)
        self.res4 = ResIn(256,256)
        self.res5 = ResIn(256,256)
        # ***upsampling part*** #
        self.upres1 = ResUpAdaInV2(256,[128,128],projection_dims,2) # N x 128 x H // 8 x W // 8 (28x28)
        self.upres2 = ResUpAdaInV2(128,[64,64],projection_dims,2) # N x 64 x H // 4 x W // 4 (56x56)
        self.attn2 = SelfAttn(64) # N x 64 x H // 8 x W // 8 (56x56)
        self.upres3 = ResUpAdaInV2(64,[32,32],projection_dims,2) # N x 32 x H // 2 x W // 2 (112x112)
        self.upres4 = ResUpAdaInV2(32,[3,3],projection_dims,2) # N x 3 x H x W (224x224)
    
    def forward(self,y,embedding):
        projection = self.projection(embedding)
        # down
        down = self.downres1(y)
        down = self.downres2(down)
        down = self.downres3(down)
        down = self.attn1(down)
        down = self.downres4(down)
        # middle
        mid = self.res1(down)
        mid = self.res2(mid)
        mid = self.res3(mid)
        mid = self.res4(mid)
        mid = self.res5(mid)
        # up
        up = self.upres1(mid,projection)
        up = self.upres2(up,projection)
        up = self.attn2(up)
        up = self.upres3(up,projection)
        out = self.upres4(up,projection)
        return out

    def forward_sequence(self,ys,embedding):
        sequence = []
        for y in ys:
            sequence.append(self.forward(y,embedding))
        return sequence

class Embedder(torch.nn.Module):
    def __init__(self,embedding_dims=128):
        super(Embedder,self).__init__()
        '''
        Inputs:
            x_i_s = video frame at time step s of sequence i (3x128x128)
            y_i_s = landmark image at time step s of sequence i (3x128x128)
            stacked
        '''
        #### Image frame ####
        self.embedding_dims = embedding_dims
        # N x 6 x H x W landmark image + video frame image (224x224)
        self.downres1 = ResDown(6,[32,32],2) # N x 32 x H // 2 x W // 2 (112x112)
        self.downres2 = ResDown(32,[64,64],2) # N x 32 x H // 4 x W // 4 (56x56)
        self.downres3 = ResDown(64,[64,64],2) # N x 64 x H // 8 x W // 8 (28x28)
        self.attn1 = SelfAttn(64) # N x 64 x H // 8 x W // 8 (28 x 28)
        self.downres4 = ResDown(64,[self.embedding_dims,self.embedding_dims],2) # N x 128 x H // 16 x W // 16 (14x14)
        # *** global sum pooling ***
        self.relu = nn.ReLU() # N x 128 
        
    def forward(self,x,y):
        cat = torch.cat((x,y),1)
        down = self.downres1(cat)
        down = self.downres2(down)
        down = self.downres3(down)
        down =  self.attn1(down)
        down = self.downres4(down)
        summed = down.view(down.size(0),down.size(1),-1).sum(-1) # down.size(0) x down.size(1)
        summed = summed.view(down.size(0),1,down.size(1)) # down.size(0) x 1 x down.size(1)
        return self.relu(summed)
    
    def average_embeddings(self,sampled_data,w_out_grad=False):
        scale = 1/len(sampled_data)
        out = []
        for x , y in sampled_data:
            if w_out_grad:
                e = self.forward(x,y).detach()
            else:
                e = self.forward(x,y)
            out.append(e)
        out = scale*torch.stack(out).mean(0)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self,num_video_sequences,projection_dims=128):
        super(Discriminator,self).__init__()
        '''
        Inputs:
            x_i_t = video frame at time step t of sequence i (3x128x128)
            y_i_t = landmark image at time step t of sequence i (3x128x128)
            stacked
        '''
        ## W matrix ##
        self.W = nn.Parameter(torch.rand(projection_dims, num_video_sequences).normal_(0.0, 0.01))
        self.w_0 = nn.Parameter(torch.rand(projection_dims, 1).normal_(0.0, 0.01))
        self.b = nn.Parameter(torch.rand(1).normal_(0.0, 0.01))
        #### Image frame ####
        # N x 6 x H x W landmark image + video frame image (224x224)
        self.downres1 = ResDown(6,[32,32],2) # N x 32 x H // 2 x W // 2 (112x112)
        self.downres2 = ResDown(32,[64,64],2) # N x 32 x H // 4 x W // 4 (56x56)
        self.downres3 = ResDown(64,[64,64],2) # N x 64 x H // 8 x W // 8 (28x28)
        self.attn1 = SelfAttn(64) # N x 64 x H // 8 x W // 8 (28 x 28)
        self.downres4 = ResDown(64,[128,128],2) # N x 128 x H // 16 x W // 16 (14x14)
        #additional res
        self.downres5 = ResDown(128,[projection_dims,projection_dims],2) # N x 128 x H // 32 x W // 32 (7x7)
        # *** global sum pooling ***
        self.relu = nn.ReLU() # N x 128 
        self.tanh = nn.Tanh()
        
    def forward(self,x,y,index):
        cat = torch.cat((x,y),1)
        activations = []
        down = self.downres1(cat)
        activations.append(down)
        down = self.downres2(down)
        activations.append(down)
        down = self.downres3(down)
        activations.append(down)
        down =  self.attn1(down)
        activations.append(down)
        down = self.downres4(down)
        activations.append(down)
        down = self.downres5(down)
        activations.append(down)
        summed = down.view(down.size(0),down.size(1),-1).sum(-1) # down.size(0) x down.size(1)
        summed = summed.view(down.size(0),down.size(1)) # down.size(0) x down.size(1)
        spliced = self.W[:, index].view(-1, 1)
        sum_spliced = spliced + self.w_0
        out = self.tanh(torch.mm(summed, sum_spliced) + self.b)
        return out, activations, spliced        

# if __name__ == "__main__":
# 	g = GeneratorV3(128)
