import test_dataset
import torch
import os
import numpy as np
from io_utils import parse_args_eposide_train
import ResNet10
import ProtoNet
import torch.nn as nn
import torch.nn.functional as F  # Import the functional module
from torch.autograd import Variable
import utils
import random
import copy
import warnings
import tqdm
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore", category=Warning)

# Define the distillation loss function
class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def train(novel_loader, model, teacher_model, optimizer, params):
    iter_num = len(novel_loader) 
    acc_all_LR = []
    model.train()
    top1 = utils.AverageMeter()
    total_loss = 0
    softmax = torch.nn.Softmax(dim=1)
    for i, (x,_) in enumerate(novel_loader):
        # divergence loss# Calculate raw predictions of the models
        x_query = img[:, params.n_support:,:,:,:].contiguous().view(params.n_way*params.n_query, *x.size()[2:]).cuda() 
        x_support = img[:,:params.n_support,:,:,:].contiguous().view(params.n_way*params.n_support, *x.size()[2:]).cuda() # (25, 3, 224, 224)
        
        with torch.no_grad():
            out_support_teacher = teacher_model(x_support)
            out_query_teacher = teacher_model(x_query)
            out_support_student = model(x_support)
            out_query_student = model(x_query)
        
        # Softmax outputs of the models (using the same temperature)
        T = 5.0  # Set your desired temperature
        p_support_teacher = F.softmax(out_support_teacher / T, dim=1)
        p_query_teacher = F.softmax(out_query_teacher / T, dim=1)
        p_support_student = F.softmax(out_support_student / T, dim=1)
        p_query_student = F.softmax(out_query_student / T, dim=1)
        
        # Calculate KL divergence loss
        loss_fn = DistillKL(T)
        loss_div = loss_fn(p_support_student, p_support_teacher) + loss_fn(p_query_student, p_query_teacher)

    

        loss = loss_div

        _, predicted = torch.max(pred_query_set[0].data, 1)
        correct = predicted.eq(query_set_y.data).cpu().sum()
        top1.update(correct.item()*100 / (query_set_y.size(0)+0.0), query_set_y.size(0))  
        
        loss.backward()
        scheduler.step()
        optimizer.step()

        total_loss = total_loss + loss.item()
    avg_loss = total_loss/float(i+1)
    return avg_loss, top1.avg




    
if __name__=='__main__':
    
    params = parse_args_eposide_train()

    setup_seed(params.seed)

    datamgr = test_dataset.Eposide_DataManager(data_path=params.source_data_path, 
                                               num_class=params.train_num_class, 
                                               image_size=params.image_size,
                                               n_way=params.n_way, 
                                               n_support=params.n_support, 
                                               n_query=params.n_query, 
                                               n_eposide=params.train_n_eposide)
    
    novel_loader = datamgr.get_data_loader() 

    model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims, 
                            list_of_stride=params.list_of_stride, 
                            list_of_dilated_rate=params.list_of_dilated_rate)

    
    if not os.path.isdir(params.save_dir):
        os.makedirs(params.save_dir)

    tmp = torch.load(params.pretrain_model_path)
    state = tmp['state']
    model.load_state_dict(state)
    model = model.cuda()    

    optimizer = torch.optim.Adam([{"params":model.parameters()}], 
                                 lr=params.lr)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Load the teacher model
    teacher_checkpoint = torch.load('./100.tar')  # Adjust the path as needed
    teacher_model = ResNet10.ResNet(list_of_out_dims=params.list_of_out_dims,
                                    list_of_stride=params.list_of_stride,
                                    list_of_dilated_rate=params.list_of_dilated_rate)
    teacher_model.load_state_dict(teacher_checkpoint['state_model'])
    teacher_model = teacher_model.cuda()
    teacher_model.eval()


    for epoch in range(params.epoch):
        train_loss, train_acc = train(novel_loader, 
                                      model,
                                      teacher_model,
                                      # Siamese_model, 
                                      # head, 
                                      # loss_fn_ce, 
                                      optimizer, 
                                      params)
        print('train:', epoch + 1, 'current epoch train loss:', train_loss, 'current epoch train acc:', train_acc)
        outfile = os.path.join(params.save_dir, '{:d}_student.tar'.format(epoch + 1))
        torch.save({
            'epoch': epoch + 1,
            'state_model': model.state_dict(),
            'state_Siamese_model': Siamese_model.state_dict()},
            outfile)
    
