import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import os
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1


filepath = "/home/stu4/project/lightgcn/data/meld_word/1/result/" + str(world.dataset) +'/'+ str(Recmodel.n_layers) + "layers"
filename = filepath + str(Recmodel.n_layers)+"_decay_"+str(world.config['decay'])+"_lr_"+str(world.config['lr'])+".txt"


# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    file_path = "/home/stu4/project/lightgcn/data/meld_word/1/result/" + world.dataset + "/layer" + str(Recmodel.n_layers)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    embedding_file = file_path + "/word_embedding_latter.pkl"
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, filename , epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
    Recmodel.saveEmbedding(embedding_file)
    print(Recmodel)
finally:
    if world.tensorboard:
        w.close()