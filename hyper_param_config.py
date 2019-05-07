# Classes for classification: HGG, LGG
# Classes for segmentation: 0 - background, 1 - necrotic tumor, 2 - edema, 3 - enhancing tumor
n_classes = 4
batch_size = 2
epochs = 10
lr = 1e-4
#momentum = 0
w_decay = 1e-5
step_size = 10
gamma = 1.
configs = "UNet-BRATS2018_batch{}_training_epochs{}_Adam_scheduler-step{}-gamma{}_lr{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, w_decay)
pre_trained_path = 'scores/UNet-BRATS2018_batch2_training_epochs10_Adam_scheduler-step10-gamma1.0_lr0.0001_w_decay1e-05/terminated_model.tar'