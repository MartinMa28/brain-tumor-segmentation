# Classes for classification: HGG, LGG
# Classes for segmentation: 0 - background, 1 - necrotic tumor, 2 - edema, 3 - enhancing tumor
n_classes = 2
batch_size = 8
epochs = 15
lr = 1e-3
#momentum = 0
w_decay = 1e-5
step_size = 10
gamma = 1.
configs = "UNet-BRATS2018_batch{}_training_epochs{}_Adam_scheduler-step{}-gamma{}_lr{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, w_decay)
pre_trained_path = '/home/martin/Documents/MartinMa/semantic_segmentation/brain_tumor_segmentation/scores/UNet-BRATS2018_batch3_training_epochs10_Adam_scheduler-step10-gamma1.0_lr0.001_w_decay1e-05/terminated_model.tar'