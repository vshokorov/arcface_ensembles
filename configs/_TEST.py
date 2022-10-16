from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r18"
config.resume = False
config.output = None
config.embedding_size = 256
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.1
config.verbose = 2000
config.dali = False

config.rec = "../dataset_list.txt"
config.num_classes = 569
config.num_image = 16380
config.num_workers = 4
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = []
