from easydict import EasyDict
opt = EasyDict()

# Report acc on all domains
opt.val_domain = 2
opt.test_on_all_dmn = True

opt.use_visdom = False
opt.visdom_port = 2000

opt.device = "cuda"
opt.seed = 1

# Learning
opt.lr_e = 3e-5
opt.groupdro_eta = 0.1 # DRO's eta hyper-parameter
opt.lmbda = 0.1 # regularizer
opt.beta1 = 0.9
opt.no_bn = False  # do not use batch normalization
opt.threshold = 40

# model size configs, used for E, F
opt.nx = 2  # dimension of the input data
opt.nh = 512  # dimension of hidden # 512
opt.nc = 2  # number of label class

opt.test_interval = 20
opt.save_interval = 100