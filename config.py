from pathlib import Path


class Configs(object):
    def __init__(self, gpus_num, model_type="Large"):
        super(Configs, self).__init__()
        # data configs
        self.rgbd_dirs = Path("RGBD_Datasets")  # RGBD data path
        self.hole_dirs = Path("Hole_Datasets")  # Hole data path
        self.save_dir = Path("checkpoints")
        self.checkpoint = None  # checkpoint path

        # dataloader configs
        self.sizes = 320  # sizes of images during training

        # optimizer setting
        self.lr = 2e-4  # learning rate
        self.wd = 0.05  # weight decay
        self.epochs = 300  # epochs numbers
        self.warmup_epochs = int(self.epochs * 0.05)  # warmup epochs
        self.batch_size = 64 // gpus_num  # batch sizes

        # multi GPU and AMP
        self.num_workers = 4  # the number of workers
        self.amp = True  # automatic mixed precision (AMP)

        # feedback
        self.feedback_iteration = 1000
        self.checkpoint_epoch = 20

        # network configs
        self.norm_type = "CNX"  # normalization type
        if model_type == "Tiny":
            self.dims = [96, 192, 384, 768]  # dimensions
            self.depths = [3, 3, 9, 3]  # block number
            self.dp_rate = 0.0
        if model_type == "Small":
            self.dims = [96, 192, 384, 768]  # dimensions
            self.depths = [3, 3, 27, 3]  # block number
            self.dp_rate = 0.1
        elif model_type == "Base":
            self.dims = [128, 256, 512, 1024]  # dimensions
            self.depths = [3, 3, 27, 3]  # block number
            self.dp_rate = 0.1
        elif model_type == "Large":
            self.dims = [192, 384, 768, 1536]  # dimensions
            self.depths = [3, 3, 27, 3]  # block number
            self.dp_rate = 0.2
