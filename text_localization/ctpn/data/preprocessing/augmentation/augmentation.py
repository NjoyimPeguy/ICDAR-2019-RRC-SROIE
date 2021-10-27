from text_localization.ctpn.data.preprocessing.transformations import SplitBoxes, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomZoomOut, RandomZoomIn, Resize


class BasicDataTransformation(object):
    def __init__(self, configs: dict):
        self.image_size = configs.IMAGE.SIZE
        self.pixel_std = configs.IMAGE.PIXEL_STD
        self.pixel_mean = configs.IMAGE.PIXEL_MEAN
        self.augment = self.get_transformation()
    
    def get_transformation(self):
        return Compose([
            Resize(image_size=self.image_size),
            ToTensor(),
            Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])
    
    def __call__(self, image, boxes):
        return self.augment(image, boxes)


class TrainDataTransformation(BasicDataTransformation):
    def __init__(self, configs):
        self.anchor_scale = configs.ANCHOR.SCALE
        
        super(TrainDataTransformation, self).__init__(configs)
    
    def get_transformation(self):
        return Compose([
            RandomHorizontalFlip(),
            RandomZoomOut(bg_color=self.pixel_mean),
            RandomZoomIn(),
            Resize(image_size=self.image_size, resize_bboxes=True),
            SplitBoxes(anchor_scale=self.anchor_scale),
            ToTensor(),
            Normalize(mean=self.pixel_mean, std=self.pixel_std)
        ])
