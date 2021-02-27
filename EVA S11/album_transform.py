import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # A.CropAndPad(4),
        A.RandomCrop(32, 32),
        A.HorizontalFlip(),
        A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, always_apply=False),
        ToTensorV2()
    ]
)

test_transform = A.Compose([A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),ToTensorV2()])