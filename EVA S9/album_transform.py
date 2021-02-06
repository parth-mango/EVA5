import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose(
    [
        A.HorizontalFlip(),
        A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, always_apply=False),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        A.RandomCrop(26, 26),
        ToTensorV2()
    ]
)

test_transform = A.Compose([A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),ToTensorV2()])