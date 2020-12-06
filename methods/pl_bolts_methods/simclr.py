"""
TODO: Add the SimCLR implementation from pl_bolts
"""
# from pl_bolts.models.self_supervised import SimCLR
# from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform

# train_dataset = MyDataset(transforms=SimCLRTrainDataTransform())
# val_dataset = MyDataset(transforms=SimCLREvalDataTransform())

# # simclr needs a lot of compute!
# model = SimCLR()
# trainer = Trainer(tpu_cores=128)
# trainer.fit(
#     model,
#     DataLoader(train_dataset),
#     DataLoader(val_dataset),
# )