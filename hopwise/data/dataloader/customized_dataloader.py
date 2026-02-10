from hopwise.data.dataloader.general_dataloader import TrainDataLoader


class SemanticMFTrainDataloader(TrainDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle)
        dataset.train_ann()
