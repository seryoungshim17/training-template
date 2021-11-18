class Trainer():
    """
    Trainer class
    method
        __init__
        _train : train model
        _valid : validate model
    """
    def __init__(self, cfg):
        pass

    def _train(self):
        pass

    def _valid(self):
        pass

    def _progress(self, batch_idx):
        """
        args
            batch_idx: batch index
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)