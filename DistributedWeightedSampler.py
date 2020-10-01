from torch.utils.data import Sampler

class DistributedWeightedSampler(Sampler):
    """Combination of distributed sampler and weighted sampler

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

    """

    def __init__(self, dataset, weights, replacement=True, num_replicas=None, rank=None, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replacement = replacement
   
    def __iter__(self):
        rand_indices = torch.multinomial(self.weights, self.total_size, self.replacement).tolist()
        # add random extra samples to make it evenly divisible if need be
        extra_samples = sample(rand_indices, (self.total_size - len(rand_indices)))
        rand_indices += extra_samples
        assert len(rand_indices) == self.total_size

        # subsample every nth item based on the rank (such that the sampler will return a different set for each process)
        rand_indices = rand_indices[self.rank:self.total_size:self.num_replicas]
        assert len(rand_indices) == self.num_samples
       
        return iter(rand_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
