# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from contextlib import contextmanager
import collections
import functools
import json
import logging
import math
import numpy as np
import os
import csv
import pickle
import random
import torch
from torch.utils.data import DataLoader, Sampler

import adaptdl.checkpoint
import adaptdl.collective
import adaptdl.env
from adaptdl.torch.epoch import current_epoch
from adaptdl.torch._metrics import (
    profile_step_start, profile_step_commit,
    set_batch_size, get_goodput_fn, get_progress, seed_sched_hints)
from adaptdl._signal import get_exit_flag

from adaptdl.torch._metrics import _metrics_state

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class ElasticSampler(Sampler):
    """
    A PyTorch Sampler which partitions data samples across multiple replicas,
    and supports deterministic continuing across checkpoint-restarts. Shuffling
    is deterministic for each epoch, and :meth:`ElasticSampler.set_epoch`
    should be invoked to obtain different orderings in different epochs.

    Arguments:
        dataset (torch.util.data.Dataset): The dataset to sample from.
        shuffle (bool): Whether the data samples should be shuffled.

    .. automethod:: __iter__
    .. automethod:: __len__
    """
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = adaptdl.env.num_replicas()
        self.rank = adaptdl.env.replica_rank()
        self.epoch = 0
        self.index = 0

    def __iter__(self):
        """
        Iterate through the samples in the dataset, in the order defined for a
        set epoch, starting at a set index. Produces only the indices for the
        local replica.

        Returns: Iterator over data sample indices.
        """
        if self.shuffle:
            # Deterministically shuffle based on epoch.
            g = torch.Generator()
            g.manual_seed(hash((self.epoch, self.index // len(self.dataset))))
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        base_index = self.index % len(self.dataset)

        # Subsample.
        local_indices = indices[base_index + self.rank::self.num_replicas]

        # Add extra samples to make it evenly divisible.
        if len(local_indices) < len(self):
            local_indices.append(indices[self.rank])
        assert len(local_indices) == len(self)
        return iter(local_indices)

    def __len__(self):
        """
        The total number of samples to be iterated through, starting at the set
        index, for the local replica.

        Returns (int): Number of samples.
        """
        base_index = self.index % len(self.dataset)
        return math.ceil((len(self.dataset) - base_index) / self.num_replicas)

    def set_epoch(self, epoch, index=0):
        """
        Set the epoch to derive samples from. Optional argument ``index`` can
        be specified to start sampling from a particular index, e.g. after a
        checkpoint-restart.

        Arguments:
            epoch (int): The epoch to sample from.
            index (int): The index to start sampling from.
        """
        self.epoch = epoch
        self.index = index


def current_dataloader():
    """
    Reference to the data loader currently being iterated.

    Returns (AdaptiveDataLoaderHelper): Current data loader.
    """
    return AdaptiveDataLoaderHelper._current


class AdaptiveDataLoaderHelper(object):
    """
    This class provides fine-grained control over adaptive training loops. It
    can be used for building more user-friendly custom data loaders, such as
    :class:`AdaptiveDataLoader`.

    Arguments:
        batch_size (int): The target total batch size across all replicas. The
            actual total batch size may be different due to rounding (each
            replica must have the same local batch size), or being scaled up
            using adaptive batch sizes.
    """

    # Epoch -> the number of dataloader loops completed so far in that epoch,
    # across all AdaptiveDataLoader objects.
    _position = collections.Counter()
    _training = None  # The AdaptiveDataLoader which loads training data.
    _current = None  # The AdaptiveDataLoader which is currently iterating.


    # SUHAS: added knobs to control what gets optimized
    _adaptive_local_bsz_bounds = False
    _use_heterogeneity_oblivious_goodput = False

    def __init__(self, batch_size=1):
        # Autoscale batch size fields.
        self._max_batch_size = None
        self._local_bsz_bounds = None
        self._local_bsz_bounds_dict = dict()
        # Create and load state.
        self._state = _AdaptiveDataLoaderState()
        adaptdl.checkpoint.load_state(self._state)
        self.batch_size = batch_size
        self.future_exit = None
        self._gradient_accumulation = False
        self._speedup_threshold = 1.05
        self._accum_count = 0

    @property
    def current_index(self):
        """
        The total number of data samples processed so far in the current loop.
        Includes the data processed by all replicas. ``None`` if this data
        loader is not currently being iterated.
        """
        if AdaptiveDataLoaderHelper._current is not self:
            return None
        return self._state.current_index

    @current_index.setter
    def current_index(self, index):
        if AdaptiveDataLoaderHelper._current is not self:
            return
        self._state.current_index = index

    @property
    def end_index(self):
        """
        (Optional) Can be used to track the end index of dataset across
         restarts.
        """
        return self._state.end_index

    @end_index.setter
    def end_index(self, index):
        """
        (Optional) Supports mutations of end_index
        """
        self._state.end_index = index

    @property
    def max_batch_size(self):
        """
        The maximum total batch size allowed for adaptive batch size. ``None``
        if adaptive batch size is disabled.
        """
        return self._max_batch_size

    @property
    def local_bsz_bounds(self):
        """
        The local batch size bounds on each replica. A pair of integers,
        (min_local_bsz, max_local_bsz).
        """
        return self._local_bsz_bounds

    @property
    def local_bsz_bounds_dict(self):
        """
        The local batch size bounds on each replica for each GPU type. A pair of integers,
        (min_local_bsz, max_local_bsz) for each GPU type stored as a dict: GPU type -> bounds.
        """
        return self._local_bsz_bounds_dict

    @property
    def current_local_bsz(self):
        """
        The current logical local batch size used by the dataloader.
        The batch size returned by the dataloader may be smaller if
        gradient accumulation is used
        """
        return self._state.current_local_bsz

    @property
    def accumulation_steps(self):
        """
        The number of batches returned by the dataloader before a
        step is taken.
        """
        return self._state.accumulation_steps

    def is_sync_step(self):
        return self._accum_count >= self._state.accumulation_steps

    def train(self):
        """
        Set this data loader to be the one used for training. Only one data
        loader may be used for training.
        """
        if AdaptiveDataLoaderHelper._training is None:
            AdaptiveDataLoaderHelper._training = self
        set_batch_size(self.batch_size, self.max_batch_size,
                       self.local_bsz_bounds, self._gradient_accumulation, self.local_bsz_bounds_dict)

    def autoscale_batch_size(self, max_batch_size, local_bsz_bounds=None,
                             gradient_accumulation=False, local_bsz_bounds_dict = None):
        """
        Enables adaptive batch size. Should be invoked once after the data
        loader object is created.

        Arguments:
            max_batch_size (int): Maximum total batch size allowed.
            local_bsz_bounds (tuple): A pair of (min_local_bsz, max_local_bsz),
                the min and max local batch sizes allowed on each replica.
            local_bsz_bounds_dict (dict) : GPU Type -> local_bsz_bounds for GPU type

        Raises:
            ValueError: If any of the provided batch size bounds are invalid.
        """
        if not isinstance(max_batch_size, int) or \
                max_batch_size < self.batch_size:
            raise ValueError("invalid max_batch_size")
        print(f"self.batch_size: {self.batch_size}, local_bsz_bounds: {local_bsz_bounds}")
        if local_bsz_bounds is not None and (
                local_bsz_bounds[0] is not None and
                local_bsz_bounds[0] > self.batch_size or
                local_bsz_bounds[1] is not None and
                local_bsz_bounds[1] < self.batch_size):
            raise ValueError("invalid local_bsz_bounds")
        self._max_batch_size = max_batch_size

        # validate local bsz bounds for each GPU type
        self._local_bsz_bounds = local_bsz_bounds
        if local_bsz_bounds_dict is not None:
            for gpu_type, gpu_bounds in local_bsz_bounds_dict.items():
                if (gpu_bounds[0] is not None and gpu_bounds[0] > self.batch_size or
                    gpu_bounds[1] is not None and gpu_bounds[1] < self.batch_size):
                    raise ValueError("invalid local_bsz_bounds for gpu type: {gpu_type} -> {gpu_bounds}, batch size: {self.batch_size}")
        self._local_bsz_bounds_dict = local_bsz_bounds_dict
        self._gradient_accumulation = gradient_accumulation
        self.train()

    def _sync_local_bsz(self):
        local_bsz_bounds = self._local_bsz_bounds
        gpu_type = adaptdl.env.gpu_type()

        # tune local bsz bounds if hints provided
        if self._adaptive_local_bsz_bounds:
            if gpu_type in self._local_bsz_bounds_dict:
                LOG.warn(f"Adapting BSZ bounds: {local_bsz_bounds} -> {self._local_bsz_bounds_dict[gpu_type]}")
                local_bsz_bounds = self._local_bsz_bounds_dict[gpu_type]
        
        if "TARGET_BATCH_SIZE" in os.environ:
            eps = 1e-8
            batch_size = int(os.environ["TARGET_BATCH_SIZE"])
            local_bsz = math.ceil(batch_size / adaptdl.env.num_replicas() - eps)
            if local_bsz_bounds is not None and local_bsz_bounds[1] is not None:
                accum_steps = math.ceil(local_bsz / local_bsz_bounds[1] - eps) - 1
            else:
                accum_steps = 0
            if adaptdl.env.num_replicas() == 1:
                accum_steps = max(1, accum_steps)
            self._state.current_local_bsz = math.ceil(local_bsz / (accum_steps + 1) - eps)
            self._state.accumulation_steps = accum_steps
            return self.current_local_bsz

        # choose heterogeneity-aware goodput function if configured
        if not self._use_heterogeneity_oblivious_goodput:
            goodput_fn = get_goodput_fn(gpu_type)
        else:
            goodput_fn = get_goodput_fn()

        if self.max_batch_size is None or goodput_fn is None:
            # No autoscale batch size, just divide batch size evenly.
            self._state.current_local_bsz = math.ceil(
                self.batch_size / adaptdl.env.num_replicas())
            self._state.accumulation_steps = 0
        elif not self._state.current_local_bsz:
            # if init, use the batch size suggested
            _, atomic_bsz, accum_steps = goodput_fn.optimize(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=local_bsz_bounds,
                accumulation=self._gradient_accumulation)
            self._state.current_local_bsz = atomic_bsz
            self._state.accumulation_steps = accum_steps
        else:
            # if not first time, we check against the relative speedup
            suggest_goodput, atomic_bsz, accum_steps = goodput_fn.optimize(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=local_bsz_bounds,
                accumulation=self._gradient_accumulation)
            # get current goodput
            current_goodput = goodput_fn(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                self.current_local_bsz, self.accumulation_steps)
            # use only if speedup is significant
            speedup = suggest_goodput / max(current_goodput, 1e-8)
            if speedup > self._speedup_threshold:
                self._state.current_local_bsz = atomic_bsz
                self._state.accumulation_steps = accum_steps
        self._state.current_local_bsz, self._state.accumulation_steps = \
            adaptdl.collective.broadcast((self._state.current_local_bsz,
                                          self._state.accumulation_steps))

        # set local bsz bounds for next invocation 
        # no-op if (not _adaptive_local_bsz_bounds)
        self._local_bsz_bounds = local_bsz_bounds

        return self.current_local_bsz

    @property
    def training(self):
        return self is AdaptiveDataLoaderHelper._training

    @contextmanager
    def profile(self, record):
        """
        Every iteration of every epoch should be profiled under this context.
        Note that, custom DataLoader writers should make sure that it gets
        called equal number of times on each replica.
        """
        # Synchronize the exit signal so all replicas exit after
        # the same iteration. Do this asynchronously to prevent
        # unnecessary blocking on the network.
        if self.future_exit is not None and self.future_exit.result():
            adaptdl.checkpoint.save_all_states()
            exit(143)  # Standard exit code response to SIGTERM.
        self.future_exit = adaptdl.collective.allreduce_async(
                    get_exit_flag(), lambda a, b: a or b)
        profile_step_start(self.current_local_bsz)
        yield
        # Don't profile the first batch since it may be slower.
        if self.training and self.current_index > self.current_batch_size and record:
            profile_step_commit(current_epoch(), self.current_batch_size, not self.is_sync_step())
        self._accum_count = 0 if self.is_sync_step() else self._accum_count + 1

    @contextmanager
    def context(self):
        """
        All iterators should be iterated under this context. It ensures
        proper cleanup of elastic context at the end of each epoch.
        """
        epoch = current_epoch()
        try:
            if AdaptiveDataLoaderHelper._current is not None:
                raise RuntimeError("overlapping dataloader \
                                    iterations detected")
            AdaptiveDataLoaderHelper._current = self
            yield
        finally:
            self._state.current_index = 0
            self._state.end_index = 0
            self._state.last_position[epoch] = self._position[epoch]
            self._position[epoch] += 1
            AdaptiveDataLoaderHelper._current = None

    @property
    def current_batch_size(self):
        return (self.current_local_bsz * (self.accumulation_steps + 1) *
                adaptdl.env.num_replicas())

    def skipdone(self):
        """
        Should be called just after entering the `_elastic` context to make
        sure that the dataloader loop is not replayed if has already finished
        before a restart.
        """

        epoch = current_epoch()
        position = self._position[epoch]
        if position <= self._state.last_position.get(epoch, -1):
            # Already completed the dataloader loop at the current
            # position, skip this loop and keep replaying the application
            # code.
            LOG.info("skipping %s loop at position %s in epoch %s",
                     self.__class__.__name__, position, epoch)
            self._position[epoch] += 1
            return True
        else:
            return False

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        """
        Output some useful metrics to TensorBoard.

        Arguments:
            writer (torch.utils.tensorboard.SummaryWriter): ``SummaryWriter``
                object to output metrics to.
            global_step (int): Global step value to record.
            tag_prefix (str): Prefix added to each metric's tag.
        """
        if adaptdl.env.replica_rank() > 0:
            return
        if tag_prefix and not tag_prefix.endswith("/"):
            tag_prefix += "/"
        writer.add_scalar(tag_prefix + "Total_Batch_Size",
                          self.current_batch_size, global_step)
        writer.add_scalar(tag_prefix + "Local_Batch_Size",
                          self.current_local_bsz, global_step)
        writer.add_scalar(tag_prefix + "Accumulation_Steps",
                          self.accumulation_steps, global_step)


class AdaptiveDataLoaderMixin(object):
    """
    This class provides elastic functionality to any custom DataLoader which
    inherits it. It defines a member _elastic of type
    :class:`AdaptiveDataLoaderHelper` which has useful methods and members to
    implement restart-safe, elastic DataLoaders. It also exposes public methods
    which can be used inside training loops directly from
    :class:`AdaptiveDataLoader`.
    """

    def __init__(self, batch_size):
        self._elastic = AdaptiveDataLoaderHelper(batch_size)

    def autoscale_batch_size(self, max_batch_size, local_bsz_bounds=None,
                             gradient_accumulation=False, optimize_app=None):
        local_bsz_bounds_dict = None

        # read offline profiles if they exist
        if optimize_app is not None:
            dir_path = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
            app_profile_filename = os.path.join("/seed/", optimize_app + ".csv")
            LOG.info(f"Trying to read app profile for {optimize_app} from {app_profile_filename}")
            app_profile = []
            if not os.path.exists(app_profile_filename):
                LOG.warn(f"Couldn't find file...")
            if os.path.exists(app_profile_filename):
                # set params
                self._elastic._adaptive_local_bsz_bounds = True
                self._elastic._use_heterogeneity_oblivious_goodput = True
                gpu_bsz_bounds = dict()
                max_local_bsz = None if local_bsz_bounds is None else local_bsz_bounds[1]
                min_local_bsz = None if local_bsz_bounds is None else local_bsz_bounds[0]

                LOG.info(f"Reading app profile for {optimize_app} from {app_profile_filename}")
                with open(app_profile_filename, 'r') as f:
                    reader = csv.DictReader(f)
                    for line in reader:
                        app_profile.append(line)
                        profile_bsz = int(line.get('local_bsz', None))
                        gpu_type = line.get('gpu_type', None)
                        if profile_bsz is not None:
                            _, cur_max = gpu_bsz_bounds.get(gpu_type, [None, None])
                            if cur_max is None or cur_max < profile_bsz:
                                gpu_bsz_bounds[gpu_type] = (min_local_bsz, profile_bsz)
                    
                    # seed hints to scheduler
                    seed_sched_hints(app_profile)

                LOG.info(f"App profile: {app_profile}")
                LOG.info(f"Inferred local_bsz_bounds dict: {gpu_bsz_bounds}")
                local_bsz_bounds_dict = gpu_bsz_bounds
                cur_gpu_type = adaptdl.env.gpu_type()
                if cur_gpu_type in local_bsz_bounds_dict:
                    LOG.info(f"Correcting local bsz bounds: {cur_gpu_type} : {local_bsz_bounds} -> {local_bsz_bounds_dict[cur_gpu_type]}")
                    local_bsz_bounds = local_bsz_bounds_dict[cur_gpu_type]

        self._elastic.autoscale_batch_size(max_batch_size, local_bsz_bounds,
                                           gradient_accumulation, local_bsz_bounds_dict)

    @property
    def current_local_bsz(self):
        if AdaptiveDataLoaderHelper._current is not self._elastic:
            return None
        return self._elastic.current_local_bsz

    @property
    def accumulation_steps(self):
        """
        The number of batches returned by the dataloader before a
        step is taken.
        """
        return self._elastic.accumulation_steps

    @property
    def current_batch_size(self):
        if AdaptiveDataLoaderHelper._current is not self._elastic:
            return None
        return self._elastic.current_batch_size

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        self._elastic.to_tensorboard(writer, global_step, tag_prefix)
    to_tensorboard.__doc__ = AdaptiveDataLoaderHelper.to_tensorboard.__doc__


def _worker_init_wrapper(worker_init_fn, num_workers):
    # Set globally-unique python and numpy seeds for each worker.

    @functools.wraps(worker_init_fn)
    def wrapper(worker_id):
        nonlocal num_workers
        num_workers = num_workers or 1
        # https://pytorch.org/docs/master/data.html#randomness-in-multi-process-data-loading.
        seed = torch.initial_seed() + adaptdl.env.replica_rank() * num_workers
        torch.manual_seed(seed)
        np.random.seed(seed % 2 ** 32)
        random.seed(seed)
        if worker_init_fn is not None:
            return worker_init_fn(worker_id)
    return wrapper


class AdaptiveDataLoader(DataLoader, AdaptiveDataLoaderMixin):
    """
    This class is a PyTorch DataLoader that also supports adaptive batch sizes
    and checkpoint-restart elasticity. Applications can typically use objects
    of this class as direct replacements for PyTorch DataLoaders. However, some
    notable differences are:

    1.  The ``batch_size`` argument defines the target total batch size across
        all replicas, rather than the local batch size on each replica.
    2.  Custom ``sampler`` and ``batch_sampler`` are not supported.
    3.  Iterating through the dataloader is only allowed from within an epoch
        loop (see :mod:`adaptdl.torch.epoch`), and only one dataloader loop is
        allowed at any given time.

    Arguments:
        dataset (torch.util.data.Dataset): Dataset from which to load the data.
        batch_size (int): The target total batch size across all replicas. The
            actual total batch size may be different due to rounding (each
            replica must have the same local batch size), or being scaled up
            using adaptive batch sizes.
        shuffle (bool): Whether the data is reshuffled at every epoch.
        **kwargs: Keyword arguments passed to ``torch.util.data.Dataloader``.

    Raises:
        ValueError: If ``sampler`` or ``batch_sampler`` are not ``None``.

    .. automethod:: __iter__
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        if kwargs.get("batch_sampler") is not None \
                or kwargs.get("sampler") is not None:
            raise ValueError("AdaptiveDataLoader does not support "
                             "custom 'sampler' or 'batch_sampler'")
        # Custom sampler is incompatible with shuffle=True, so we always set
        # shuffle=False in __init__ and let our own sampler do the shuffling.
        kwargs["sampler"] = ElasticSampler(dataset, shuffle=shuffle)
        kwargs["worker_init_fn"] = _worker_init_wrapper(
            kwargs.get("worker_init_fn"), kwargs.get("num_workers"))
        super().__init__(dataset, batch_size, shuffle=False, **kwargs)
        AdaptiveDataLoaderMixin.__init__(self, batch_size)

    def __iter__(self):
        """
        Iterate over batches of data. When adaptive batch size is disabled,
        stops after the entire dataset has been processed once in total by all
        replicas. This means if there are K replicas, then this method will
        iterate over ~1/K of the dataset. When adaptive batch size is enabled,
        stops after making enough statistical progress roughly equivalent to
        one pass over the dataset with non-adaptive batch size. In this case,
        the dataset may be processed more than once.

        A checkpoint-restart may be triggered in-between each batch. In this
        case, the current iteration state will be saved and restored after the
        restart, and continue where it left off.
        """
        epoch = current_epoch()
        with self._elastic.context():
            if self._elastic.skipdone():
                return
            done = False
            efficiency_file = None
            if "TRACE_EFFICIENCY" in os.environ and adaptdl.env.replica_rank() == 0:
                efficiency_file = open(adaptdl.env.checkpoint_path() + "/efficiency.txt", "a")
            if "TRACE_THROUGHPUT" in os.environ:
                assert "TRACE_EFFICIENCY" not in os.environ
                self._elastic._state.current_local_bsz = self.batch_sampler.batch_size = \
                    int(self._elastic.local_bsz_bounds[0])
            while not done:
                self.sampler.set_epoch(epoch, index=self._elastic.current_index)  # noqa: E501
                if "TRACE_THROUGHPUT" not in os.environ:
                    self.batch_sampler.batch_size = self._elastic._sync_local_bsz()
                for idx, batch in enumerate(super().__iter__()):
                    record = idx >= 5 if "TRACE_THROUGHPUT" in os.environ else True
                    with self._elastic.profile(record):
                        yield batch
                        if self._elastic.training and efficiency_file is not None:
                            json.dump({
                                "progress": get_progress(),
                                "norm": _metrics_state().grad_params[0],
                                "var": _metrics_state().grad_params[1],
                                "initial_batch_size": self.batch_size,
                                "current_batch_size": self.current_batch_size,
                                "num_replicas": adaptdl.env.num_replicas(),
                            }, efficiency_file)
                            efficiency_file.write("\n")
                        # Increment by the number of data samples processed
                        self._elastic.current_index += adaptdl.env.num_replicas() * self.batch_sampler.batch_size
                        if "TRACE_THROUGHPUT" in os.environ:
                            if idx >= 20:
                                LOG.info(self.batch_sampler.batch_size)
                                break
                        elif self._elastic.max_batch_size is not None and \
                           get_progress() >= len(self.dataset) * \
                           (epoch + 1) / self.batch_size:
                            done = True
                            break
                if "TRACE_THROUGHPUT" in os.environ:
                    if idx <= 3:  # Batch size becoming too large, stop now.
                        done = True
                    elif self.batch_sampler.batch_size >= self._elastic.local_bsz_bounds[1]:
                        done = True
                    else:
                        self._elastic._state.current_local_bsz = self.batch_sampler.batch_size = \
                            min(round(self.batch_sampler.batch_size * 2 ** 0.5),
                                self._elastic.local_bsz_bounds[1])
                elif self._elastic.max_batch_size is None:
                    done = True
                self._elastic.current_index -= \
                    self._elastic.current_index % -len(self.dataset)
            if efficiency_file is not None:
                efficiency_file.close()
            elif "TRACE_THROUGHPUT" in os.environ:
                if adaptdl.env.replica_rank() == 0:
                    profile = [{"local_bsz": k[2], **v} for k, v in _metrics_state().profile.items()]
                    with open(adaptdl.env.checkpoint_path() + "/throughput.txt", "w") as f:
                        json.dump({"profile": profile}, f)
                exit(0)


class _AdaptiveDataLoaderState(adaptdl.checkpoint.State):

    # Assume dataloaders are initialized in the same order in every replica.
    # Keep a map of epoch -> number of dataloaders initialized so far in that
    # epoch, and use that count to construct a unique name for the state.
    init_count = collections.Counter()

    def __init__(self):
        if current_dataloader() is not None:
            raise RuntimeError("dataloader may not be initialized during "
                               "dataloader iteration")
        epoch = current_epoch()
        count = _AdaptiveDataLoaderState.init_count[epoch]
        super().__init__("adaptdl-dataloader-epoch{}-{}".format(epoch, count))
        _AdaptiveDataLoaderState.init_count[epoch] += 1

        self.current_index = 0   # Index within the current dataloader loop.
        self.end_index = 0       # End index of the current DataLoader loop.
        self.last_position = {}  # Epoch -> position of last completed loop.
        self.current_local_bsz = 0
        self.accumulation_steps = 0

    def save(self, fileobj):
        pickle.dump((self.current_index, self.end_index,
                     self.last_position), fileobj)

    def load(self, fileobj):
        self.current_index, self.end_index, self.last_position = \
           pickle.load(fileobj)
