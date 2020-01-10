r"""Entry point for training a XLNet based model for AICUP2019 Task1.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import ast
import csv
import glob
import gzip
import json
import math
import errno
import msgpack
import logging
import argparse
import datetime
import functools
import itertools
from pathlib import Path
from collections import defaultdict

# print('some red text1', file=sys.stderr)
import colorama
from colorama import Fore, Back, Style
# print(Fore.RED + 'some red text2' + Style.RESET_ALL, file=sys.stderr)
colorama.init(
)  # this needs to run before first run of tf_logging._get_logger()
# print(Fore.RED + 'some red text3' + Style.RESET_ALL, file=sys.stderr)
import tensorflow as tf
import tensorflow_hub as hub
# print(Fore.RED + 'some red text4' + Style.RESET_ALL, file=sys.stderr)
from tensorflow.python.ops import variables, inplace_ops
from tensorflow.python.data.ops import iterator_ops
# from tensorflow.contrib.data.python.ops.iterator_ops import _Saveable # 1.11
from tensorflow.python.data.experimental.ops.iterator_ops import _Saveable  # 1.12
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.training import training_util
from tensorflow.python.framework import meta_graph
from tensorflow.python.data.util import nest
from tensorflow.python.util.nest import is_sequence
from tensorflow.contrib.layers.python.layers import adaptive_clipping_fn
from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import tf_logging
import numpy as np
import coloredlogs
from tqdm import tqdm
from xlnet import xlnet

_NEG_INF = -1e9

tfversion = tuple([int(s) for s in tf.__version__.split('-')[0].split('.')])


def verify_input_path(p):
    # get absolute path to dataset directory
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # doesn't exist
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    return path


def verify_output_path(p):
    # get absolute path to dataset directory
    path = Path(os.path.abspath(os.path.expanduser(p)))
    # existing file
    if path.exists():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)
    # is dir
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
    # assert dirs
    path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    return path


class TqdmFile(object):
    """ A file-like object that will write to tqdm"""
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            # print(Fore.RED + 'some red text' + Style.RESET_ALL)
            if tfversion[0] == 1 and tfversion[1] <= 12:
                tqdm.write(x, file=self.file)
            else:
                tqdm.write(x.rstrip(), file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


# Disable cpp warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Show debugging output, default: tf.logging.INFO

logger = None
FLAGS = None


def pad_to_multiples(features, labels, pad_to_mutiples_of=8, padding_values=0):
    """Nvidia Volta Tensor Cores are enabled when data shape is multiples of 8
    """
    max_len = tf.shape(labels)[1]
    target_len = tf.cast(
        tf.multiply(
            tf.ceil(tf.truediv(max_len, pad_to_mutiples_of)), pad_to_mutiples_of
        ), tf.int32
    )
    paddings = [[0, 0], [0, target_len - max_len]]
    features['protein'] = tf.pad(
        tensor=features['protein'],
        paddings=paddings,
        constant_values=padding_values
    )
    return features, tf.pad(
        tensor=labels, paddings=paddings, constant_values=padding_values
    )


def bucket_by_sequence_length_and_pad_to_multiples(
    element_length_func,
    bucket_boundaries,
    bucket_batch_sizes,
    padded_shapes=None,
    padding_values=None,
    pad_to_mutiples_of=None,
    pad_to_bucket_boundary=False
):
    """A transformation that buckets elements in a `Dataset` by length.

    Nvidia Volta Tensor Cores are enabled when data shape is multiples of 8

    Elements of the `Dataset` are grouped together by length and then are padded
    and batched.

    This is useful for sequence tasks in which the elements have variable length.
    Grouping together elements that have similar lengths reduces the total
    fraction of padding in a batch which increases training step efficiency.

    Args:
      element_length_func: function from element in `Dataset` to `tf.int32`,
        determines the length of the element, which will determine the bucket it
        goes into.
      bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
      bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
        `len(bucket_boundaries) + 1`.
      padded_shapes: Nested structure of `tf.TensorShape` to pass to
        @{tf.data.Dataset.padded_batch}. If not provided, will use
        `dataset.output_shapes`, which will result in variable length dimensions
        being padded out to the maximum length in each batch.
      padding_values: Values to pad with, passed to
        @{tf.data.Dataset.padded_batch}. Defaults to padding with 0.
      pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
        size to maximum length in batch. If `True`, will pad dimensions with
        unknown size to bucket boundary, and caller must ensure that the source
        `Dataset` does not contain any elements with length longer than
        `max(bucket_boundaries)`.

    Returns:
      A `Dataset` transformation function, which can be passed to
      @{tf.data.Dataset.apply}.

    Raises:
      ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
    """
    with tf.name_scope("bucket_by_sequence_length_and_pad_to_multiples"):
        if len(bucket_batch_sizes) != (len(bucket_boundaries) + 1):
            raise ValueError(
                "len(bucket_batch_sizes) must equal len(bucket_boundaries) + 1"
            )

        batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)

        def element_to_bucket_id(*args):
            """Return int64 id of the length bucket for this element."""
            seq_length = element_length_func(*args)

            boundaries = list(bucket_boundaries)
            buckets_min = [np.iinfo(np.int32).min] + boundaries
            buckets_max = boundaries + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less_equal(buckets_min, seq_length),
                tf.less(seq_length, buckets_max)
            )
            bucket_id = tf.reduce_min(tf.where(conditions_c))

            return bucket_id

        def window_size_fn(bucket_id):
            # The window size is set to the batch size for this bucket
            window_size = batch_sizes[bucket_id]
            return window_size

        def make_padded_shapes(shapes, none_filler=None):
            padded = []
            # print('shapes', shapes)
            for shape in nest.flatten(shapes):
                # print('shape', shape)
                shape = tf.TensorShape(shape)
                # print(tf.TensorShape(None))
                shape = [none_filler if d.value is None else d for d in shape]
                # print(shape)
                padded.append(shape)
            return nest.pack_sequence_as(shapes, padded)

        def batching_fn(bucket_id, grouped_dataset):
            """Batch elements in dataset."""
            # ({'protein': TensorShape(None), 'lengths': TensorShape([])}, TensorShape(None))
            print(grouped_dataset.output_shapes)
            batch_size = batch_sizes[bucket_id]
            none_filler = None
            if pad_to_bucket_boundary:
                err_msg = (
                    "When pad_to_bucket_boundary=True, elements must have "
                    "length <= max(bucket_boundaries)."
                )
                check = tf.assert_less(
                    bucket_id,
                    tf.constant(len(bucket_batch_sizes) - 1, dtype=tf.int64),
                    message=err_msg
                )
                with tf.control_dependencies([check]):
                    boundaries = tf.constant(bucket_boundaries, dtype=tf.int64)
                    bucket_boundary = boundaries[bucket_id]
                    none_filler = bucket_boundary
            # print(padded_shapes or grouped_dataset.output_shapes)
            shapes = make_padded_shapes(
                padded_shapes or grouped_dataset.output_shapes,
                none_filler=none_filler
            )
            return grouped_dataset.padded_batch(
                batch_size, shapes, padding_values
            )

        def _apply_fn(dataset):
            return dataset.apply(
                tf.contrib.data.group_by_window(
                    element_to_bucket_id,
                    batching_fn,
                    window_size_func=window_size_fn
                )
            )

        return _apply_fn


def debug_serving():
    import tensorflow as tf
    import tensorflow.contrib.eager as tfe
    tf.enable_eager_execution()
    import numpy as np
    protein_strs = tf.constant(['FLI', 'MVPA', 'GS'])
    # <tf.Tensor: id=440, shape=(3,), dtype=string, numpy=array([b'FLI', b'MVPA', b'GS'], dtype=object)>
    proteins = [
        [aa_index[a] for a in np.array(ps).tolist().decode('utf-8')]
        for ps in protein_strs
    ]
    proteins = [[1, 2, 3], [4, 5, 1, 6], [1, 2]]
    np_proteins = [np.array(p, dtype=np.uint8) for p in proteins]

    def make_sequence_example(protein):
        # The object we return
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        # sequence_length = len(protein)
        # ex.context.feature["length"].int64_list.value.append(sequence_length)
        # Feature lists for the two sequential features of our example
        fl_protein = ex.feature_lists.feature_list["protein"]
        for aa in protein:
            fl_protein.feature.add().bytes_list.value.append(aa.tostring())
        return ex

    # make_sequence_example(np_proteins[0])
    # feature_lists {
    #     feature_list {
    #         key: "protein"
    #         value {
    #             feature {
    #                 bytes_list {
    #                     value: "\001"
    #                 }
    #             }
    #             feature {
    #                 bytes_list {
    #                     value: "\002"
    #                 }
    #             }
    #             feature {
    #                 bytes_list {
    #                     value: "\003"
    #                 }
    #             }
    #         }
    #     }
    # }
    # serialized = make_sequence_example(np_proteins[0]).SerializeToString()
    serialized = [
        make_sequence_example(npp).SerializeToString() for npp in np_proteins
    ]
    # b'\x12"\n \n\x07protein\x12\x15\n\x05\n\x03\n\x01\x01\n\x05\n\x03\n\x01\x02\n\x05\n\x03\n\x01\x03'
    context_features = {
        # 'length': tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        'protein': tf.FixedLenSequenceFeature([], dtype=tf.string)
        # uint8 is not in the list of allowed values: float, int64, string
    }
    example_names = ['t1', 'test2', 'aaa']
    context, sequence, lengths = tf.io.parse_sequence_example(
        serialized=serialized,
        # A vector (1-D Tensor) of type string containing binary serialized
        # serialized `SequenceExample` proto.
        context_features=context_features,
        # A `dict` mapping feature keys to `FixedLenFeature` or
        # `VarLenFeature` values. These features are associated with a
        # `SequenceExample` as a whole.
        sequence_features=sequence_features,
        # A `dict` mapping feature keys to
        # `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
        # associated with data within the `FeatureList` section of the
        # `SequenceExample` proto.
        example_names=example_names,
        # A vector (1-D Tensor) of strings (optional), the name of
        # the serialized proto.
        name=None
        # A name for this operation (optional).
    )
    # context = {}
    # sequence = {'protein': <tf.Tensor: id=3, shape=(3, 4), dtype=string, numpy=
    #     array([[b'\x01', b'\x02', b'\x03', b''],
    #            [b'\x04', b'\x05', b'\x01', b'\x06'],
    #         [b'\x01', b'\x02', b'', b'']], dtype=object)>}
    # lengths = {'protein': <tf.Tensor: id=4, shape=(3,), dtype=int64, numpy=array([3, 4, 2], dtype=int64)>})
    mode = tf.estimator.ModeKeys.PREDICT
    dataset = tf.data.Dataset.from_tensor_slices(serialized)
    dataset = dataset.map(
        functools.partial(parse_sequence_example, mode=mode),
        num_parallel_calls=None
    )
    dataset = dataset.padded_batch(
        batch_size=10, padded_shapes={
            'protein': [None],
            'lengths': []
        }
    )
    sequences = tf.data.experimental.get_single_element(dataset)
    # sequences = {'protein': <tf.Tensor: id=328, shape=(3, 4), dtype=int32, numpy=
    # array([[1, 2, 3, 0],
    #     [4, 5, 1, 6],
    #     [1, 2, 0, 0]])>, 'lengths': <tf.Tensor: id=327, shape=(3,), dtype=int32, numpy=array([3, 4, 2])>}
    iterator = tfe.Iterator(dataset)
    print(iterator.next())

    decoded = [
        tf.decode_raw(
            bytes=x, out_type=tf.uint8, little_endian=True, name=None
        ) for x in sequence['protein']
    ]


def make_sequence(protein):
    sequence = {}
    sequence['protein'] = tf.cast(x=protein, dtype=tf.int32, name=None)
    sequence['lengths'] = tf.shape(input=protein, name=None,
                                   out_type=tf.int32)[0]
    return sequence


def serving_input_str_receiver_fn():
    """An input receiver that expects a serialized tf.SequenceExample."""
    serialized = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_protein_string_tensor'
    )
    mapping = tf.constant([x for x in aa_list])
    table = tf.contrib.lookup.index_table_from_tensor(
        mapping=mapping, num_oov_buckets=1, default_value=-1
    )
    receiver_tensors = {'protein_sequences': serialized}
    mode = tf.estimator.ModeKeys.PREDICT
    dataset = tf.data.Dataset.from_tensor_slices(serialized)
    dataset = dataset.map(
        lambda x: table.lookup(tf.string_split([x], delimiter="").values)
    )
    dataset = dataset.map(make_sequence)
    # dataset = dataset.map(functools.partial(parse_sequence_example, mode=mode))
    dataset = dataset.padded_batch(
        batch_size=1000000, padded_shapes={
            'protein': [None],
            'lengths': []
        }
    )
    sequences = tf.data.experimental.get_single_element(dataset)
    return tf.estimator.export.ServingInputReceiver(sequences, receiver_tensors)


def serving_input_dataset_receiver_fn():
    """An input receiver that expects a serialized tf.SequenceExample."""
    serialized = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor'
    )
    print_op = tf.print(
        "serialized:",
        serialized,
        serialized[0].dtype,
        output_stream=sys.stderr
    )
    # tf.logging.info("serialized:", serialized, type(serialized[0]))
    receiver_tensors = {'sequences': serialized}
    mode = tf.estimator.ModeKeys.PREDICT
    with tf.control_dependencies([print_op]):
        dataset = tf.data.Dataset.from_tensor_slices(serialized)
        dataset = dataset.map(
            functools.partial(parse_sequence_example, mode=mode),
            num_parallel_calls=None
        )
        dataset = dataset.padded_batch(
            batch_size=1000000,
            padded_shapes={
                'protein': [None],
                'lengths': []
            }
        )
        sequences = tf.data.experimental.get_single_element(dataset)
        return tf.estimator.export.ServingInputReceiver(
            sequences, receiver_tensors
        )


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.SequenceExample."""
    serialized = tf.placeholder(
        dtype=tf.string, shape=[None], name='input_example_tensor'
    )
    receiver_tensors = {'sequences': serialized}
    context_features = {
        # 'length': tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        'protein': tf.FixedLenSequenceFeature([], dtype=tf.string)
    }
    context, sequence, lengths = tf.io.parse_sequence_example(
        serialized=serialized,
        # A scalar (0-D Tensor) of type string, a single binary
        # serialized `SequenceExample` proto.
        context_features=context_features,
        # A `dict` mapping feature keys to `FixedLenFeature` or
        # `VarLenFeature` values. These features are associated with a
        # `SequenceExample` as a whole.
        sequence_features=sequence_features,
        # A `dict` mapping feature keys to
        # `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
        # associated with data within the `FeatureList` section of the
        # `SequenceExample` proto.
        example_names=None,
        # A scalar (0-D Tensor) of strings (optional), the name of
        # the serialized proto.
        name=None
        # A name for this operation (optional).
    )
    sequence['protein'] = tf.decode_raw(
        bytes=sequence['protein'],
        out_type=tf.uint8,
        little_endian=True,
        name=None
    )
    # tf.Tensor: shape=(sequence_length, 1), dtype=uint8
    sequence['protein'] = tf.cast(
        x=sequence['protein'], dtype=tf.int32, name=None
    )
    # embedding_lookup expects int32 or int64
    # tf.Tensor: shape=(sequence_length, 1), dtype=int32
    sequence['protein'] = tf.squeeze(
        input=sequence['protein'],
        axis=[],
        # An optional list of `ints`. Defaults to `[]`.
        # If specified, only squeezes the dimensions listed. The dimension
        # index starts at 0. It is an error to squeeze a dimension that is not 1.
        # Must be in the range `[-rank(input), rank(input))`.
        name=None
    )
    # tf.Tensor: shape=(sequence_length, ), dtype=int32
    # tf.Tensor: shape=(batch_size, sequence_length, ), dtype=int32
    # protein = tf.one_hot(protein, params.vocab_size)
    sequence['lengths'] = lengths['protein']
    return tf.estimator.export.ServingInputReceiver(sequence, receiver_tensors)


class EpochCheckpointInputPipelineHookSaver(tf.train.Saver):
    """`Saver` with a different default `latest_filename`.

  This is used in the `CheckpointInputPipelineHook` to avoid conflicts with
  the model ckpt saved by the `CheckpointSaverHook`.
  """
    def __init__(
        self,
        var_list,
        latest_filename,
        sharded=False,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=10000.0,
        defer_build=False,
        save_relative_paths=True
    ):
        super(EpochCheckpointInputPipelineHookSaver, self).__init__(
            var_list,
            sharded=sharded,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            defer_build=defer_build,
            save_relative_paths=save_relative_paths
        )
        self._latest_filename = latest_filename

    def save(
        self,
        sess,
        save_path,
        global_step=None,
        latest_filename=None,
        meta_graph_suffix="meta",
        write_meta_graph=True,
        write_state=True,
        strip_default_attrs=False
    ):
        return super(EpochCheckpointInputPipelineHookSaver, self).save(
            sess, save_path, global_step, latest_filename or
            self._latest_filename, meta_graph_suffix, write_meta_graph,
            write_state, strip_default_attrs
        )


class EpochCheckpointInputPipelineHook(tf.train.SessionRunHook):
    """Checkpoints input pipeline state every N steps or seconds.

    This hook saves the state of the iterators in the `Graph` so that when
    training is resumed the input pipeline continues from where it left off.
    This could potentially avoid overfitting in certain pipelines where the
    number of training steps per eval are small compared to the dataset
    size or if the training pipeline is pre-empted.

    Differences from `CheckpointSaverHook`:
    1. Saves only the input pipelines in the "iterators" collection and not the
       global variables or other saveable objects.
    2. Does not write the `GraphDef` and `MetaGraphDef` to the summary.

    Example of checkpointing the training pipeline:

    ```python
    est = tf.estimator.Estimator(model_fn)
    while True:
      est.train(
          train_input_fn,
          hooks=[tf.contrib.data.CheckpointInputPipelineHook(est)],
          steps=train_steps_per_eval)
      # Note: We do not pass the hook here.
      metrics = est.evaluate(eval_input_fn)
      if should_stop_the_training(metrics):
        break
    ```

    This hook should be used if the input pipeline state needs to be saved
    separate from the model checkpoint. Doing so may be useful for a few reasons:
    1. The input pipeline checkpoint may be large, if there are large shuffle
       or prefetch buffers for instance, and may bloat the checkpoint size.
    2. If the input pipeline is shared between training and validation, restoring
       the checkpoint during validation may override the validation input
       pipeline.

    For saving the input pipeline checkpoint alongside the model weights use
    @{tf.contrib.data.make_saveable_from_iterator} directly to create a
    `SaveableObject` and add to the `SAVEABLE_OBJECTS` collection. Note, however,
    that you will need to be careful not to restore the training iterator during
    eval. You can do that by not adding the iterator to the SAVEABLE_OBJECTS
    collector when building the eval graph.
    """
    def __init__(
        self,
        checkpoint_dir,
        config,
        save_timer=None,
        save_secs=None,
        save_steps=None,
        checkpoint_basename="input",
        listeners=None,
        defer_build=False,
        save_relative_paths=True
    ):
        """Initializes a `EpochCheckpointInputPipelineHook`.
        Creates a custom EpochCheckpointInputPipelineHookSaver

        Args:
            checkpoint_dir: `str`, base directory for the checkpoint files.
        save_timer: `SecondOrStepTimer`, timer to save checkpoints.
        save_secs: `int`, save every N secs.
        save_steps: `int`, save every N steps.
        checkpoint_basename: `str`, base name for the checkpoint files.
        listeners: List of `CheckpointSaverListener` subclass instances.
            Used for callbacks that run immediately before or after this hook saves
            the checkpoint.
        config: tf.estimator.RunConfig.

        Raises:
            ValueError: One of `save_steps` or `save_secs` should be set.
            ValueError: At most one of saver or scaffold should be set.
        """
        # `checkpoint_basename` is "input.ckpt" for non-distributed pipelines or
        # of the form "input_<task_type>_<task_id>.ckpt" for distributed pipelines.
        # Note: The default `checkpoint_basename` used by `CheckpointSaverHook` is
        # "model.ckpt". We intentionally choose the input pipeline checkpoint prefix
        # to be different to avoid conflicts with the model checkpoint.

        # pylint: disable=protected-access
        tf.logging.info("Create EpochCheckpointInputPipelineHook.")
        self._checkpoint_dir = checkpoint_dir
        self._config = config
        self._defer_build = defer_build
        self._save_relative_paths = save_relative_paths

        self._checkpoint_prefix = checkpoint_basename
        if self._config.num_worker_replicas > 1:
            # Distributed setting.
            suffix = "_{}_{}".format(
                self._config.task_type, self._config.task_id
            )
            self._checkpoint_prefix += suffix
        # pylint: enable=protected-access

        # We use a composition paradigm instead of inheriting from
        # `CheckpointSaverHook` because `Estimator` does an `isinstance` check
        # to check whether a `CheckpointSaverHook` is already present in the list
        # of hooks and if not, adds one. Inheriting from `CheckpointSaverHook`
        # would thwart this behavior. This hook checkpoints *only the iterators*
        # and not the graph variables.
        self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
        if save_timer:
            self._timer = save_timer
        else:
            self._timer = tf.train.SecondOrStepTimer(
                every_secs=save_secs, every_steps=save_steps
            )
        self._listeners = listeners or []
        self._steps_per_run = 1

        # Name for the protocol buffer file that will contain the list of most
        # recent checkpoints stored as a `CheckpointState` protocol buffer.
        # This file, kept in the same directory as the checkpoint files, is
        # automatically managed by the `Saver` to keep track of recent checkpoints.
        # The default name used by the `Saver` for this file is "checkpoint". Here
        # we use the name "checkpoint_<checkpoint_prefix>" so that in case the
        # `checkpoint_dir` is the same as the model checkpoint directory, there are
        # no conflicts during restore.
        self._latest_filename = self._checkpoint_prefix + '.latest'
        self._first_run = True

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        # Build a Saver that saves all iterators in the `GLOBAL_ITERATORS`
        # collection
        iterators = tf.get_collection(iterator_ops.GLOBAL_ITERATORS)
        saveables = [_Saveable(i) for i in iterators]
        self._saver = EpochCheckpointInputPipelineHookSaver(
            saveables,
            self._latest_filename,
            sharded=False,
            max_to_keep=self._config.keep_checkpoint_max,
            keep_checkpoint_every_n_hours=self._config.
            keep_checkpoint_every_n_hours,
            defer_build=self._defer_build,
            save_relative_paths=self._save_relative_paths
        )

        self._summary_writer = tf.summary.FileWriterCache.get(
            self._checkpoint_dir
        )
        self._global_step_tensor = training_util._get_or_create_global_step_read(
        )  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use EpochCheckpointInputPipelineHook."
            )
        for l in self._listeners:
            l.begin()

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
        session: A TensorFlow Session that has been created.
        coord: A Coordinator object which keeps track of all threads.
        """
        global_step = session.run(self._global_step_tensor)
        self._timer.update_last_triggered_step(global_step)

    def _maybe_restore_input_ckpt(self, session):
        # Ideally this should be run in after_create_session but is not for the
        # following reason:
        # Currently there is no way of enforcing an order of running the
        # `SessionRunHooks`. Hence it is possible that the `_DatasetInitializerHook`
        # is run *after* this hook. That is troublesome because
        # 1. If a checkpoint exists and this hook restores it, the initializer hook
        #    will override it.
        # 2. If no checkpoint exists, this hook will try to save an initialized
        #    iterator which will result in an exception.
        #
        # As a temporary fix we enter the following implicit contract between this
        # hook and the _DatasetInitializerHook.
        # 1. The _DatasetInitializerHook initializes the iterator in the call to
        #    after_create_session.
        # 2. This hook saves the iterator on the first call to `before_run()`, which
        #    is guaranteed to happen after `after_create_session()` of all hooks
        #    have been run.

        # Check if there is an existing checkpoint. If so, restore from it.
        # pylint: disable=protected-access
        latest_checkpoint_path = tf.train.latest_checkpoint(
            self._checkpoint_dir, latest_filename=self._latest_filename
        )
        if latest_checkpoint_path:
            self._get_saver().restore(session, latest_checkpoint_path)

    def before_run(self, run_context):
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """
        if self._first_run:
            self._maybe_restore_input_ckpt(run_context.session)
            self._first_run = False
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
            stale_global_step + self._steps_per_run
        ):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                if self._save(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
        session: A TensorFlow Session that will be soon closed.
        """

        # delete latest checkpoint file
        input_checkpoint_files = Path(self._checkpoint_dir
                                     ).glob(self._checkpoint_prefix + '*')
        # print(input_checkpoint_files)
        for f in input_checkpoint_files:
            if f.exists():
                f.unlink()
                # print('DELETE: ', f)
        tf.logging.debug("Removed input checkpoints")

        last_step = session.run(self._global_step_tensor)
        for l in self._listeners:
            l.end(session, last_step)

    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        tf.logging.info(
            "Saving\033[31m input\033[0m checkpoints for %d into %s.", step,
            self._save_path
        )

        for l in self._listeners:
            l.before_save(session, step)

        self._get_saver().save(session, self._save_path, global_step=step)
        self._summary_writer.add_session_log(
            SessionLog(
                status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path
            ),  # pylint: disable=no-member
            step
        )

        should_stop = False
        for l in self._listeners:
            if l.after_save(session, step):
                tf.logging.info(
                    "A CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l)
                )
                should_stop = True
        return should_stop

    def _get_saver(self):
        return self._saver


class EpochCheckpointSaverHook(tf.train.CheckpointSaverHook):
    """This checkpoint saver hook saves two types of checkpoints:

    1. step:
    * Saves on save_secs or save_steps
    * Does not save on begin or end
    * Saves input pipeline state to continue training the remaining examples in the current epoch
    * Separately configurable garbage collection criteria from epoch
        * Defaults: max_to_keep=10, keep_checkpoint_every_n_hours=6
    * The default list of CheckpointSaverListener does not run on step checkpoint saves,
      you may configure a separate list of CheckpointSaverListeners by setting the step_listeners init arg
    * filename = step
    * latest_filename = step.latest

    2. epoch:
    * Does not save on save_secs or save_steps
    * Saves on epoch end
    * Does not save input pipeline
    * Separately configurable garbage collection criteria from step
        * Does not garbage collect by default
            * Defaults: max_to_keep=9999, keep_checkpoint_every_n_hours=999999
        * set epoch_saver to a custom tf.train.Saver to change defaults
    * The default list of CheckpointSaverListener only runs on epoch checkpoint saves,
      this includes the default _NewCheckpointListenerForEvaluate added by tf.estimator.train_and_evaluate
      which runs the eval loop after every new checkpoint
    * filename = epoch
    * latest_filename = epoch.latest

    Usage:
    * Added to the list of EstimatorSpec.training_chief_hooks in your model_fn.
      * This prevents the default CheckpointSaverHook from being added
    * The end of an "epoch" is defined as the input_fn raising the OutOfRangeError,
      don't repeat the dataset or set the repeat_count to 1 if you want the "expected" behavior of
      one "epoch" being one iteration over all of the training data.
    * estimator.train or tf.estimator.train_and_evaluate will exit after the OutOfRangeError,
      wrap it with a for loop to train a limited number of epochs or a while True loop to train forever.

    Fixes more than one graph event per run warning in Tensorboard
    """
    def __init__(
        self,
        checkpoint_dir,
        epoch_tensor=None,
        save_timer=None,
        save_secs=None,
        save_steps=None,
        saver=None,
        checkpoint_basename=None,
        scaffold=None,
        listeners=None,
        step_listeners=None,
        epoch_saver=None,
        epoch_basename='epoch',
        step_basename='step',
        epoch_latest_filename='epoch.latest',
        step_latest_filename='step.latest'
    ):
        """Maintains compatibility with the `CheckpointSaverHook`.

        Args:
        checkpoint_dir: `str`, base directory for the checkpoint files.
        save_timer: `SecondOrStepTimer`, timer to save checkpoints.
        save_secs: `int`, save a step checkpoint every N secs.
        save_steps: `int`, save a step checkpoint every N steps.
        saver: `Saver` object, used for saving a final step checkpoint.
        checkpoint_basename: `str`, base name for the checkpoint files.
        scaffold: `Scaffold`, use to get saver object a final step checkpoint.
        listeners: List of `CheckpointSaverListener` subclass instances.
            Used for callbacks that run immediately before or after this hook saves
            a epoch checkpoint.
        step_listeners: List of `CheckpointSaverListener` subclass instances.
            Used for callbacks that run immediately before or after this hook saves
            a step checkpoint.
        epoch_saver: `Saver` object, used for saving a epoch checkpoint.
        step_basename: `str`, base name for the step checkpoint files.
        epoch_basename: `str`, base name for the epoch checkpoint files.

        Raises:
        ValueError: One of `save_steps` or `save_secs` should be set.
        ValueError: At most one of saver or scaffold should be set.
        """
        tf.logging.info("Create EpochCheckpointSaverHook.")
        if saver is not None and scaffold is not None:
            raise ValueError("You cannot provide both saver and scaffold.")
        self._saver = saver
        self._checkpoint_dir = checkpoint_dir
        checkpoint_basename = checkpoint_basename or ''
        epoch_basename = ''.join(
            (checkpoint_basename, epoch_basename or 'step')
        )
        step_basename = ''.join((checkpoint_basename, step_basename or 'step'))
        self._epoch_save_path = os.path.join(checkpoint_dir, epoch_basename)
        self._step_save_path = os.path.join(checkpoint_dir, step_basename)
        self._epoch_latest_filename = epoch_latest_filename or 'epoch.latest'
        self._step_latest_filename = step_latest_filename or 'step.latest'
        self._scaffold = scaffold
        if save_timer:
            self._timer = save_timer
        else:
            self._timer = tf.train.SecondOrStepTimer(
                every_secs=save_secs, every_steps=save_steps
            )
        self._epoch_listeners = listeners or []
        # In _train_with_estimator_spec
        # saver_hooks[0]._listeners.extend(saving_listeners)
        self._step_listeners = step_listeners or []
        self._listeners = self._step_listeners
        self._epoch_saver = epoch_saver
        self._steps_per_run = 1
        self._epoch_tensor = epoch_tensor

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        self._summary_writer = tf.summary.FileWriterCache.get(
            self._checkpoint_dir
        )
        self._global_step_tensor = training_util._get_or_create_global_step_read(
        )  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use EpochCheckpointSaverHook."
            )

        if self._epoch_saver is None:
            self._epoch_saver = tf.train.Saver(
                sharded=False,
                max_to_keep=9999,
                keep_checkpoint_every_n_hours=999999,
                defer_build=False,
                save_relative_paths=True
            )

        for l in self._epoch_listeners:
            l.begin()
        for l in self._step_listeners:
            l.begin()
            l._is_first_run = False

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
        session: A TensorFlow Session that has been created.
        coord: A Coordinator object which keeps track of all threads.
        """
        global_step = session.run(self._global_step_tensor)
        # We do write graph and saver_def at the first call of before_run.
        # We cannot do this in begin, since we let other hooks to change graph and
        # add variables in begin. Graph is finalized after all begin calls.
        tf.train.write_graph(
            tf.get_default_graph().as_graph_def(add_shapes=True),
            self._checkpoint_dir, "graph.pbtxt"
        )
        saver_def = self._get_saver().saver_def if self._get_saver() else None
        graph = tf.get_default_graph()
        meta_graph_def = meta_graph.create_meta_graph_def(
            graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def
        )
        self._summary_writer.add_graph(graph, global_step=global_step)
        self._summary_writer.add_meta_graph(
            meta_graph_def, global_step=global_step
        )
        # The checkpoint saved here is the state at step "global_step".
        # do not save any checkpoints at start
        # self._save(session, global_step)
        self._timer.update_last_triggered_step(global_step)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
        run_context: A `SessionRunContext` object.
        run_values: A SessionRunValues object.
        """
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
            stale_global_step + self._steps_per_run
        ):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                if self._save_step(run_context.session, global_step):
                    run_context.request_stop()

    def end(self, session):
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
        session: A TensorFlow Session that will be soon closed.
        """
        # savables = tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
        # savables_ref = tf.get_collection_ref(tf.GraphKeys.SAVEABLE_OBJECTS)
        # print('SAVEABLE_OBJECTS before', len(savables_ref), savables_ref)
        # # remove tensorflow.contrib.data.python.ops.iterator_ops._Saveable object
        # for v in savables:
        #     if isinstance(v, _Saveable):
        #         savables_ref.remove(v)
        # print('SAVEABLE_OBJECTS after', len(savables_ref), savables_ref)

        last_step = session.run(self._global_step_tensor)
        epoch = None
        if self._epoch_tensor is not None:
            epoch = session.run(self._epoch_tensor)

        if last_step != self._timer.last_triggered_step():
            self._save_step(session, last_step)

        self._save_epoch(session, last_step, epoch)

        for l in self._epoch_listeners:
            # _NewCheckpointListenerForEvaluate will run here at end
            l.end(session, last_step)

        for l in self._step_listeners:
            l.end(session, last_step)

    def _save_epoch(self, session, step, epoch):
        """Saves the latest checkpoint, returns should_stop."""
        if epoch:
            save_path = '{}-{}'.format(self._epoch_save_path, epoch)
        else:
            save_path = self._epoch_save_path
        tf.logging.info(
            "Saving\033[1;31m epoch\033[0m checkpoints for %d into %s.", step,
            save_path
        )

        for l in self._epoch_listeners:
            l.before_save(session, step)

        self._get_epoch_saver().save(
            sess=session,
            save_path=save_path,
            global_step=step,
            latest_filename=self._epoch_latest_filename,
            meta_graph_suffix="meta",
            write_meta_graph=True,
            write_state=True,
            strip_default_attrs=False
        )

        should_stop = False
        for l in self._epoch_listeners:
            # _NewCheckpointListenerForEvaluate will not run here
            # since _is_first_run == True, it will run at end
            if l.after_save(session, step):
                tf.logging.info(
                    "An Epoch CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l)
                )
                should_stop = True
        return should_stop

    def _save_step(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        tf.logging.info(
            "Saving\033[1;31m step\033[0m checkpoints for %d into %s.", step,
            self._step_save_path
        )

        for l in self._step_listeners:
            l.before_save(session, step)

        saver = self._get_step_saver()

        saver.save(
            sess=session,
            save_path=self._step_save_path,
            global_step=step,
            # latest_filename=self._step_latest_filename,
            latest_filename=None,
            meta_graph_suffix="meta",
            write_meta_graph=True,
            write_state=True,
            strip_default_attrs=False
        )
        self._summary_writer.add_session_log(
            SessionLog(
                status=SessionLog.CHECKPOINT,
                checkpoint_path=self._step_save_path
            ),  # pylint: disable=no-member
            step
        )

        should_stop = False
        for l in self._step_listeners:
            if l.after_save(session, step):
                tf.logging.info(
                    "A Step CheckpointSaverListener requested that training be stopped. "
                    "listener: {}".format(l)
                )
                should_stop = True
        return should_stop

    def _save(self, session, step):
        """Saves the latest checkpoint, returns should_stop."""
        return self._save_step(session, step)

    def _get_epoch_saver(self):
        return self._epoch_saver

    def _get_step_saver(self):
        if self._saver is not None:
            return self._saver
        elif self._scaffold is not None:
            return self._scaffold.saver

        # Get saver from the SAVERS collection if present.
        collection_key = tf.GraphKeys.SAVERS
        savers = tf.get_collection(collection_key)
        if not savers:
            raise RuntimeError(
                "No items in collection {}. Please add a saver to the collection "
                "or provide a saver or scaffold.".format(collection_key)
            )
        elif len(savers) > 1:
            raise RuntimeError(
                "More than one item in collection {}. "
                "Please indicate which one to use by passing it to the constructor."
                .format(collection_key)
            )

        self._saver = savers[0]
        return savers[0]

    def _get_saver(self):
        return self._get_step_saver()


orig_stdout = sys.stdout


class EpochProgressBarHook(tf.train.SessionRunHook):
    def __init__(
        self,
        total,
        initial_tensor,
        n_tensor,
        postfix_tensors=None,
        every_n_iter=None
    ):
        self._total = total
        self._initial_tensor = initial_tensor
        self._n_tensor = n_tensor
        self._postfix_tensors = postfix_tensors
        self._every_n_iter = every_n_iter

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        pass

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
        session: A TensorFlow Session that has been created.
        coord: A Coordinator object which keeps track of all threads.
        """
        initial = session.run(self._initial_tensor)
        epoch = initial // self._total
        epoch_initial = initial % self._total
        # print('after_create_session', initial, epoch)
        # setup progressbar
        self.pbar = tqdm(
            total=self._total,
            unit='seq',
            desc='Epoch {}'.format(epoch),
            mininterval=0.1,
            maxinterval=10.0,
            miniters=None,
            file=orig_stdout,
            dynamic_ncols=True,
            smoothing=0,
            bar_format=None,
            initial=epoch_initial,
            postfix=None
        )

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """
        return tf.train.SessionRunArgs(self._n_tensor)

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
        run_context: A `SessionRunContext` object.
        run_values: A SessionRunValues object.
        """
        # print('run_values', run_values.results)
        # update progressbar
        self.pbar.update(run_values.results)

    def end(self, session):
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
        session: A TensorFlow Session that will be soon closed.
        """
        self.pbar.close()


class EvalProgressBarHook(tf.train.SessionRunHook):
    def __init__(
        self, total, n_tensor, postfix_tensors=None, every_n_iter=None
    ):
        self._total = total
        self._n_tensor = n_tensor

    def begin(self):
        """Called once before using the session.

        When called, the default graph is the one that will be launched in the
        session.  The hook can modify the graph by adding new operations to it.
        After the `begin()` call the graph will be finalized and the other callbacks
        can not modify the graph anymore. Second call of `begin()` on the same
        graph, should not change the graph.
        """
        pass

    def after_create_session(self, session, coord):
        """Called when new TensorFlow session is created.

        This is called to signal the hooks that a new session has been created. This
        has two essential differences with the situation in which `begin` is called:

        * When this is called, the graph is finalized and ops can no longer be added
            to the graph.
        * This method will also be called as a result of recovering a wrapped
            session, not only at the beginning of the overall session.

        Args:
        session: A TensorFlow Session that has been created.
        coord: A Coordinator object which keeps track of all threads.
        """
        # print('after_create_session', initial, epoch)
        # setup progressbar
        self.pbar = tqdm(
            total=self._total,
            unit='seq',
            desc='Eval',
            mininterval=0.1,
            maxinterval=10.0,
            miniters=None,
            file=orig_stdout,
            dynamic_ncols=True,
            smoothing=0,
            bar_format=None,
            initial=0,
            postfix=None
        )

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().

        You can return from this call a `SessionRunArgs` object indicating ops or
        tensors to add to the upcoming `run()` call.  These ops/tensors will be run
        together with the ops/tensors originally passed to the original run() call.
        The run args you return can also contain feeds to be added to the run()
        call.

        The `run_context` argument is a `SessionRunContext` that provides
        information about the upcoming `run()` call: the originally requested
        op/tensors, the TensorFlow Session.

        At this point graph is finalized and you can not add ops.

        Args:
        run_context: A `SessionRunContext` object.

        Returns:
        None or a `SessionRunArgs` object.
        """
        return tf.train.SessionRunArgs(self._n_tensor)

    def after_run(self, run_context, run_values):
        """Called after each call to run().

        The `run_values` argument contains results of requested ops/tensors by
        `before_run()`.

        The `run_context` argument is the same one send to `before_run` call.
        `run_context.request_stop()` can be called to stop the iteration.

        If `session.run()` raises any exceptions then `after_run()` is not called.

        Args:
        run_context: A `SessionRunContext` object.
        run_values: A SessionRunValues object.
        """
        # print('run_values', run_values.results)
        # update progressbar
        self.pbar.update(run_values.results)

    def end(self, session):
        """Called at the end of session.

        The `session` argument can be used in case the hook wants to run final ops,
        such as saving a last checkpoint.

        If `session.run()` raises exception other than OutOfRangeError or
        StopIteration then `end()` is not called.
        Note the difference between `end()` and `after_run()` behavior when
        `session.run()` raises OutOfRangeError or StopIteration. In that case
        `end()` is called but `after_run()` is not called.

        Args:
        session: A TensorFlow Session that will be soon closed.
        """
        self.pbar.close()


class ColoredLoggingTensorHook(tf.train.LoggingTensorHook):
    """Prints the given tensors every N local steps, every N seconds, or at end.

    The tensors will be printed to the log, with `INFO` severity. If you are not
    seeing the logs, you might want to add the following line after your imports:

    ```python
      tf.logging.set_verbosity(tf.logging.INFO)
    ```

    Note that if `at_end` is True, `tensors` should not include any tensor
    whose evaluation produces a side effect such as consuming additional inputs.
    """
    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(
            self._iter_count
        )
        if self._formatter:
            if elapsed_secs is not None:
                tf.logging.info(
                    "%s (%.3f sec)", self._formatter(tensor_values),
                    elapsed_secs
                )
            else:
                tf.logging.info(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                stats.append("%s = %s" % (tag, tensor_values[tag]))
            if elapsed_secs is not None:
                tf.logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
            else:
                tf.logging.info("%s", ", ".join(stats))
        np.set_printoptions(**original)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class SaveEvaluationResultHook(tf.train.SessionRunHook):
    """Saves evaluation results to disk for external use.
    Saves one file per batch in JSON format
    Remove padding for each sequence example and save:
    * protien sequence data
    * correct class
    * correct class prediction rank
    * correct class prediction probability
    * rank 1 prediction class
    * rank 1 prediction probability
    * rank N prediction class
    * rank N prediction probability

    protein shape=(batch_size, sequence_length), dtype=int32
    labels shape=(batch_size, sequence_length), dtype=int32
    top_probs shape=(batch_size, sequence_length, predict_top_k), dtype=float32
    top_classes shape=(batch_size, sequence_length, predict_top_k), dtype=int32

    logits shape=(batch_size, sequence_length, num_classes), dtype=float32
    ```python
    import tensorflow as tf
    tf.enable_eager_execution()
    aa_list = ' FLIMVPAWGSTYQNCO*UHKRDEBZX-'
    predict_top_k = 2
    batch, length, depth = (2, 3, 5)
    protein = tf.cast(tf.random_uniform([batch, length]) * len(aa_list), dtype=tf.int32)
    labels = tf.cast(tf.random_uniform([batch, length]) * depth, dtype=tf.int32)
    logits = tf.random_uniform([batch, length, depth])
    all_probs = tf.nn.softmax(logits=logits, axis=-1, name='softmax_tensor')
    top_probs, top_classes = tf.nn.top_k(all_probs, predict_top_k)
    label_prob = tf.gather(all_probs, tf.expand_dims(labels, -1), batch_dims=-1)
    # label_rank = tf.gather(tf.contrib.framework.argsort(all_probs, direction='DESCENDING'), tf.expand_dims(labels, -1), batch_dims=-1)
    label_rank = tf.reshape(tf.where(tf.equal(tf.contrib.framework.argsort(all_probs, direction='DESCENDING'), tf.expand_dims(labels, -1)))[:,-1], tf.shape(labels))
    label_rank = tf.reshape(tf.where(tf.equal(top_classes, tf.expand_dims(labels, -1)))[:,-1], tf.shape(labels))

    ```
    """
    def __init__(
        self,
        # protein, lengths, labels, label_prob, label_rank,
        # top_classes, top_probs,
        tensors,
        model_dir,
        output_dir=None,
        output_prefix=None,
        output_format='json'
    ):
        """Initializes this hook.
        Args:
          protein: protien sequence data.
          lengths: sequence lengths.
          labels: correct class.
          label_prob: correct class prediction rank.
          label_rank: correct class prediction probability.
          top_classes: rank N prediction class.
          top_probs: rank N prediction probability.
          output_dir: The output directory to save evaluation files. default: ${model_dir}/eval-${global_step}
          output_prefix: The output filename which will be suffixed by the current
            eval step. default: ${model_dir}@${global_step}-${eval_step}.${output_format}
          output_format: default: json
        """
        # self._protein = protein
        # self._lengths = lengths
        # self._labels = labels
        # self._label_prob = label_prob
        # self._label_rank = label_rank
        # self._top_classes = top_classes
        # self._top_probs = top_probs
        self._tensors = tensors
        self._model_dir = model_dir
        self._output_dir = output_dir
        self._output_prefix = output_prefix
        self._output_format = output_format.lower()
        self._first_run = True

    def begin(self):
        # if self._protein is None:
        #     raise RuntimeError('The model did not define any protein.')
        # if not self._labels:
        #     raise RuntimeError('The model did not define any labels.')
        # if not self._label_prob:
        #     raise RuntimeError('The model did not define label_prob.')
        # if not self._label_rank:
        #     raise RuntimeError('The model did not define label_rank.')
        # if not self._top_classes:
        #     raise RuntimeError('The model did not define top_classes.')
        # if not self._top_probs:
        #     raise RuntimeError('The model did not define top_probs.')
        self._tensors['global_step'] = tf.train.get_global_step(
        )  # global step of checkpoint
        if self._tensors['global_step'] is None:
            raise RuntimeError(
                'Global step should be created to use SaveEvaluationResultHook.'
            )
        self._tensors['eval_step'
                     ] = tf.contrib.training.get_or_create_eval_step(
                     )  # a counter for the evaluation step

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return tf.train.SessionRunArgs(self._tensors)

    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        results = run_values.results
        global_step = results['global_step']
        del results['global_step']
        eval_step = results['eval_step']
        del results['eval_step']
        lengths = results['lengths'].tolist()
        del results['lengths']
        # generate default output_dir and output_prefix if needed
        if not self._output_dir:
            self._output_dir = str(
                Path(self._model_dir) / 'eval-{}'.format(global_step)
            )
        if not self._output_prefix:
            self._output_prefix = '{}@{}'.format(
                Path(self._model_dir).name, global_step
            )
        output_path = Path(self._output_dir) / '{}-{}.{}'.format(
            self._output_prefix, eval_step, self._output_format
        )
        if self._first_run:
            self._first_run = False
            # make sure directories exist
            output_path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
        # remove padding
        reslist = []
        for i in range(len(lengths)):
            res = {'length': lengths[i]}
            for k in results.keys():
                # print(i, k, lengths[i])
                res[k] = results[k][i][:lengths[i]].tolist()
            reslist.append(res)

        if self._output_format == 'json':
            with output_path.open(encoding='utf-8', mode='w') as f:  # pylint: disable=no-member
                json.dump(
                    reslist, f, indent=2, sort_keys=False, cls=NumpyEncoder
                )
            # e1 = json.loads(Path(r'4951306-1.json').read_text())
        elif self._output_format == 'msgpack':
            with output_path.open(mode='wb') as f:  # pylint: disable=no-member
                msgpack.dump(reslist, f)
            # e1 = msgpack.loads(Path(r'4951306-1.msgpack').read_bytes())
        elif self._output_format == 'msgpack.gz':
            with gzip.open(output_path, mode='wb') as f:  # pylint: disable=no-member
                msgpack.dump(reslist, f)
            # e1 = msgpack.loads(Path(r'4951306-1.msgpack').read_bytes())

    def end(self, session):
        tf.logging.info('Evaluation results saved to %s', self._output_dir)
        # if self._post_evaluation_fn is not None:
        #     self._post_evaluation_fn(self._current_step, self._output_path)


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""
    def __init__(
        self,
        learning_rate,
        weight_decay_rate=0.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=None,
        name="AdamWeightDecayOptimizer"
    ):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()
            )
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer()
            )

            # Standard Adam update.
            next_m = (
                tf.multiply(self.beta_1, m) +
                tf.multiply(1.0 - self.beta_1, grad)
            )
            next_v = (
                tf.multiply(self.beta_2, v) +
                tf.multiply(1.0 - self.beta_2, tf.square(grad))
            )

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)]
            )
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


# Transformer Layers
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :], d_model
    )
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
        output, attention_weights
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)
    # seq_len_k == seq_len_v
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


def gelu(x):
    """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (
        1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    )
    return x * cdf


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                dff,
                activation=gelu,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
            ),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(
                d_model,
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
            )  # (batch_size, seq_len, d_model)
        ]
    )


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(
            units=d_model,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        self.wk = tf.keras.layers.Dense(
            units=d_model,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )
        self.wv = tf.keras.layers.Dense(
            units=d_model,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        self.dropout = tf.keras.layers.Dropout(rate)
        self.dense = tf.keras.layers.Dense(
            d_model,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, training):
        batch_size = tf.shape(q)[0]

        v = self.wv(v)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        q = self.wq(q)  # (batch_size, seq_len, d_model)

        v = self.split_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len_v, depth)
        k = self.split_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len_k, depth)
        q = self.split_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len_q, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(
            q, k, transpose_b=True
        )  # (..., seq_len_q, seq_len_k)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -10000.0)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1
        )  # (..., seq_len_q, seq_len_k)
        # seq_len_k == seq_len_v
        attention_weights = self.dropout(attention_weights, training=training)
        scaled_attention = tf.matmul(
            attention_weights, v
        )  # (..., seq_len_q, depth_v)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)

        output = self.dense(
            concat_attention
        )  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, rate)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, mask):

        attn_output, _ = self.mha(
            inputs, inputs, inputs, mask, training=training
        )  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(
            inputs + attn_output
        )  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, rate)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-12)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, attend_to, training, inputs_mask, attend_to_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, _ = self.mha1(
            inputs, k=inputs, q=inputs, mask=inputs_mask, training=training
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs)

        attn2, _ = self.mha2(
            attend_to,
            k=attend_to,
            q=out1,
            mask=attend_to_mask,
            training=training
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(
            attn2 + out1
        )  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out2
        )  # (batch_size, target_seq_len, d_model)

        return out3


class DecoderLayer5(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer5, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, rate)
        self.title_mha = MultiHeadAttention(d_model, num_heads, rate)
        self.authors_mha = MultiHeadAttention(d_model, num_heads, rate)
        self.categories_mha = MultiHeadAttention(d_model, num_heads, rate)
        self.fields_mha = MultiHeadAttention(d_model, num_heads, rate)
        self.year_mha = MultiHeadAttention(d_model, num_heads, rate)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-12)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.title_dropout = tf.keras.layers.Dropout(rate)
        self.authors_dropout = tf.keras.layers.Dropout(rate)
        self.categories_dropout = tf.keras.layers.Dropout(rate)
        self.fields_dropout = tf.keras.layers.Dropout(rate)
        self.year_dropout = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training, inputs_mask, title, title_mask, authors, authors_mask, categories, categories_mask, fields, fields_mask, year, year_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, _ = self.mha1(
            inputs, k=inputs, q=inputs, mask=inputs_mask, training=training
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + inputs)

        title_attn, _ = self.title_mha(
            title,
            k=title,
            q=out1,
            mask=title_mask,
            training=training
        )  # (batch_size, target_seq_len, d_model)
        title_attn = self.title_dropout(title_attn, training=training)
        authors_attn, _ = self.authors_mha(
            authors,
            k=authors,
            q=out1,
            mask=authors_mask,
            training=training
        )  # (batch_size, target_seq_len, d_model)
        authors_attn = self.authors_dropout(authors_attn, training=training)
        categories_attn, _ = self.categories_mha(
            categories,
            k=categories,
            q=out1,
            mask=categories_mask,
            training=training
        )  # (batch_size, target_seq_len, d_model)
        categories_attn = self.categories_dropout(categories_attn, training=training)
        fields_attn, _ = self.fields_mha(
            fields,
            k=fields,
            q=out1,
            mask=fields_mask,
            training=training
        )  # (batch_size, target_seq_len, d_model)
        fields_attn = self.fields_dropout(fields_attn, training=training)
        year_attn, _ = self.year_mha(
            year,
            k=year,
            q=out1,
            mask=year_mask,
            training=training
        )  # (batch_size, target_seq_len, d_model)
        year_attn = self.year_dropout(year_attn, training=training)
        out2 = self.layernorm2(
            out1 + title_attn + authors_attn + categories_attn + fields_attn + year_attn
        )  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(
            ffn_output + out2
        )  # (batch_size, target_seq_len, d_model)

        return out3


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate=0.1
    ):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, self.d_model
        )

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        maximum_position_encoding,
        rate=0.1
    ):
        super(TransformerDecoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model
        )

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(inputs)[1]

        inputs = self.embedding(inputs)  # (batch_size, target_seq_len, d_model)
        inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        inputs += self.pos_encoding[:, :seq_len, :]
        inputs = self.dropout(inputs, training=training)

        for i in range(self.num_layers):
            inputs = self.dec_layers[i](
                inputs, enc_output, training, look_ahead_mask, padding_mask
            )

        # inputs.shape == (batch_size, target_seq_len, d_model)
        return inputs


def parse_aicup3_v16_tfrecords(serialized, mode, params):
    """Parse a single aicup task1 v1 record which is expected to be a tensorflow.Example."""
    features = {
        'sentence_lengths': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'length': tf.FixedLenFeature(shape=(), dtype=tf.string),
        # 'embeddings': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'title_pooled': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'meta_pooled': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'segment_ids': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'authors': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'categories': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'fields': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'year': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'input_ids': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'title_length': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'title_input_ids': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'meta_length': tf.FixedLenFeature(shape=(), dtype=tf.string),
        'meta_input_ids': tf.FixedLenFeature(shape=(), dtype=tf.string)
    }
    if mode != tf.estimator.ModeKeys.PREDICT:
        features['sentence_labels'] = tf.FixedLenFeature(
            shape=(), dtype=tf.string
        )
        features['labels'] = tf.FixedLenFeature(shape=(), dtype=tf.string)
        features['article_labels'] = tf.FixedLenFeature(
            shape=(), dtype=tf.string
        )
    parsed = tf.parse_single_example(
        serialized=serialized,
        # A scalar (0-D Tensor) of type string, a single binary
        # serialized `Example` proto.
        features=features,
        # A `dict` mapping feature keys to `FixedLenFeature` or
        # `VarLenFeature` values.
        example_names=None,
        #  A scalar string Tensor, the associated name (optional).
        name=None
        # A name for this operation (optional).
    )
    features = {}
    features['sentence_lengths'] = tf.decode_raw(
        parsed['sentence_lengths'],
        out_type=tf.int32,
        little_endian=True,
        name=None
    )
    features['sentence_sequence_length'] = tf.shape(
        input=features['sentence_lengths'], name=None, out_type=tf.int32
    )[0]
    features['length'] = tf.decode_raw(
        parsed['length'], out_type=tf.int32, little_endian=True, name=None
    )[0]
    # features['embeddings'] = tf.reshape(
    #     tf.decode_raw(
    #         parsed['embeddings'],
    #         out_type=tf.float32,
    #         little_endian=True,
    #         name=None
    #     ), [features['length'], -1]
    # )
    features['title_pooled'] = tf.decode_raw(
        parsed['title_pooled'],
        out_type=tf.float32,
        little_endian=True,
        name=None
    )
    features['meta_pooled'] = tf.decode_raw(
        parsed['meta_pooled'],
        out_type=tf.float32,
        little_endian=True,
        name=None
    )
    features['segment_ids'] = tf.cast(
        tf.decode_raw(
            parsed['segment_ids'],
            out_type=tf.uint8,
            little_endian=True,
            name=None
        ), tf.int32
    )
    features['authors'] = tf.decode_raw(
        parsed['authors'], out_type=tf.int32, little_endian=True, name=None
    )
    features['categories'] = tf.cast(
        tf.decode_raw(
            parsed['categories'],
            out_type=tf.uint8,
            little_endian=True,
            name=None
        ), tf.int32
    )
    features['fields'] = tf.cast(
        tf.decode_raw(
            parsed['fields'], out_type=tf.uint8, little_endian=True, name=None
        ), tf.int32
    )
    features['year'] = tf.cast(
        tf.decode_raw(
            parsed['year'], out_type=tf.uint8, little_endian=True, name=None
        ), tf.int32
    )  # 12
    features['input_ids'] = tf.decode_raw(
        parsed['input_ids'], out_type=tf.int32, little_endian=True, name=None
    )
    features['title_length'] = tf.decode_raw(
        parsed['title_length'],
        out_type=tf.int32,
        little_endian=True,
        name=None
    )[0]
    features['title_input_ids'] = tf.decode_raw(
        parsed['title_input_ids'],
        out_type=tf.int32,
        little_endian=True,
        name=None
    )
    features['meta_length'] = tf.decode_raw(
        parsed['meta_length'], out_type=tf.int32, little_endian=True, name=None
    )[0]
    features['meta_input_ids'] = tf.decode_raw(
        parsed['meta_input_ids'],
        out_type=tf.int32,
        little_endian=True,
        name=None
    )
    if mode != tf.estimator.ModeKeys.PREDICT:
        features['sentence_labels'] = tf.cast(
            tf.decode_raw(
                parsed['sentence_labels'],
                out_type=tf.uint8,
                little_endian=True,
                name=None
            ), tf.int32
        )
        labels = tf.cast(
            tf.decode_raw(parsed['labels'], out_type=tf.uint8), tf.int32
        )
        features['article_labels'] = tf.cast(
            tf.decode_raw(
                parsed['article_labels'],
                out_type=tf.uint8,
                little_endian=True,
                name=None
            ), tf.int32
        )[0]
        # sparse_softmax_cross_entropy_with_logits expects int32 or int64
        # tf.Tensor: shape=(sequence_length,), dtype=int32
        return features, labels
    else:
        return features


def input_fn(mode, params, config):
    """Estimator `input_fn`.
    Args:
      mode: Specifies if training, evaluation or
            prediction. tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
      params: model_params `dict` of hyperparameters.  Will receive what
              is passed to Estimator in `params` parameter. This allows
              to configure Estimators from hyper parameter tuning.positional_encoding
      config: run_config configuration object. Will receive what is passed
              to Estimator in `config` parameter, or the default `config`.
              Allows updating things in your `model_fn` based on
              configuration such as `num_ps_replicas`, or `model_dir`.
    Returns:
      A 'tf.data.Dataset' object
    """
    # the file names will be shuffled randomly during training
    dataset = tf.data.TFRecordDataset.list_files(
        file_pattern=params.tfrecord_pattern[mode],
        # A string or scalar string `tf.Tensor`, representing
        # the filename pattern that will be matched.
        shuffle=mode == tf.estimator.ModeKeys.TRAIN
        # If `True`, the file names will be shuffled randomly.
        # Defaults to `True`.
    )

    # Apply the interleave, prefetch, and shuffle first to reduce memory usage.

    # Preprocesses params.dataset_parallel_reads files concurrently and interleaves records from each file.
    def tfrecord_dataset(filename):
        return tf.data.TFRecordDataset(
            filenames=filename,
            # containing one or more filenames
            compression_type=None,
            # one of `""` (no compression), `"ZLIB"`, or `"GZIP"`.
            buffer_size=params.dataset_buffer * 1024 * 1024
            # the number of bytes in the read buffer. 0 means no buffering.
        )  # 256 MB

    dataset = dataset.interleave(
        map_func=tfrecord_dataset,
        # A function mapping a nested structure of tensors to a Dataset
        cycle_length=params.dataset_parallel_reads,
        # The number of input Datasets to interleave from in parallel.
        block_length=1,
        # The number of consecutive elements to pull from an input
        # `Dataset` before advancing to the next input `Dataset`.
        num_parallel_calls=None
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        if tfversion[0] == 1 and tfversion[1] <= 13:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(
                    buffer_size=params.shuffle_buffer,
                    # the maximum number elements that will be buffered when prefetching.
                    count=params.repeat_count
                    # the number of times the dataset should be repeated
                )
            )
        else:
            dataset = dataset.shuffle(buffer_size=params.shuffle_buffer)
            if params.repeat_count != 1:
                dataset = dataset.repeat(count=params.repeat_count)

    parse_fn = parse_aicup3_v16_tfrecords
    dataset = dataset.map(
        functools.partial(parse_fn, mode=mode, params=params),
        num_parallel_calls=int(params.num_cpu_threads / 2)
    )

    # Our inputs are variable length, so bucket, dynamic batch and pad them.
    # embed_dims = {
    #     'bert_uncased_L-12_H-768_A-12': 768,
    #     'bert_cased_L-12_H-768_A-12': 768,
    #     'bert_uncased_L-24_H-1024_A-16': 1024,
    #     'bert_cased_L-24_H-1024_A-16': 1024
    # }
    if mode != tf.estimator.ModeKeys.PREDICT:
        padded_shapes = (
            {
                # 'embeddings': [None, embed_dims[params.hub_model]],
                # 'title_embeddings': [768],
                'title_pooled': [768],
                'meta_pooled': [768],
                'length': [],
                'sentence_sequence_length': [],
                'sentence_lengths': [None],
                'sentence_labels': [None],
                'article_labels': [],
                'segment_ids': [None],
                'authors': [None],
                'categories': [None],
                'fields': [None],
                'year': [1],
                'input_ids': [None],
                'title_length': [],
                'title_input_ids': [None],
                'meta_length': [],
                'meta_input_ids': [None]
            },
            [None]
        )
        dataset = dataset.apply(
            tf.data.experimental.bucket_by_sequence_length(
                element_length_func=lambda features, labels: features['length'],
                bucket_boundaries=[2**x for x in range(5, 13)],
                # [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
                # [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
                bucket_batch_sizes=[
                    min(params.batch_size * 2**x, params.max_batch_size)
                    for x in range(8, -1, -1)
                ],
                padded_shapes=padded_shapes,
                padding_values=None,  # Defaults to padding with 0.
                pad_to_bucket_boundary=False
            )
        )
    else:
        padded_shapes = {
            'title_pooled': [768],
            'meta_pooled': [768],
            'length': [],
            'sentence_sequence_length': [],
            'sentence_lengths': [None],
            'segment_ids': [None],
            'authors': [None],
            'categories': [None],
            'fields': [None],
            'year': [1],
            'input_ids': [None],
            'title_length': [],
            'title_input_ids': [None],
            'meta_length': [],
            'meta_input_ids': [None]
        }
        dataset = dataset.padded_batch(
            params.predict_batch_size, padded_shapes=padded_shapes
        )

    dataset = dataset.prefetch(
        buffer_size=params.prefetch_buffer  # 64 batches
        # A `tf.int64` scalar `tf.Tensor`, representing the
        # maximum number batches that will be buffered when prefetching.
    )
    return dataset


class RunConfig(object):
  """RunConfig contains hyperparameters that could be different
  between pretraining and finetuning.
  These hyperparameters can also be changed from run to run.
  We store them separately from XLNetConfig for flexibility.
  """
  def __init__(self, is_training, use_tpu, use_bfloat16, dropout, dropatt,
               init="normal", init_range=0.1, init_std=0.02, mem_len=None,
               reuse_len=None, bi_data=False, clamp_len=-1, same_length=False):
    """
    Args:
      is_training: bool, whether in training mode.
      use_tpu: bool, whether TPUs are used.
      use_bfloat16: bool, use bfloat16 instead of float32.
      dropout: float, dropout rate.
      dropatt: float, dropout rate on attention probabilities.
      init: str, the initialization scheme, either "normal" or "uniform".
      init_range: float, initialize the parameters with a uniform distribution
        in [-init_range, init_range]. Only effective when init="uniform".
      init_std: float, initialize the parameters with a normal distribution
        with mean 0 and stddev init_std. Only effective when init="normal".
      mem_len: int, the number of tokens to cache.
      reuse_len: int, the number of tokens in the currect batch to be cached
        and reused in the future.
      bi_data: bool, whether to use bidirectional input pipeline.
        Usually set to True during pretraining and False during finetuning.
      clamp_len: int, clamp all relative distances larger than clamp_len.
        -1 means no clamping.
      same_length: bool, whether to use the same attention length for each token.
    """
    self.init = init
    self.init_range = init_range
    self.init_std = init_std
    self.is_training = is_training
    self.dropout = dropout
    self.dropatt = dropatt
    self.use_tpu = use_tpu
    self.use_bfloat16 = use_bfloat16
    self.mem_len = mem_len
    self.reuse_len = reuse_len
    self.bi_data = bi_data
    self.clamp_len = clamp_len
    self.same_length = same_length


def model_fn(features, labels, mode, params, config):
    # labels shape=(batch_size, sequence_length), dtype=int32
    is_train = mode == tf.estimator.ModeKeys.TRAIN

    # embeddings = features['embeddings']
    # embeddings shape=(batch_size, sequence_length, 768), dtype=float32
    # title_embeddings = features['title_embeddings']
    title_pooled = features['title_pooled']
    meta_pooled = features['meta_pooled']
    # embeddings shape=(batch_size, 768), dtype=float32
    lengths = features['length']
    # lengths shape=(batch_size, ), dtype=int32
    max_sentences = params.max_sentences
    sentence_sequence_lengths = features['sentence_sequence_length']
    # sentence_lengths = features['sentence_lengths']
    segment_ids = features['segment_ids']
    # segment_ids shape=(batch_size, sequence_length), dtype=int32
    authors = features['authors']
    categories = features['categories']
    fields = features['fields']
    year = features['year']
    input_ids = features['input_ids']
    title_lengths = features['title_length']
    title_input_ids = features['title_input_ids']
    meta_lengths = features['meta_length']
    meta_input_ids = features['meta_input_ids']
    global_step = tf.train.get_global_step()
    # global_step is assign_add 1 in tf.train.Optimizer.apply_gradients
    batch_size = tf.shape(lengths)[0]
    # number of sequences per epoch
    seq_total = batch_size
    if mode == tf.estimator.ModeKeys.TRAIN:
        seq_total = params.metadata['train']['articles']
    else:
        seq_total = params.metadata['test']['articles']

    task1_embeddings = tf.constant(params.metadata['task1_embeddings'])
    task2_embeddings = tf.constant(params.metadata['task2_embeddings'])
    if mode != tf.estimator.ModeKeys.PREDICT:
        sentence_labels = features['sentence_labels']
        sentence_labels_flat = tf.RaggedTensor.from_tensor(
            sentence_labels, padding=0
        ).flat_values
        sentence_classes = tf.nn.embedding_lookup(
            task1_embeddings, sentence_labels_flat
        )
        article_labels = features['article_labels']
        article_classes = tf.nn.embedding_lookup(
            task2_embeddings, article_labels
        )

    if params.use_tensor_ops:
        float_type = tf.float16
    else:
        float_type = tf.float32

    with tf.name_scope("masking"):
        if mode != tf.estimator.ModeKeys.PREDICT:
            word_maxlen = tf.shape(labels)[1]
        else:
            word_maxlen = tf.shape(input_ids)[1]
        mask = tf.sequence_mask(
            lengths=lengths, maxlen=word_maxlen, dtype=float_type
        )  # 0 if padding
        meta_maxlen = tf.shape(meta_input_ids)[1]
        meta_mask = tf.sequence_mask(
            lengths=meta_lengths, maxlen=meta_maxlen, dtype=tf.int32
        )  # 0 if padding
        title_maxlen = tf.shape(title_input_ids)[1]
        title_mask = tf.sequence_mask(
            lengths=title_lengths, maxlen=title_maxlen, dtype=tf.int32
        )  # 0 if padding
        sentence_maxlen = tf.reduce_max(sentence_sequence_lengths)
        sentence_mask = tf.sequence_mask(
            lengths=sentence_sequence_lengths, maxlen=sentence_maxlen, dtype=tf.float32
        )
        all_probs_mask = tf.expand_dims(sentence_mask, axis=2)
        word_padding_mask = tf.cast(tf.math.equal(mask, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
        sent_padding_mask = tf.cast(
            tf.math.equal(sentence_mask, 0), tf.float32
        )[:, tf.newaxis, tf.newaxis, :]
        meta_padding_mask = tf.cast(tf.math.equal(meta_mask, 0),
                                    tf.float32)[:, tf.newaxis, tf.newaxis, :]
        title_padding_mask = tf.cast(tf.math.equal(title_mask, 0),
                                     tf.float32)[:, tf.newaxis, tf.newaxis, :]
        authors_padding_mask = tf.cast(tf.math.equal(authors, 0),
                                     tf.float32)[:, tf.newaxis, tf.newaxis, :]
        categories_padding_mask = tf.cast(tf.math.equal(categories, 0),
                                     tf.float32)[:, tf.newaxis, tf.newaxis, :]
        fields_padding_mask = tf.cast(tf.math.equal(fields, 0),
                                     tf.float32)[:, tf.newaxis, tf.newaxis, :]
    # mask shape=(batch_size, sequence_length), dtype=float32

    ## ckpt method
    init_checkpoint_root = Path(params.init_checkpoint_root)
    xlnet_config_file = str(init_checkpoint_root / 'xlnet_config.json')
    xlnet_config = xlnet.XLNetConfig(json_path=xlnet_config_file)

    with tf.variable_scope('abstract'):
        if params.use_xlnet_zero_seg_ids:
            seg_id = tf.zeros_like(tf.transpose(segment_ids, [1, 0]), dtype=tf.int32) # int32
        else:
            seg_id = tf.transpose(segment_ids, [1, 0]) # int32
        inp = tf.transpose(input_ids, [1, 0]) # int32
        inp_mask = tf.transpose(tf.cast(tf.math.equal(mask, 0), tf.float32), [1, 0]) # float32
        run_config = RunConfig(
            is_training=is_train,
            use_tpu=False,
            use_bfloat16=False,
            dropout=params.xlnet_dropout,
            dropatt=params.xlnet_dropout
        )
        abstract_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=inp,
            seg_ids=seg_id,
            input_mask=inp_mask
        )
        abstract_features = tf.transpose(abstract_model.get_sequence_output(), [1, 0, 2])
        abstract_pooled = abstract_model.get_pooled_out('first', False)
        bert_hidden_size = abstract_features.shape[-1].value

    # init_from_checkpoint
    init_checkpoint = str(init_checkpoint_root / params.ckpt_name)
    tf.train.init_from_checkpoint(init_checkpoint, {'/': 'abstract/'})

    # with tf.variable_scope('title'):
    #     title_inp = tf.transpose(title_input_ids, [1, 0]) # int32
    #     title_inp_mask = tf.transpose(tf.cast(tf.math.equal(title_mask, 0), tf.float32), [1, 0]) # float32
    #     run_config = RunConfig(
    #         is_training=is_train,
    #         use_tpu=False,
    #         use_bfloat16=False,
    #         dropout=params.xlnet_dropout,
    #         dropatt=params.xlnet_dropout
    #     )
    #     title_model = xlnet.XLNetModel(
    #         xlnet_config=xlnet_config,
    #         run_config=run_config,
    #         input_ids=title_inp,
    #         seg_ids=tf.zeros_like(title_inp, tf.int32),
    #         input_mask=title_inp_mask
    #     )
    #     title_features = tf.transpose(title_model.get_sequence_output(), [1, 0, 2])
    # # init_from_checkpoint
    # tf.train.init_from_checkpoint(init_checkpoint, {'/': 'title/'})

    if params.attend_to == 'authors':
        with tf.variable_scope('authors', values=[authors]):
            authors_vocab = len(params.metadata['authors_categories'])  # 6216
            authors_features = tf.keras.layers.Embedding(
                input_dim=authors_vocab,
                output_dim=bert_hidden_size,  # 64
                embeddings_initializer=tf.truncated_normal_initializer(stddev=0.02),
                embeddings_regularizer=None,
                activity_regularizer=None,
                embeddings_constraint=None,
                mask_zero=False,
                input_length=None
            )(inputs=authors)
            # authors_features: shape=(batch_size, authors_length, 768), dtype=float32
        attend_to = authors_features
        attend_to_mask = authors_padding_mask
    elif params.attend_to == 'categories':
        with tf.variable_scope('categories', values=[categories]):
            categories_vocab = len(params.metadata['cats_categories'])  # 140
            categories_features = tf.keras.layers.Embedding(
                input_dim=categories_vocab,
                output_dim=bert_hidden_size,  # 256
                embeddings_initializer=tf.truncated_normal_initializer(stddev=0.02),
                embeddings_regularizer=None,
                activity_regularizer=None,
                embeddings_constraint=None,
                mask_zero=False,
                input_length=None
            )(inputs=categories)
            # categories_features: shape=(batch_size, categories_length, 768), dtype=float32
        attend_to = categories_features
        attend_to_mask = categories_padding_mask
    elif params.attend_to == 'fields':
        with tf.variable_scope('fields', values=[fields]):
            fields_vocab = len(params.metadata['fields_categories'])  # 22
            fields_features = tf.keras.layers.Embedding(
                input_dim=fields_vocab,
                output_dim=bert_hidden_size,  # 16
                embeddings_initializer=tf.truncated_normal_initializer(stddev=0.02),
                embeddings_regularizer=None,
                activity_regularizer=None,
                embeddings_constraint=None,
                mask_zero=False,
                input_length=None
            )(inputs=fields)
            # fields_features: shape=(batch_size, fields_length, 768), dtype=float32
        attend_to = fields_features
        attend_to_mask = fields_padding_mask
    elif params.attend_to == 'year':
        with tf.variable_scope('year', values=[year]):
            year_vocab = len(params.metadata['year_categories'])  # 6216
            year_features = tf.keras.layers.Embedding(
                input_dim=year_vocab,
                output_dim=bert_hidden_size,  # 768
                embeddings_initializer=tf.truncated_normal_initializer(stddev=0.02),
                embeddings_regularizer=None,
                activity_regularizer=None,
                embeddings_constraint=None,
                mask_zero=False,
                input_length=None
            )(inputs=year)
            # year_features shape=(batch_size, 1, 768), dtype=float32
        attend_to = year_features
        attend_to_mask = None
    else:
        attend_to = abstract_features
        attend_to_mask = word_padding_mask

    # with tf.variable_scope('input'):
        # catagorical_input = tf.concat(values=[year_features, authors_features, categories_features, fields_features], axis=1)
        # # catagorical_length = tf.shape(catagorical_input)[1]
        # catagorical_input = tf.keras.layers.LayerNormalization(
        #     axis=-1,
        #     epsilon=0.001,
        #     center=True,
        #     scale=True,
        #     beta_initializer='zeros',
        #     gamma_initializer='ones',
        #     beta_regularizer=None,
        #     gamma_regularizer=None,
        #     beta_constraint=None,
        #     gamma_constraint=None,
        #     trainable=True,
        #     name=None
        # )(inputs=catagorical_input)
        # catagorical_input = tf.keras.layers.Dropout(
        #     rate=params.embedded_dropout,  # 0.2
        #     noise_shape=None,
        #     seed=None,
        # )(inputs=catagorical_input, training=is_train)
        # if params.attend_to == 'title':
        #     attend_to = title_features
        #     attend_to_mask = title_padding_mask

    # bidirectional rnn
    with tf.variable_scope('word_decoder'):
        outputs = abstract_features
        # if params.use_transformer_positional_encoding:
        #     outputs += positional_encoding(max_sentences, params.sent_transformer_d_model)[:, :sentence_maxlen, :]
        for i in range(params.word_num_layers):
            outputs = DecoderLayer(
                d_model=params.word_d_model,
                num_heads=int(params.word_d_model / 64),
                dff=int(params.word_d_model * params.word_dff_x),
                rate=params.word_dropout_rate
            )(
                training=is_train,
                inputs=outputs,
                inputs_mask=word_padding_mask,
                # attend_to=catagorical_input,
                # attend_to_mask=None
                attend_to=attend_to,
                attend_to_mask=attend_to_mask
            )
            # (batch_size, input_seq_len, d_model)
            # outputs = DecoderLayer5v4(
            #     d_model=params.word_d_model,
            #     num_heads=int(params.word_d_model / 64),
            #     dff=int(params.word_d_model * params.word_dff_x),
            #     rate=params.word_dropout_rate
            # )(
            #     training=is_train,
            #     inputs=outputs,
            #     inputs_mask=word_padding_mask,
            #     title=title_features,
            #     title_mask=title_padding_mask,
            #     authors=authors_features,
            #     authors_mask=authors_padding_mask,
            #     categories=categories_features,
            #     categories_mask=categories_padding_mask,
            #     fields=fields_features,
            #     fields_mask=fields_padding_mask,
            #     year=year_features,
            #     year_mask=None
            # )
            # (batch_size, input_seq_len, d_model)

    with tf.variable_scope('sentence_pooling'):
        word_outputs = outputs
        sentence_inputs = tf.reshape(
            tensor=tf.map_fn(
                lambda i: tf.concat(
                    [
                        tf.slice(
                            [
                                tf.nn.relu(tf.reduce_max(s, axis=0))
                                for s in tf.dynamic_partition(
                                    data=tf.slice(
                                        outputs[i], [0, 0],
                                        [lengths[i],
                                         tf.shape(outputs)[-1]]
                                    ),
                                    partitions=tf.
                                    slice(segment_ids[i], [0], [lengths[i]]),
                                    num_partitions=max_sentences
                                )
                            ], [0, 0],
                            [sentence_maxlen,
                             tf.shape(outputs)[-1]]
                        )
                    ],
                    axis=1
                ), tf.range(batch_size), tf.float32
            ),
            shape=[batch_size, sentence_maxlen, outputs.shape[-1]]
        )

    # bidirectional rnn
    with tf.variable_scope('sent_decoder'):
        outputs = sentence_inputs
        if params.sent_num_layers > 0 and params.use_transformer_positional_encoding:
            outputs += positional_encoding(max_sentences, params.sent_d_model
                                          )[:, :sentence_maxlen, :]
        for i in range(params.sent_num_layers):
            outputs = DecoderLayer(
                d_model=params.sent_d_model,
                num_heads=int(params.sent_d_model / 64),
                dff=int(params.sent_d_model * params.sent_dff_x),
                rate=params.sent_dropout_rate
            )(
                training=is_train,
                inputs=outputs,
                inputs_mask=sent_padding_mask,
                # attend_to=catagorical_input,
                # attend_to_mask=None
                attend_to=attend_to,
                attend_to_mask=attend_to_mask
            )
            # (batch_size, input_seq_len, d_model)
            # outputs = DecoderLayer5v4(
            #     d_model=params.sent_d_model,
            #     num_heads=int(params.sent_d_model / 64),
            #     dff=int(params.sent_d_model * params.sent_dff_x),
            #     rate=params.sent_dropout_rate
            # )(
            #     training=is_train,
            #     inputs=outputs,
            #     inputs_mask=sent_padding_mask,
            #     title=title_features,
            #     title_mask=title_padding_mask,
            #     authors=authors_features,
            #     authors_mask=authors_padding_mask,
            #     categories=categories_features,
            #     categories_mask=categories_padding_mask,
            #     fields=fields_features,
            #     fields_mask=fields_padding_mask,
            #     year=year_features,
            #     year_mask=None
            # )
            # (batch_size, input_seq_len, d_model)

    # dense layer
    with tf.variable_scope('sentence_dense'):
        if params.sent_dense_units[0] != None:
            for i, units in enumerate(params.sent_dense_units):  # [256,128,64]
                with tf.variable_scope(f'dense_{i}_{units}'):
                    outputs = tf.keras.layers.Dense(
                        units=units,
                        activation=gelu,
                        use_bias=True,
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=0.02
                        ),
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None
                    )(inputs=outputs)
                    outputs = tf.keras.layers.Dropout(
                        rate=params.dense_dropout,  # 0.2
                        noise_shape=None,
                        seed=None,
                    )(inputs=outputs, training=is_train)
                # logits shape=(batch_size, sequence_length, num_classes), dtype=float32
    
    # with tf.variable_scope('article_dense'):
    #     article_outputs = abstract_pooled
    #     if params.arti_dense_units[0] != None:
    #         for i, units in enumerate(params.arti_dense_units):  # [256,128,64]
    #             with tf.variable_scope(f'dense_{i}_{units}'):
    #                 article_outputs = tf.keras.layers.Dense(
    #                     units=units,
    #                     activation=gelu,
    #                     use_bias=True,
    #                     kernel_initializer=tf.truncated_normal_initializer(
    #                         stddev=0.02
    #                     ),
    #                     bias_initializer='zeros',
    #                     kernel_regularizer=None,
    #                     bias_regularizer=None,
    #                     activity_regularizer=None,
    #                     kernel_constraint=None,
    #                     bias_constraint=None
    #                 )(inputs=article_outputs)
    #                 article_outputs = tf.keras.layers.Dropout(
    #                     rate=params.dense_dropout,  # 0.2
    #                     noise_shape=None,
    #                     seed=None,
    #                 )(inputs=article_outputs, training=is_train)
    #             # logits shape=(batch_size, sequence_length, num_classes), dtype=float32

    # output layer
    with tf.variable_scope('output'):
        logits = tf.layers.dense(
            inputs=outputs,
            units=6,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            bias_initializer=tf.zeros_initializer(dtype=float_type),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name='dense',
            reuse=None
        )
        # logits shape=(batch_size, sequence_length, num_classes), dtype=float32
    # with tf.variable_scope('article_output'):
    #     article_logits = tf.layers.dense(
    #         inputs=article_outputs,
    #         units=4,
    #         activation=None,
    #         use_bias=True,
    #         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
    #         bias_initializer=tf.zeros_initializer(dtype=float_type),
    #         kernel_regularizer=None,
    #         bias_regularizer=None,
    #         activity_regularizer=None,
    #         kernel_constraint=None,
    #         bias_constraint=None,
    #         trainable=True,
    #         name='dense',
    #         reuse=None
    #     )
    #     # article_logits shape=(batch_size, 4), dtype=float32

    # loss
    if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope('loss'):
            losses = tf.nn.weighted_cross_entropy_with_logits(
                labels=tf.cast(
                    tf.nn.embedding_lookup(
                        task1_embeddings, sentence_labels
                    ), tf.float32
                ) * params.scale_label + (1 - params.scale_label) / 2.0,
                logits=tf.cast(logits, tf.float32),
                pos_weight=params.loss_pos_weight
            )
            masked_losses = losses * all_probs_mask
            # losses shape=(batch_size, sequence_length, 6), dtype=float32
            # average across batch_size and sequence_length
            loss = tf.reduce_sum(masked_losses) / \
                tf.cast(tf.reduce_sum(sentence_sequence_lengths), dtype=tf.float32)
        # # tf.summary.scalar('loss', loss)
        # with tf.variable_scope('article_loss'):
        #     article_losses = tf.nn.weighted_cross_entropy_with_logits(
        #         labels=tf.cast(
        #             tf.nn.embedding_lookup(
        #                 task2_embeddings, article_labels
        #             ), tf.float32
        #         ) * params.scale_label + (1 - params.scale_label) / 2.0,
        #         logits=tf.cast(article_logits, tf.float32),
        #         pos_weight=params.article_loss_pos_weight
        #     )
        #     loss += tf.reduce_sum(article_losses) / tf.cast(batch_size, dtype=tf.float32)

    # predictions
    #
    with tf.variable_scope('predictions'):
        predictions = {}
        all_probs = tf.sigmoid(x=logits, name='sigmoid') * all_probs_mask
        # all_probs shape=(batch_size, target_output_lengths, 6), dtype=float32
        predicted_sentence_class_scores = tf.RaggedTensor.from_tensor(
            all_probs, lengths=sentence_sequence_lengths, ragged_rank=1
        ).flat_values
        # predicted_sentence_class_scores shape=(number_sentences_in_batch, 6), dtype=float32
        predictions['predicted_sentence_class_scores'] = predicted_sentence_class_scores
        predicted_sentence_classes = tf.cast(
            predicted_sentence_class_scores > params.predict_threshold,
            tf.int32
        )
        predictions['predicted_sentence_classes'] = predicted_sentence_classes
        # predicted_article_class_scores = tf.sigmoid(x=article_logits, name='sigmoid')
        # predictions['predicted_article_class_scores'] = predicted_article_class_scores
        # predicted_article_classes = tf.cast(
        #     predicted_article_class_scores > params.article_predict_threshold,
        #     tf.int32
        # )
        # predictions['predicted_article_classes'] = predicted_article_classes
        
    # default saver is added in estimator._train_with_estimator_spec
    # training.Saver(
    #   sharded=True,
    #   max_to_keep=self._config.keep_checkpoint_max,
    #   keep_checkpoint_every_n_hours=(
    #       self._config.keep_checkpoint_every_n_hours),
    #   defer_build=True,
    #   save_relative_paths=True)
    scaffold = tf.train.Scaffold(
        saver=tf.train.Saver(
            sharded=False,
            max_to_keep=config.keep_checkpoint_max,
            keep_checkpoint_every_n_hours=(
                config.keep_checkpoint_every_n_hours
            ),
            defer_build=True,
            save_relative_paths=True
        )
    )

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions, # PREDICT
            export_outputs={ # DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default"
                # tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)
            },
            scaffold=scaffold,
            prediction_hooks=None
        )

    with tf.variable_scope(
        'batch_metrics', values=[sentence_classes, predicted_sentence_classes]
    ):
        # with tf.control_dependencies([print_op]):
        is_correct = tf.cast(
            tf.equal(sentence_classes, predicted_sentence_classes), tf.float32
        )
        num_values = tf.ones_like(is_correct)
        batch_accuracy = tf.math.divide(
            tf.reduce_sum(is_correct), tf.reduce_sum(num_values)
        )
        TP = tf.cast(
            tf.count_nonzero(predicted_sentence_classes * sentence_classes),
            tf.float32
        )
        FP = tf.cast(
            tf.count_nonzero(
                predicted_sentence_classes * (sentence_classes - 1)
            ), tf.float32
        )
        FN = tf.cast(
            tf.count_nonzero(
                (predicted_sentence_classes - 1) * sentence_classes
            ), tf.float32
        )
        batch_precision = tf.math.divide_no_nan(TP, (TP + FP))
        batch_recall = tf.math.divide_no_nan(TP, (TP + FN))
        batch_f1 = tf.math.divide_no_nan(
            2 * batch_precision * batch_recall,
            (batch_precision + batch_recall)
        )
        tf.summary.scalar('batch_accuracy', batch_accuracy)
        tf.summary.scalar('batch_precision', batch_precision)
        tf.summary.scalar('batch_recall', batch_recall)
        tf.summary.scalar('batch_f1', batch_f1)

    # with tf.variable_scope(
    #     'batch_metrics_article', values=[article_classes, predicted_article_classes]
    # ):
    #     # with tf.control_dependencies([print_op]):
    #     is_correct_article = tf.cast(
    #         tf.equal(article_classes, predicted_article_classes), tf.float32
    #     )
    #     num_values_article = tf.ones_like(is_correct_article)
    #     batch_accuracy_article = tf.math.divide(
    #         tf.reduce_sum(is_correct_article), tf.reduce_sum(num_values_article)
    #     )
    #     TP_article = tf.cast(
    #         tf.count_nonzero(predicted_article_classes * article_classes),
    #         tf.float32
    #     )
    #     FP_article = tf.cast(
    #         tf.count_nonzero(
    #             predicted_article_classes * (article_classes - 1)
    #         ), tf.float32
    #     )
    #     FN_article = tf.cast(
    #         tf.count_nonzero(
    #             (predicted_article_classes - 1) * article_classes
    #         ), tf.float32
    #     )
    #     batch_precision_article = tf.math.divide_no_nan(TP_article, (TP_article + FP_article))
    #     batch_recall_article = tf.math.divide_no_nan(TP_article, (TP_article + FN_article))
    #     batch_f1_article = tf.math.divide_no_nan(
    #         2 * batch_precision_article * batch_recall_article,
    #         (batch_precision_article + batch_recall_article)
    #     )
    #     tf.summary.scalar('batch_accuracy_article', batch_accuracy_article)
    #     tf.summary.scalar('batch_precision_article', batch_precision_article)
    #     tf.summary.scalar('batch_recall_article', batch_recall_article)
    #     tf.summary.scalar('batch_f1_article', batch_f1_article)

    if mode == tf.estimator.ModeKeys.EVAL:

        def metric_fn(labels, predictions, scores):
            metrics = {
                'auc': tf.metrics.auc(
                    labels=labels,
                    predictions=scores,
                ),
                'f1_score': tf.contrib.metrics.f1_score(
                    labels=labels,
                    predictions=scores,
                ),
                'f1_class': tf.contrib.metrics.f1_score(
                    labels=labels,
                    predictions=predictions,
                ),
                'precision': tf.metrics.precision(
                    labels=labels,
                    predictions=predictions,
                ),
                'recall': tf.metrics.recall(
                    labels=labels,
                    predictions=predictions,
                )
                # 'auc_article': tf.metrics.auc(
                #     labels=labels_article,
                #     predictions=scores_article,
                # ),
                # 'f1_score_article': tf.contrib.metrics.f1_score(
                #     labels=labels_article,
                #     predictions=scores_article,
                # ),
                # 'f1_class_article': tf.contrib.metrics.f1_score(
                #     labels=labels_article,
                #     predictions=predictions_article,
                # ),
                # 'precision_article': tf.metrics.precision(
                #     labels=labels_article,
                #     predictions=predictions_article,
                # ),
                # 'recall_article': tf.metrics.recall(
                #     labels=labels_article,
                #     predictions=predictions_article,
                # )
            }
            if params.eval_precision_recall_at_equal_thresholds == True:
                metrics['precision_recall_at_equal_thresholds'] = tf.contrib.metrics.precision_recall_at_equal_thresholds(
                    labels=tf.cast(labels, tf.bool),
                    predictions=scores,
                    weights=None,
                    num_thresholds=1001,
                    use_locking=True
                )
                # metrics['precision_recall_at_equal_thresholds_article'] = tf.contrib.metrics.precision_recall_at_equal_thresholds(
                #     labels=tf.cast(labels_article, tf.bool),
                #     predictions=scores_article,
                #     weights=None,
                #     num_thresholds=1001,
                #     use_locking=True
                # )
            return metrics

        eval_metric_ops = metric_fn(
            sentence_classes, predicted_sentence_classes,
            predicted_sentence_class_scores
            # article_classes, predicted_article_classes,
            # predicted_article_class_scores
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,  # EVAL, TRAIN
            eval_metric_ops=eval_metric_ops,  # EVAL
            scaffold=scaffold,
            evaluation_hooks=None
        )

    # optimizer list
    optimizers = {
        'adagrad':
            tf.train.AdagradOptimizer,
        'adam':
            lambda lr: tf.train.AdamOptimizer(lr, epsilon=params.adam_epsilon),
        # lambda lr: tf.train.AdamOptimizer(lr, epsilon=1e-08),
        'nadam':
            lambda lr: tf.contrib.opt.
            NadamOptimizer(lr, epsilon=params.adam_epsilon),
        'ftrl':
            tf.train.FtrlOptimizer,
        'momentum':
            lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9),
        'rmsprop':
            tf.train.RMSPropOptimizer,
        'sgd':
            tf.train.GradientDescentOptimizer,
    }

    # optimizer
    with tf.variable_scope('optimizer'):
        # clip_gradients = params.gradient_clipping_norm
        clip_gradients = adaptive_clipping_fn(
            std_factor=params.clip_gradients_std_factor,  # 2.
            decay=params.clip_gradients_decay,  # 0.95
            static_max_norm=params.clip_gradients_static_max_norm,  # 6.
            global_step=global_step,
            report_summary=True,
            epsilon=np.float32(1e-7),
            name=None
        )

        def learning_rate_warmup(
            global_step,
            warmup_steps,
            repeat_steps=0,
            start=0.01,
            warmup_schedule='exp'
        ):
            """Learning rate warmup multiplier."""
            local_step = global_step
            if repeat_steps > 0:
                local_step = global_step % repeat_steps
            if not warmup_steps:
                return tf.constant(1.)

            tf.logging.info(
                'Applying %s learning rate warmup for %d steps',
                warmup_schedule, warmup_steps
            )

            local_step = tf.cast(local_step, dtype=tf.float32)
            warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
            start = tf.cast(start, dtype=tf.float32)
            warmup = tf.constant(1.)
            if warmup_schedule == 'exp':
                warmup = tf.exp(tf.log(start) / warmup_steps
                               )**(warmup_steps - local_step)
            else:
                assert warmup_schedule == 'linear'
                warmup = (
                    (tf.constant(1.) - start) / warmup_steps
                ) * local_step + start
            return tf.where(local_step < warmup_steps, warmup, tf.constant(1.))

        decay = params.learning_rate_decay_fn.lower()
        lr_schedule = 1.0  # if not decay or decay == 'none':
        if decay == 'noisy_linear_cosine_decay':
            lr = tf.train.noisy_linear_cosine_decay(
                params.learning_rate,
                global_step,
                decay_steps=params.learning_rate_decay_steps,  # 27000000
                initial_variance=1.0,
                variance_decay=0.55,
                num_periods=0.5,
                alpha=0.0,
                beta=0.001,
                name=None
            )
        elif decay == 'exponential_decay':
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1.0,
                decay_steps=params.learning_rate_decay_steps,  # 27000000
                decay_rate=params.learning_rate_decay_rate,  # 0.95
                staircase=False,
                name=None
            )
            lr_schedule *= schedule(tf.math.maximum(global_step - params.warmup_steps, 0))
        if params.warmup_steps > 0:
            lr_warmup = learning_rate_warmup(
                global_step,
                warmup_steps=params.warmup_steps,  # 35000
                repeat_steps=params.warmup_repeat_steps,  # 0
                start=params.warmup_start_lr,  # 0.001,
                warmup_schedule=params.warmup_schedule,  # 'exp'
            )
        if params.layer_warmup_steps > 0:
            layer_warmup = learning_rate_warmup(
                global_step,
                warmup_steps=params.layer_warmup_steps,  # 35000
                repeat_steps=params.warmup_repeat_steps,  # 0
                start=params.warmup_start_lr,  # 0.001,
                warmup_schedule=params.warmup_schedule,  # 'exp'
            )
        # Add learning rate to summary
        lr = params.learning_rate * lr_warmup * lr_schedule
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('layer_schedule', layer_warmup * lr_schedule)
        tf.summary.scalar('lr_schedule', lr_warmup * lr_schedule)
        if params.optimizer == 'bertadam':
            optimizer = AdamWeightDecayOptimizer(
                learning_rate=lr,
                weight_decay_rate=0.01 * lr_warmup * lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "/bias"]
            )
            if params.use_fp16:
                optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
            
            # 0 abstract/model/transformer/r_w_bias:0' shape=(24, 16, 64)
            # 1 abstract/model/transformer/r_r_bias:0' shape=(24, 16, 64)
            # 2 abstract/model/transformer/word_embedding/lookup_table:0' shape=(32000, 1024)
            # 3 abstract/model/transformer/r_s_bias:0' shape=(24, 16, 64)
            # 4 abstract/model/transformer/seg_embed:0' shape=(24, 2, 16, 64)
            # 5 abstract/model/transformer/layer_0/rel_attn/q/kernel:0' shape=(1024, 16, 64)
            # 6 abstract/model/transformer/layer_0/rel_attn/k/kernel:0' shape=(1024, 16, 64)
            # 7 abstract/model/transformer/layer_0/rel_attn/v/kernel:0' shape=(1024, 16, 64)
            # 8 abstract/model/transformer/layer_0/rel_attn/r/kernel:0' shape=(1024, 16, 64)
            # 9 abstract/model/transformer/layer_0/rel_attn/o/kernel:0' shape=(1024, 16, 64)
            tvars = tf.trainable_variables()
            if params.train_seg_embed:
                tvars = tvars[4:5] + tvars[5 + params.freeze_layers * 13:]
            else:
                tvars = tvars[5 + params.freeze_layers * 13:]
            grads_and_vars = optimizer.compute_gradients(loss, tvars)
            grads_and_vars = [(g,v) for g,v in grads_and_vars if g is not None]
            grads, tvars = list(zip(*grads_and_vars))
            for i in range(7):
                print(i, '=======', tvars[i], '======')
            # This is how the model was pre-trained.
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            if params.layer_decay_rate != 1.0:
                n_layer = xlnet_config.n_layer
                for i in range(len(grads)):
                    for l in range(n_layer):
                        prefix = f'abstract/model/transformer/layer_{l}/'
                        if tvars[i].name[:len(prefix)] == prefix:
                            abs_rate = params.layer_decay_rate ** (n_layer - 1 - l)
                            grads[i] *= abs_rate * layer_warmup
                            # tf.logging.info("Apply mult {:.4f} to layer-{} grad of {}".format(rate, l, tvars[i].name))
                            break
                    
            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=global_step
            )
            # Normally the global step update is done inside of `apply_gradients`.
            # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
            # a different optimizer, you should probably take this line out.
            new_global_step = global_step + 1
            train_op = tf.group(train_op, [global_step.assign(new_global_step)])
        else:
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step,
                learning_rate=lr,  # 0.001
                optimizer=optimizers[params.optimizer.lower()],
                gradient_noise_scale=None,
                gradient_multipliers=None,
                # some gradient clipping stabilizes training in the beginning.
                # clip_gradients=clip_gradients,
                # clip_gradients=6.,
                # clip_gradients=None,
                # learning_rate_decay_fn=learning_rate_decay_fn,
                update_ops=None,
                variables=None,
                name=None,
                summaries=[
                    # 'gradients',
                    # 'gradient_norm',
                    'loss',
                    # 'learning_rate' # only added if learning_rate_decay_fn is not None
                ],
                colocate_gradients_with_ops=True,
                increment_global_step=True
            )

    group_inputs = [train_op]

    # runtime numerical checks
    if params.check_nans:
        checks = tf.add_check_numerics_ops()
        group_inputs = [checks]

    # update accuracy
    # group_inputs.append(metrics['accuracy'][1])

    # record total number of examples processed
    examples_processed = tf.get_variable(
        name='examples_processed',
        initializer=tf.cast(0, tf.int64),
        trainable=False,
        dtype=tf.int64,
        aggregation=tf.VariableAggregation.SUM
    )
    # print('examples_processed', examples_processed)
    group_inputs.append(
        tf.assign_add(
            examples_processed,
            tf.cast(batch_size, tf.int64),
            name='update_examples_processed'
        )
    )
    epoch = examples_processed // seq_total
    group_inputs.append(epoch)
    progress = examples_processed / seq_total - tf.cast(epoch, tf.float64)
    group_inputs.append(progress)

    train_op = tf.group(*group_inputs)

    # if params.debug:
    #     train_op = tf.cond(
    #         pred=tf.logical_or(
    #             tf.is_nan(tf.reduce_max(embeddings)),
    #             tf.equal(global_step, 193000)
    #         ),
    #         false_fn=lambda: train_op,
    #         true_fn=lambda: tf.Print(
    #             train_op,
    #             # data=[global_step, metrics['accuracy'][0], lengths, loss, losses, predictions['classes'], labels, mask, protein, embeddings],
    #             data=[global_step, batch_accuracy, lengths, loss, embeddings],
    #             message='## DEBUG LOSS: ',
    #             summarize=50000
    #         )
    #     )

    training_hooks = []
    # INFO:tensorflow:global_step/sec: 2.07549
    training_hooks.append(
        tf.train.StepCounterHook(
            output_dir=params.model_dir,
            every_n_steps=params.log_step_count_steps
        )
    )

    # INFO:tensorflow:accuracy = 0.16705106, examples = 15000, loss = 9.688441, step = 150 (24.091 sec)
    def logging_formatter(v):
        return 'TP:\033[1;32m {:5.0f}\033[0m, precision:\033[1;32m {:9.5%}\033[0m, recall:\033[1;32m {:9.5%}\033[0m, f1:\033[1;32m {:9.5%}\033[0m, accuracy:\033[1;32m {:9.5%}\033[0m, loss:\033[1;32m {:8.5f}\033[0m, lr:\033[1;32m {:8.5f}\033[0m, step:\033[1;32m {:7,d}\033[0m'.format(
            v['TP'], v['precision'], v['recall'], v['f1'], v['accuracy'],
            v['loss'], v['learning_rate'], v['step']
        )

    tensors = {
        'TP': TP,
        'precision': batch_precision,
        'recall': batch_recall,
        'f1': batch_f1,
        'accuracy': batch_accuracy,
        'loss': loss,
        'step': global_step,
        'learning_rate': lr
        # 'input_size': tf.shape(protein),
        # 'examples': examples_processed
    }
    # def logging_formatter(v):
    #     return 'TP:\033[1;32m {:5.0f}\033[0m, precision:\033[1;32m {:6.2%}/{:6.2%}\033[0m, recall:\033[1;32m {:6.2%}/{:6.2%}\033[0m, f1:\033[1;32m {:6.2%}/{:6.2%}\033[0m, accuracy:\033[1;32m {:6.2%}/{:6.2%}\033[0m, loss:\033[1;32m {:8.5f}\033[0m, lr:\033[1;32m {:10.8f}\033[0m, step:\033[1;32m {:6,d}\033[0m'.format(
    #         v['TP'], v['precision'], v['precision_article'], v['recall'], v['recall_article'], v['f1'], v['f1_article'], v['accuracy'], v['accuracy_article'],
    #         v['loss'], v['learning_rate'], v['step']
    #     )
    
    # tensors = {
    #     'TP': TP,
    #     'precision': batch_precision,
    #     'recall': batch_recall,
    #     'f1': batch_f1,
    #     'accuracy': batch_accuracy,
    #     'precision_article': batch_precision_article,
    #     'recall_article': batch_recall_article,
    #     'f1_article': batch_f1_article,
    #     'accuracy_article': batch_accuracy_article,
    #     'loss': loss,
    #     'step': global_step,
    #     'learning_rate': lr
    #     # 'input_size': tf.shape(protein),
    #     # 'examples': examples_processed
    # }
    # if is_train:
    #     tensors['epoch'] = epoch
    #     tensors['progress'] = progress

    training_hooks.append(
        ColoredLoggingTensorHook(
            tensors=tensors,
            every_n_iter=params.log_step_count_steps,
            at_end=False,
            formatter=logging_formatter
        )
    )
    training_hooks.append(
        EpochProgressBarHook(
            total=seq_total,
            initial_tensor=examples_processed,
            n_tensor=batch_size,
            postfix_tensors=None,
            every_n_iter=params.log_step_count_steps
        )
    )
    if params.trace:
        training_hooks.append(
            tf.train.ProfilerHook(
                save_steps=params.save_summary_steps,
                output_dir=params.model_dir,
                show_dataflow=True,
                show_memory=True
            )
        )
    training_hooks.append(
        EpochCheckpointInputPipelineHook(
            checkpoint_dir=params.model_dir,
            config=config,
            save_secs=None,  # 10m
            # save_secs=params.save_checkpoints_secs,  # 10m
            save_steps=params.save_checkpoints_steps,
            # save_steps=None,
        )
    )

    training_chief_hooks = []
    # # saving_listeners like _NewCheckpointListenerForEvaluate
    # # will be called on the first CheckpointSaverHook
    # training_chief_hooks.append(tf.train.CheckpointSaverHook(
    #     checkpoint_dir=params.model_dir,
    #     # effectively only save on start and end of MonitoredTrainingSession
    #     save_secs=30 * 24 * 60 * 60,
    #     save_steps=None,
    #     checkpoint_basename="model.epoch",
    #     saver=tf.train.Saver(
    #         sharded=False,
    #         max_to_keep=0,
    #         defer_build=False,
    #         save_relative_paths=True
    #     )
    # ))
    # # Add a second CheckpointSaverHook to save every save_checkpoints_secs
    # training_chief_hooks.append(tf.train.CheckpointSaverHook(
    #     checkpoint_dir=params.model_dir,
    #     save_secs=params.save_checkpoints_secs, # 10m
    #     save_steps=None,
    #     checkpoint_basename="model.step",
    #     scaffold=scaffold
    # ))
    training_chief_hooks.append(
        EpochCheckpointSaverHook(
            checkpoint_dir=params.model_dir,
            epoch_tensor=epoch,
            save_secs=None,  # 10m
            # save_secs=params.save_checkpoints_secs,  # 10m
            save_steps=params.save_checkpoints_steps,
            # save_steps=None,
            scaffold=scaffold
        )
    )

    # local training:
    # all_hooks=[
    # EpochCheckpointSaverHook, Added into training_chief_hooks in this model_fn
    # SummarySaverHook, # Added into chief_hooks in MonitoredTrainingSession()
    # _DatasetInitializerHook, # Added into worker_hooks in Estimator._train_model_default
    # NanTensorHook, # Added into worker_hooks in Estimator._train_with_estimator_spec
    # StepCounterHook, # Added into training_hooks in this model_fn
    # LoggingTensorHook,  # Added into training_hooks in this model_fn
    # EpochCheckpointInputPipelineHook # Added into training_hooks in this model_fn
    # ]

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,  # PREDICT
        loss=loss,  # EVAL, TRAIN
        train_op=train_op,  # TRAIN
        scaffold=scaffold,
        training_chief_hooks=training_chief_hooks,
        training_hooks=training_hooks
    )


# https://github.com/tensorflow/models/blob/69cf6fca2106c41946a3c395126bdd6994d36e6b/tutorials/rnn/quickdraw/train_model.py


def create_estimator_and_specs(run_config):
    """Creates an Estimator, TrainSpec and EvalSpec."""

    # build hyperparameters
    model_params = tf.contrib.training.HParams(
        command=FLAGS.command,
        model_dir=run_config.model_dir,
        model_dir_prefix=FLAGS.model_dir_prefix,
        tfrecord_pattern={
            tf.estimator.ModeKeys.TRAIN: FLAGS.training_data,
            tf.estimator.ModeKeys.EVAL: FLAGS.eval_data,
            tf.estimator.ModeKeys.PREDICT: FLAGS.predict_data
        },
        data_version=FLAGS.data_version,
        metadata_path=FLAGS.metadata_path,
        max_sentences=FLAGS.max_sentences,
        experiment_name=FLAGS.experiment_name,
        host_script_name=FLAGS.host_script_name,
        job=FLAGS.job,
        max_epochs=FLAGS.max_epochs,
        max_runs=FLAGS.max_runs,
        eval_predict_checkpoint=FLAGS.eval_predict_checkpoint,
        predict_prefix=FLAGS.predict_prefix,
        article_predict_prefix=FLAGS.article_predict_prefix,
        predict_sample_submission=FLAGS.predict_sample_submission,
        predict_sample_submission_article=FLAGS.predict_sample_submission_article,
        predict_threshold=FLAGS.predict_threshold,
        article_predict_threshold=FLAGS.article_predict_threshold,
        eval_dir=FLAGS.eval_dir,
        eval_prefix=FLAGS.eval_prefix,
        eval_format=FLAGS.eval_format,
        eval_level=FLAGS.eval_level,
        eval_precision_recall_at_equal_thresholds=FLAGS.
        eval_precision_recall_at_equal_thresholds,
        predict_top_k=FLAGS.predict_top_k,
        num_gpus=FLAGS.num_gpus,
        num_cpu_threads=FLAGS.num_cpu_threads,
        random_seed=FLAGS.random_seed,
        use_xla=FLAGS.use_xla,
        use_fp16=FLAGS.use_fp16,
        use_tensor_ops=FLAGS.use_tensor_ops,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_secs=FLAGS.save_checkpoints_secs,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        log_step_count_steps=FLAGS.log_step_count_steps,
        eval_delay_secs=FLAGS.eval_delay_secs,
        eval_throttle_secs=FLAGS.eval_throttle_secs,
        steps=FLAGS.steps,
        eval_steps=FLAGS.eval_steps,
        dataset_buffer=FLAGS.dataset_buffer,  # 256 MB
        dataset_parallel_reads=FLAGS.dataset_parallel_reads,  # 1
        shuffle_buffer=FLAGS.shuffle_buffer,  # 16 * 1024 examples
        repeat_count=FLAGS.repeat_count,  # -1 = Repeat the input indefinitely.
        batch_size=FLAGS.batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        max_batch_size=FLAGS.max_batch_size,
        prefetch_buffer=FLAGS.prefetch_buffer,  # batches
        dense_catagorical_fields=FLAGS.dense_catagorical_fields,  # [None]
        conv_catagorical_fields=FLAGS.conv_catagorical_fields,  # [0,1,2,3,4]
        sent_conv_catagorical_fields=FLAGS.
        sent_conv_catagorical_fields,  # [0,1,2,3,4]
        segment_id_input=FLAGS.segment_id_input,  # [None]
        # hub_model=FLAGS.hub_model,  #
        init_checkpoint_root=FLAGS.init_checkpoint_root,  # 
        ckpt_name=FLAGS.ckpt_name,  # 
        title_init_checkpoint_root=FLAGS.title_init_checkpoint_root,  # 
        title_ckpt_name=FLAGS.title_ckpt_name,  # 
        title_dense_units=FLAGS.title_dense_units,  # 64
        authors_embed_dim=FLAGS.authors_embed_dim,  # 64
        authors_rnn_units=FLAGS.authors_rnn_units,  # 64
        categories_embed_dim=FLAGS.categories_embed_dim,  # 256
        categories_rnn_units=FLAGS.categories_rnn_units,  # 256
        fields_embed_dim=FLAGS.fields_embed_dim,  # 16
        fields_rnn_units=FLAGS.fields_rnn_units,  # 16
        input_norm=FLAGS.input_norm,  # 16
        rnn_norm=FLAGS.rnn_norm,  # 16
        dense_norm=FLAGS.dense_norm,  # 16
        embed_dim=FLAGS.embed_dim,  # 32
        use_xlnet_zero_seg_ids=FLAGS.use_xlnet_zero_seg_ids,  # 0.1
        xlnet_dropout=FLAGS.xlnet_dropout,  # 0.1
        embedded_dropout=FLAGS.embedded_dropout,  # 0.2
        feature_dropout=FLAGS.feature_dropout,  # 0.2
        conv_bank_size=FLAGS.conv_bank_size,  # 32
        conv_filters=FLAGS.conv_filters,  # 32
        conv_kernel_size=FLAGS.conv_kernel_size,  # 7
        conv_strides=FLAGS.conv_strides,  # 1
        conv_dropout=FLAGS.conv_dropout,  # 0.2
        use_conv_batch_norm=FLAGS.use_conv_batch_norm,
        use_conv_residual=FLAGS.use_conv_residual,
        use_rnn_residual=FLAGS.use_rnn_residual,
        use_dense_residual=FLAGS.use_dense_residual,
        use_conv_highway=FLAGS.use_conv_highway,
        conv_highway_depth=FLAGS.conv_highway_depth,
        conv_highway_units=FLAGS.conv_highway_units,
        rnn_cell_type=FLAGS.rnn_cell_type,
        rnn_num_units=FLAGS.rnn_num_units,  # list
        rnn_dropout=FLAGS.rnn_dropout,
        rnn_recurrent_dropout=FLAGS.rnn_recurrent_dropout,
        dense_units=FLAGS.dense_units,
        dense_dropout=FLAGS.dense_dropout,
        attend_to=FLAGS.attend_to,  # 12
        word_num_layers=FLAGS.word_num_layers,  # 12
        word_d_model=FLAGS.word_d_model,  # 768
        word_dff_x=FLAGS.word_dff_x,  # 4
        word_dropout_rate=FLAGS.word_dropout_rate,  # 0.1
        sent_num_layers=FLAGS.sent_num_layers,  # 12
        sent_d_model=FLAGS.sent_d_model,  # 768
        sent_dff_x=FLAGS.sent_dff_x,  # 4
        sent_dropout_rate=FLAGS.sent_dropout_rate,  # 0.1
        use_transformer_positional_encoding=FLAGS.
        use_transformer_positional_encoding,  # 3072
        sent_pool_abstract_features=FLAGS.sent_pool_abstract_features,
        sent_conv_bank_size=FLAGS.sent_conv_bank_size,
        sent_rnn_cell_type=FLAGS.sent_rnn_cell_type,
        sent_rnn_num_units=FLAGS.sent_rnn_num_units,
        sent_dense_units=FLAGS.sent_dense_units,
        arti_dense_units=FLAGS.arti_dense_units,
        use_crf=FLAGS.use_crf,  # True
        use_batch_renorm=FLAGS.use_batch_renorm,
        loss_type=FLAGS.loss_type,
        scale_label=FLAGS.scale_label,
        loss_pos_weight=FLAGS.loss_pos_weight,
        article_loss_pos_weight=FLAGS.article_loss_pos_weight,
        num_classes=FLAGS.num_classes,
        clip_gradients_std_factor=FLAGS.clip_gradients_std_factor,  # 2.
        clip_gradients_decay=FLAGS.clip_gradients_decay,  # 0.95
        # 6.
        clip_gradients_static_max_norm=FLAGS.clip_gradients_static_max_norm,
        no_bert_training_steps=FLAGS.no_bert_training_steps,  # 224
        learning_rate_decay_fn=FLAGS.learning_rate_decay_fn,
        learning_rate_decay_steps=FLAGS.learning_rate_decay_steps,  # 2000
        learning_rate_decay_rate=FLAGS.learning_rate_decay_rate,  # 0.7
        train_seg_embed=FLAGS.train_seg_embed,  # False
        freeze_layers=FLAGS.freeze_layers,  # 12
        layer_decay_rate=FLAGS.layer_decay_rate,  # 1.0
        layer_warmup_steps=FLAGS.layer_warmup_steps,  # 2000
        learning_rate=FLAGS.learning_rate,  # 0.001
        warmup_steps=FLAGS.warmup_steps,  # 35000 (10% epoch)
        warmup_repeat_steps=FLAGS.warmup_repeat_steps,  # 0
        warmup_start_lr=FLAGS.warmup_start_lr,  # 0.001
        warmup_schedule=FLAGS.warmup_schedule,  # exp
        optimizer=FLAGS.optimizer,
        adam_epsilon=FLAGS.adam_epsilon,
        check_nans=FLAGS.check_nans,
        trace=FLAGS.trace,
        debug=FLAGS.debug,
        metadata=FLAGS.metadata
    )

    # hook = tf_debug.LocalCLIDebugHook()

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params=model_params
    )

    # save model_params to model_dir/hparams.json
    hparams_path = Path(
        estimator.model_dir,
        'hparams-{:%Y-%m-%d-%H-%M-%S}.json'.format(datetime.datetime.now())
    )
    hparams_path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
    hparams_path.write_text(model_params.to_json(indent=2, sort_keys=False))

    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn,
        # A function that provides input data for training as minibatches.
        # max_steps=FLAGS.steps or None,  # 0
        max_steps=None,
        # Positive number of total steps for which to train model. If None, train forever.
        hooks=None
        # passed into estimator.train(hooks)
        # and then into _train_with_estimator_spec(hooks)
        # Iterable of `tf.train.SessionRunHook` objects to run
        # on all workers (including chief) during training.
        # CheckpointSaverHook? Not here, need only to run on cchief, put in
        # estimator_spec.training_chief_hooks
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn,
        # A function that constructs the input data for evaluation.
        steps=FLAGS.eval_steps,  # 10
        # Positive number of steps for which to evaluate model. If
        # `None`, evaluates until `input_fn` raises an end-of-input exception.
        name=None,
        # Name of the evaluation if user needs to run multiple
        # evaluations on different data sets. Metrics for different evaluations
        # are saved in separate folders, and appear separately in tensorboard.
        hooks=None,
        # Iterable of `tf.train.SessionRunHook` objects to run
        # during evaluation.
        exporters=None,
        # Iterable of `Exporter`s, or a single one, or `None`.
        # `exporters` will be invoked after each evaluation.
        start_delay_secs=10,
        # start_delay_secs=FLAGS.eval_delay_secs,  # 30 * 24 * 60 * 60
        # used for distributed training continuous evaluator only
        # Int. Start evaluating after waiting for this many seconds.
        throttle_secs=10
        # throttle_secs=FLAGS.eval_throttle_secs  # 30 * 24 * 60 * 60
        # full dataset at batch=4 currently needs 15 days
        # adds a StopAtSecsHook(eval_spec.throttle_secs)
        # Do not re-evaluate unless the last evaluation was
        # started at least this many seconds ago. Of course, evaluation does not
        # occur if no new checkpoints are available, hence, this is the minimum.
    )

    return estimator, train_spec, eval_spec


def main(unused_args):
    # setup colored logger
    coloredlogs.DEFAULT_FIELD_STYLES = dict(
        asctime=dict(color='green'),
        hostname=dict(color='magenta', bold=True),
        levelname=dict(color='black', bold=True),
        programname=dict(color='cyan', bold=True),
        name=dict(color='blue')
    )
    coloredlogs.DEFAULT_LEVEL_STYLES = dict(
        spam=dict(color='green', faint=True),
        debug=dict(color='green'),
        verbose=dict(color='blue'),
        info=dict(),
        notice=dict(color='magenta'),
        warning=dict(color='yellow'),
        success=dict(color='green', bold=True),
        error=dict(color='red'),
        critical=dict(color='red', bold=True)
    )

    if tfversion[0] == 1 and tfversion[1] <= 11:
        logger = tf_logging._get_logger()  # 1.11
    else:
        # >>> logging.getLogger().handlers
        # [<ABSLHandler (NOTSET)>]
        # >>> logger = tf.get_logger()
        # >>> logger.handlers
        # []
        logger = tf.get_logger()  # 1.12 # pylint: disable=no-member
        if len(logging.getLogger().handlers) != 0:
            # Remove ABSLHandler
            logging.getLogger().handlers.pop()
        if len(logger.handlers) == 0:
            # Add our own handler
            _handler = logging.StreamHandler(sys.stderr)
            _handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
            logger.addHandler(_handler)

    # print(Fore.RED + 'some red text' + Style.RESET_ALL, file=logger.handlers[0].stream)

    # set logger.handler.stream to output to our TqdmFile
    for h in logger.handlers:
        # <StreamHandler <stderr> (NOTSET)>
        # <StandardErrorHandler <stderr> (DEBUG)>
        # print(h)
        h.acquire()
        try:
            h.flush()
            orig_stdout = h.stream
            h.stream = TqdmFile(file=h.stream)
        finally:
            h.release()

    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.logging.debug('DEBUG==========test==========DEBUG')

    # check tfrecords data exists
    if FLAGS.job == 'train' and len(glob.glob(FLAGS.training_data)) == 0:
        msg = 'No training data files found for pattern: {}'.format(
            FLAGS.training_data
        )
        tf.logging.fatal(msg)
        raise IOError(msg)
    if FLAGS.job in {'eval', 'train'} and len(glob.glob(FLAGS.eval_data)) == 0:
        msg = 'No evaluation data files found for pattern: {}'.format(
            FLAGS.eval_data
        )
        tf.logging.fatal(msg)
        raise IOError(msg)
    if FLAGS.job == 'predict' and len(glob.glob(FLAGS.predict_data)) == 0:
        msg = 'No predict data files found for pattern: {}'.format(
            FLAGS.predict_data
        )
        tf.logging.fatal(msg)
        raise IOError(msg)
    if len(glob.glob(FLAGS.metadata_path)) == 0:
        msg = 'No metadata file found for pattern: {}'.format(
            FLAGS.metadata_path
        )
        tf.logging.fatal(msg)
        raise IOError(msg)

    # parse metadata
    FLAGS.metadata = None
    with open(FLAGS.metadata_path) as f:
        FLAGS.metadata = json.load(f)

    # read num_classes from metadata
    if not FLAGS.num_classes or FLAGS.num_classes < 1:
        # 35 for seq2seq, 33 for multilabel
        FLAGS.num_classes = len(FLAGS.metadata['task1_categories'])

    # set predict_top_k to num_classes if the given value doesn't make sense
    if not FLAGS.predict_top_k or FLAGS.predict_top_k < 1 or FLAGS.predict_top_k > FLAGS.num_classes:
        FLAGS.predict_top_k = FLAGS.num_classes

    # Hardware info
    FLAGS.num_gpus = FLAGS.num_gpus or tf.contrib.eager.num_gpus()
    FLAGS.num_cpu_threads = FLAGS.num_cpu_threads or os.cpu_count()

    # multi gpu distribution strategy
    distribution = None
    if FLAGS.num_gpus > 1:
        distribution = tf.contrib.distribute.MirroredStrategy(
            num_gpus=FLAGS.num_gpus
        )
        tf.logging.info('MirroredStrategy num_gpus: {}'.format(FLAGS.num_gpus))

    if FLAGS.eval_steps == -1:
        FLAGS.eval_steps = None

    # Set the seeds
    if FLAGS.random_seed == -1:
        FLAGS.random_seed = None
    else:
        np.random.seed(FLAGS.random_seed)
        if tfversion[0] == 1 and tfversion[1] <= 13:
            tf.set_random_seed(FLAGS.random_seed)
        else:
            tf.random.set_random_seed(FLAGS.random_seed)

    # Use JIT XLA
    # session_config = tf.ConfigProto(log_device_placement=True)
    session_config = tf.ConfigProto(allow_soft_placement=True)
    # default session config when init Estimator
    session_config.graph_options.rewrite_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE  # pylint: disable=no-member
    if FLAGS.use_xla:
        session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1  # pylint: disable=no-member
        # session_config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.NO_MEM_OPT

    if FLAGS.job == 'crossover':
        checkpoints = [c.strip() for c in FLAGS.crossover_checkpoints.split(',')]
        checkpoints = [c for c in checkpoints if c]
        num_chk = len(checkpoints)
        assert num_chk > 1, f'Provice 2 or more checkpoints to --crossover_checkpoints, {checkpoints}'
        crossover_meta = {
            'checkpoints': checkpoints,
            'best_threshold_f1': {}
        }
        var_shapes = tf.train.list_variables(checkpoints[0])
        var_dtypes = {}
        var_values = defaultdict(list)
        global_steps = []
        for checkpoint in checkpoints:
            tf.logging.info("Read from checkpoint %s", checkpoint)
            reader = tf.train.load_checkpoint(checkpoint)
            with tqdm(
                total=len(var_shapes),
                unit='vars',
                dynamic_ncols=True,
                ascii=True,
                smoothing=0.1,
                desc='read'
            ) as t:
                for (name, _) in var_shapes:
                    t.update()
                    if reader.has_tensor(name):
                        tensor = reader.get_tensor(name)
                        if name == 'global_step':
                            global_steps.append(tensor)
                        else:
                            var_dtypes[name] = tensor.dtype
                            var_values[name].append(tensor)
        del var_shapes
        # build crossover values
        print(f'build crossover weights: {len(var_values)}')
        crossover_weights = defaultdict(dict)
        with tqdm(
            total=len(var_values),
            unit='vars',
            dynamic_ncols=True,
            ascii=True,
            smoothing=0.1,
            desc='weights'
        ) as t:
            for name, values in var_values.items():
                t.update()
                if num_chk == 2:
                    w = FLAGS.crossover_weights
                    for i in np.arange(w[0], w[1], w[2]):
                        j = 100 - i
                        crossover_weights[f'{j:05.2f}-{i:05.2f}'] = [j,i]
                elif num_chk == 3:
                    # [[1, 1, 8], [1, 2, 7], [1, 3, 6], [1, 1, 1]]
                    i = 939.9
                    j = 60.1
                    for k in np.arange(250, 110, -10):
                        crossover_weights[f'{i/10:05.2f}-{j/10:05.2f}-{k/10:05.2f}'] = [i,j,k]
                else:
                    crossover_weights['avg'] = None
        # crossover_values = defaultdict(dict)
        # with tqdm(
        #     total=len(var_values),
        #     unit='vars',
        #     dynamic_ncols=True,
        #     ascii=True,
        #     smoothing=0.1,
        #     desc='build'
        # ) as t:
        #     for name, values in var_values.items():
        #         t.update()
        #         # crossover_values['min'][name] = np.min(values, axis=0)
        #         # crossover_values['max'][name] = np.max(values, axis=0)
        #         if num_chk == 2:
        #             w = FLAGS.crossover_weights
        #             for i in np.arange(w[0], w[1], w[2]):
        #                 j = 100 - i
        #                 if name[:9] == 'optimizer':
        #                     v = np.average(values, axis=0)
        #                 else:
        #                     v = np.average(values, axis=0, weights=[j,i])
        #                 crossover_values[f'{j:.2f}-{i:.2f}'][name] = v
        #         elif num_chk == 3:
        #             # [[1, 1, 8], [1, 2, 7], [1, 3, 6], [1, 1, 1]]
        #             i = 939.9
        #             j = 60.1
        #             for k in np.arange(250, 110, -10):
        #                 if name[:9] == 'optimizer':
        #                     v = np.average(values, axis=0)
        #                 else:
        #                     v = np.average(values, axis=0, weights=[i,j,k])
        #                 crossover_values[f'{i/10:.2f}-{j/10:.2f}-{k/10:.2f}'][name] = v
        #             # w = FLAGS.crossover_weights
        #             # for w in FLAGS.crossover_weights:
        #             #     for weights in sorted(set(list(itertools.permutations(w))), reverse=True):
        #             #         if name[:9] == 'optimizer':
        #             #             v = np.average(values, axis=0)
        #             #         else:
        #             #             v = np.average(values, axis=0, weights=weights)
        #             #         crossover_values['-'.join([str(s) for s in weights])][name] = v
        #         else:
        #             crossover_values['avg'][name] = np.average(values, axis=0)
        
        # prep variables
        print('prep variables')
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            tf_vars = [
                tf.get_variable(name, shape=var_values[name][0].shape, dtype=var_dtypes[name])
                for name in var_values
            ]
        del var_dtypes
        placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
        assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
        global_step = tf.Variable(max(global_steps), name="global_step", trainable=False, dtype=tf.int64)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=9999)
        
        # save crossover checkpoint and evaluate
        print('save crossover checkpoint and evaluate')
        crossover_output_prefix = verify_output_path(FLAGS.crossover_output_prefix)
        # for method, values in crossover_values.items():
        for method, weights in crossover_weights.items():
            crossover_values = {}
            with tqdm(
                total=len(var_values),
                unit='vars',
                dynamic_ncols=True,
                ascii=True,
                smoothing=0.1,
                desc=f'build-{method}'
            ) as t:
                for name, values in var_values.items():
                    t.update()
                    if name[:9] == 'optimizer':
                        v = np.average(values, axis=0)
                    else:
                        v = np.average(values, axis=0, weights=weights)
                    crossover_values[name] = v
            print(f'=== saving {method} ===')
            crossover_checkpoint = verify_output_path(crossover_output_prefix / method / method)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for p, assign_op, value in zip(placeholders, assign_ops, crossover_values.values()):
                    sess.run(assign_op, {p: value})
                # Use the built saver to save the averaged checkpoint.
                saver.save(sess, str(crossover_checkpoint))
            del crossover_values
            # evaluate
            model_dir = str(crossover_checkpoint.parent)
            print(f'=== evaluate {method}: {model_dir} ===')
            estimator, train_spec, eval_spec = create_estimator_and_specs(
                run_config=tf.estimator.RunConfig(
                    model_dir=model_dir,
                    tf_random_seed=FLAGS.random_seed,
                    save_summary_steps=None,
                    save_checkpoints_steps=None,
                    save_checkpoints_secs=None,
                    keep_checkpoint_max=9999,
                    log_step_count_steps=None,
                    session_config=session_config
                )
            )
            eval_result_metrics = estimator.evaluate(
                input_fn=eval_spec.input_fn,
                steps=None,
                hooks=eval_spec.hooks,
                checkpoint_path=str(crossover_checkpoint),
                name=f'evaluate-{method}'
            )
            # calculate best threshold
            precision = np.array(
                eval_result_metrics['precision_recall_at_equal_thresholds'][4]
            )
            recall = np.array(
                eval_result_metrics['precision_recall_at_equal_thresholds'][5]
            )
            thresholds = np.array(
                eval_result_metrics['precision_recall_at_equal_thresholds'][6]
            )
            eval_result_metrics['best_threshold_f1'] = max(
                zip(thresholds, 2 * precision * recall / (precision + recall)),
                key=lambda x: x[1]
            )
            print(f"=== best_threshold_f1 ({method}): ({eval_result_metrics['best_threshold_f1'][0]:.3f}, {eval_result_metrics['best_threshold_f1'][1]:.8f})")
            crossover_meta['best_threshold_f1'][method] = eval_result_metrics['best_threshold_f1']
            crossover_output_temp = crossover_output_prefix / f'{crossover_output_prefix.name}-crossover-meta-progress.json'
            with crossover_output_temp.open(encoding='utf-8', mode='w') as f:  # pylint: disable=no-member
                json.dump(
                    crossover_meta,
                    f,
                    indent=2,
                    sort_keys=False,
                    cls=NumpyEncoder
                )
        
        # save results
        print('save results')
        crossover_output = crossover_output_prefix / f'{crossover_output_prefix.name}-crossover-meta.json'
        with crossover_output.open(encoding='utf-8', mode='a') as f:  # pylint: disable=no-member
            json.dump(
                crossover_meta,
                f,
                indent=2,
                sort_keys=False,
                cls=NumpyEncoder
            )
        return
    
    epoch_checkpoint_re = re.compile(r'.*epoch-(\d+)-\d+$')
    while True:
        # figure out the model dir to use
        model_dir = FLAGS.model_dir
        if FLAGS.job == 'train':
            if FLAGS.model_dir:
                assert FLAGS.max_runs <= 1, f"set --model_dir_prefix instead of --model_dir for multiple runs"
            else:
                assert FLAGS.model_dir_prefix, f"set either --model_dir_prefix or --model_dir"
            run_i = 1
            while run_i <= FLAGS.max_runs or FLAGS.max_runs == -1:
                epoch_i = 0
                model_dir = FLAGS.model_dir or f"{FLAGS.model_dir_prefix}-{run_i:05d}"
                latest_checkpoint = tf.train.latest_checkpoint(model_dir, latest_filename='epoch.latest')
                if not latest_checkpoint:
                    break # got fresh run
                # parse epoch_i, error out if unparsable
                epoch_i = int(epoch_checkpoint_re.match(latest_checkpoint).group(1))
                if epoch_i < FLAGS.max_epochs or FLAGS.max_epochs == -1:
                    break # got partial run
                run_i += 1
                if FLAGS.model_dir:
                    tf.logging.info(f"Finished 1 run of {FLAGS.max_epochs} epochs!")
                    return
            else:
                tf.logging.info(f"Finished all {FLAGS.max_runs} runs of {FLAGS.max_epochs} epochs!")
                return
        assert model_dir, f"empty model_dir"
        # Parse experiment_name and host_script_name if needed
        # Example: --model_dir=${MODELDIR}/Attention_lr0.5_ws35000_${CARDTYPE}_${HOSTSCRIPT}.${NUMGPU}
        # experiment_name = Attention
        # host_script_name = ${HOSTSCRIPT}.${NUMGPU}
        model_dir_parts = Path(model_dir).name.split('_')
        if len(model_dir_parts) > 1:
            if FLAGS.experiment_name == 'PARSE':  # Default: Exp
                FLAGS.experiment_name = model_dir_parts[0]
            if FLAGS.host_script_name == 'PARSE':  # Default: tensorflow
                FLAGS.host_script_name = model_dir_parts[-1]
        # '\x1b[32m%(asctime)s,%(msecs)03d\x1b[0m \x1b[1;35m%(hostname)s\x1b[0m \x1b[34m%(name)s[%(process)d]\x1b[0m \x1b[1;30m%(levelname)s\x1b[0m %(message)s'
        coloredlogs.DEFAULT_LOG_FORMAT = f'\x1b[32m%(asctime)s,%(msecs)03d\x1b[0m \x1b[1;35m{FLAGS.experiment_name}\x1b[0m \x1b[34m{FLAGS.host_script_name}[%(process)d]\x1b[0m \x1b[1;30m%(levelname)s\x1b[0m %(message)s'
        coloredlogs.install(
            level='DEBUG',
            logger=logger,
            milliseconds=True,
            stream=logger.handlers[0].stream
        )

        estimator, train_spec, eval_spec = create_estimator_and_specs(
            run_config=tf.estimator.RunConfig(
                train_distribute=distribution,
                model_dir=model_dir,
                # Directory to save model parameters, graph and etc. This can
                # also be used to load checkpoints from the directory into a estimator to
                # continue training a previously saved model. If `PathLike` object, the
                # path will be resolved. If `None`, the model_dir in `config` will be used
                # if set. If both are set, they must be same. If both are `None`, a
                # temporary directory will be used.
                tf_random_seed=FLAGS.random_seed,  # 33
                # Random seed for TensorFlow initializers.
                # Setting this value allows consistency between reruns.
                save_summary_steps=FLAGS.save_summary_steps,  # 10
                # if not None, a SummarySaverHook will be added in MonitoredTrainingSession()
                # The frequency, in number of global steps, that the
                # summaries are written to disk using a default SummarySaverHook. If both
                # `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
                # the default summary saver isn't used. Default 100.
                save_checkpoints_steps=None,  # 100
                # Save checkpoints every this many steps.
                # save_checkpoints_secs=None,
                # We will define our own CheckpointSaverHook in EstimatorSpec.training_chief_hooks
                save_checkpoints_secs=FLAGS.save_checkpoints_secs,  # 10m
                # if not None, a CheckpointSaverHook will be added in MonitoredTrainingSession()
                # Save checkpoints every this many seconds with
                # CheckpointSaverHook. Can not be specified with `save_checkpoints_steps`.
                # Defaults to 600 seconds if both `save_checkpoints_steps` and
                # `save_checkpoints_secs` are not set in constructor.
                # If both `save_checkpoints_steps` and `save_checkpoints_secs` are None,
                # then checkpoints are disabled.
                keep_checkpoint_max=FLAGS.keep_checkpoint_max,  # 5
                # Maximum number of checkpoints to keep.  As new checkpoints
                # are created, old ones are deleted.  If None or 0, no checkpoints are
                # deleted from the filesystem but only the last one is kept in the
                # `checkpoint` file.  Presently the number is only roughly enforced.  For
                # example in case of restarts more than max_to_keep checkpoints may be
                # kept.
                keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,  # 6
                # keep an additional checkpoint
                # every `N` hours. For example, if `N` is 0.5, an additional checkpoint is
                # kept for every 0.5 hours of training, this is in addition to the
                # keep_checkpoint_max checkpoints.
                # Defaults to 10,000 hours.
                log_step_count_steps=None,  # Customized LoggingTensorHook defined in model_fn
                # if not None, a StepCounterHook will be added in MonitoredTrainingSession()
                # log_step_count_steps=FLAGS.log_step_count_steps,  # 10
                # The frequency, in number of global steps, that the
                # global step/sec will be logged during training.
                session_config=session_config
            )
        )

        if FLAGS.job == 'eval':
            if not FLAGS.eval_predict_checkpoint:
                FLAGS.eval_predict_checkpoint = tf.train.latest_checkpoint(
                    model_dir
                )
            tf.logging.info(
                'Evaluating checkpoint: %s', FLAGS.eval_predict_checkpoint
            )
            eval_result_metrics = estimator.evaluate(
                input_fn=eval_spec.input_fn,
                steps=
                None,  # Number of steps for which to evaluate model. If None, evaluates until input_fn raises an end-of-input exception.
                hooks=eval_spec.hooks,
                checkpoint_path=FLAGS.eval_predict_checkpoint,
                name='evaluate'
            )
            # calculate best threshold
            global_step = eval_result_metrics['global_step']
            precision = np.array(
                eval_result_metrics['precision_recall_at_equal_thresholds'][4]
            )
            recall = np.array(
                eval_result_metrics['precision_recall_at_equal_thresholds'][5]
            )
            thresholds = np.array(
                eval_result_metrics['precision_recall_at_equal_thresholds'][6]
            )
            eval_result_metrics['best_threshold'] = max(
                zip(thresholds, 2 * precision * recall / (precision + recall)),
                key=lambda x: x[1]
            )
            print('best_threshold:', eval_result_metrics['best_threshold'])
            # precision_article = np.array(
            #     eval_result_metrics['precision_recall_at_equal_thresholds_article'][4]
            # )
            # recall_article = np.array(
            #     eval_result_metrics['precision_recall_at_equal_thresholds_article'][5]
            # )
            # thresholds_article = np.array(
            #     eval_result_metrics['precision_recall_at_equal_thresholds_article'][6]
            # )
            # eval_result_metrics['best_threshold_article'] = max(
            #     zip(thresholds_article, 2 * precision_article * recall_article / (precision_article + recall_article)),
            #     key=lambda x: x[1]
            # )
            # print('best_threshold_article:', eval_result_metrics['best_threshold_article'])
            # # save eval results
            # if not FLAGS.eval_dir:
            #     FLAGS.eval_dir = str(
            #         Path(model_dir) / 'eval-{}'.format(global_step)
            #     )
            # if not FLAGS.eval_prefix:
            #     FLAGS.eval_prefix = '{}@{}'.format(
            #         Path(model_dir).name, global_step
            #     )
            # output_path = Path(FLAGS.eval_dir
            #                 ) / '{}-metrics.{}'.format(FLAGS.eval_prefix, 'json')
            # # assert dirs
            # output_path.parent.mkdir(parents=True, exist_ok=True)  # pylint: disable=no-member
            # with output_path.open(encoding='utf-8', mode='a') as f:  # pylint: disable=no-member
            #     json.dump(
            #         eval_result_metrics,
            #         f,
            #         indent=2,
            #         sort_keys=False,
            #         cls=NumpyEncoder
            #     )
        elif FLAGS.job == 'export':
            if not FLAGS.export_checkpoint:
                FLAGS.export_checkpoint = tf.train.latest_checkpoint(
                    model_dir
                )
            if not FLAGS.export_dir:
                FLAGS.export_dir = str(Path(model_dir) / 'export')
            tf.logging.info('Exporting checkpoint: %s', FLAGS.export_checkpoint)
            export_dir = estimator.export_saved_model(
                export_dir_base=FLAGS.export_dir,
                # serving_input_receiver_fn=serving_input_dataset_receiver_fn,
                serving_input_receiver_fn=serving_input_str_receiver_fn,
                assets_extra=None,
                as_text=False,
                checkpoint_path=FLAGS.export_checkpoint
            )
            tf.logging.info('Checkpoint exported to: %s', export_dir)
            # signature_def['serving_default']:
            # The given SavedModel SignatureDef contains the following input(s):
            #     inputs['protein_sequences'] tensor_info:
            #         dtype: DT_STRING
            #         shape: (-1)
            #         name: input_protein_string_tensor:0
            # The given SavedModel SignatureDef contains the following output(s):
            #     outputs['classes'] tensor_info:
            #         dtype: DT_INT32
            #         shape: (-1, -1)
            #         name: predictions/ArgMax:0
            #     outputs['top_classes'] tensor_info:
            #         dtype: DT_INT32
            #         shape: (-1, -1, 3)
            #         name: predictions/TopKV2:1
            #     outputs['top_probs'] tensor_info:
            #         dtype: DT_FLOAT
            #         shape: (-1, -1, 3)
            #         name: predictions/TopKV2:0
            # Method name is: tensorflow/serving/predict
        elif FLAGS.job == 'predict':
            if not FLAGS.eval_predict_checkpoint:
                FLAGS.eval_predict_checkpoint = tf.train.latest_checkpoint(
                    model_dir
                )
            tf.logging.info('Loading checkpoint: %s', FLAGS.eval_predict_checkpoint)
            predictions = estimator.predict(
                input_fn=input_fn,
                # predict_keys=[
                #     'predicted_sentence_classes', 'predicted_sentence_class_scores',
                #     'predicted_article_classes', 'predicted_article_class_scores'
                # ],
                predict_keys=[
                    'predicted_sentence_classes', 'predicted_sentence_class_scores'
                ],
                hooks=None,
                checkpoint_path=FLAGS.eval_predict_checkpoint,
                # yield_single_examples=True
                yield_single_examples=False
            )
            sentence_pred_list = []
            sentence_scores_list = []
            # article_pred_list = []
            # article_scores_list = []
            total = math.ceil(20000 / FLAGS.predict_batch_size)
            with tqdm(
                # total=131166,
                # unit='sentences',
                total=total,
                unit='batches',
                # total=FLAGS.metadata['test']['articles'],
                # unit='articles',
                dynamic_ncols=True,
                ascii=True,
                smoothing=0.1,
                desc='predictions'
            ) as t:
                for i, p in enumerate(predictions):
                    t.update()
                    # print(i, p)
                    # pred_list.append(p)
                    # sentence_pred_list.append(p['predicted_sentence_classes'].tolist())
                    # sentence_scores_list.append(
                    #     p['predicted_sentence_class_scores'].tolist()
                    # )
                    # article_pred_list.append(p['predicted_article_classes'].tolist())
                    # article_scores_list.append(
                    #     p['predicted_article_class_scores'].tolist()
                    # )
                    # if i > 10:
                    #     break
                    sentence_pred_list += p['predicted_sentence_classes'].tolist()
                    sentence_scores_list += p['predicted_sentence_class_scores'].tolist()
                    # article_pred_list += p['predicted_article_classes'].tolist()
                    # article_scores_list += p['predicted_article_class_scores'].tolist()
            # print(pred_list)
            # save predictions
            print('save predictions')
            output_path = Path(
                FLAGS.eval_predict_checkpoint + FLAGS.predict_prefix +
                '-predict-sentence.json'
            )
            with output_path.open(encoding='utf-8', mode='w') as f:  # pylint: disable=no-member
                json.dump(sentence_pred_list, f, indent=2, sort_keys=False, cls=NumpyEncoder)
            # output_path = Path(
            #     FLAGS.eval_predict_checkpoint + FLAGS.article_predict_prefix +
            #     '-predict-article.json'
            # )
            # with output_path.open(encoding='utf-8', mode='w') as f:  # pylint: disable=no-member
            #     json.dump(article_pred_list, f, indent=2, sort_keys=False, cls=NumpyEncoder)
            
            # save predictions
            print('save scores')
            output_path = Path(
                FLAGS.eval_predict_checkpoint + FLAGS.predict_prefix +
                '-scores-sentence.json'
            )
            with output_path.open(encoding='utf-8', mode='w') as f:  # pylint: disable=no-member
                json.dump(
                    sentence_scores_list, f, indent=2, sort_keys=False, cls=NumpyEncoder
                )
            # output_path = Path(
            #     FLAGS.eval_predict_checkpoint + FLAGS.article_predict_prefix +
            #     '-scores-article.json'
            # )
            # with output_path.open(encoding='utf-8', mode='w') as f:  # pylint: disable=no-member
            #     json.dump(
            #         article_scores_list, f, indent=2, sort_keys=False, cls=NumpyEncoder
            #     )

            # assert sentence count
            print('assert sentence count')
            # parse task1_sample_submission.csv
            sentence_count = 0
            predict_sample_submission_path = Path(FLAGS.predict_sample_submission)
            with predict_sample_submission_path.open(
                encoding='utf-8', mode='r'
            ) as f, tqdm(
                dynamic_ncols=True,
                ascii=True,
                desc=predict_sample_submission_path.name,
                unit='lines'
            ) as t:
                rows = csv.DictReader(f)
                for row in rows:
                    t.update()
                    article, sentence = row['order_id'].split('_')
                    if int(article[1:]) <= 20000:
                        sentence_count += 1
            print(
                f'Sentences in {predict_sample_submission_path.name}: {sentence_count}'
            )
            print(f'Sentences in sentence_pred_list: {len(sentence_pred_list)}')
            # assert
            if len(sentence_pred_list) != sentence_count:
                print(f"===DEBUG: len(sentence_pred_list) != sentence_count")

            # write submission.csv
            print('write submission.csv')
            predict_sample_submission_path = Path(FLAGS.predict_sample_submission)
            output_path = Path(
                FLAGS.eval_predict_checkpoint + FLAGS.predict_prefix +
                '-submission-sentence.csv'
            )
            with predict_sample_submission_path.open(
                encoding='utf-8', mode='r'
            ) as f, output_path.open(encoding='utf-8', mode='w',
                                    newline='') as out_f, tqdm(
                                        dynamic_ncols=True,
                                        ascii=True,
                                        smoothing=0.1,
                                        desc=predict_sample_submission_path.name,
                                        unit='lines'
                                    ) as t:
                rows = csv.DictReader(f)
                writer = csv.DictWriter(
                    out_f,
                    fieldnames=rows.fieldnames,
                    quoting=csv.QUOTE_MINIMAL,
                    lineterminator='\n'
                )
                writer.writeheader()
                for i, row in enumerate(rows):
                    t.update()
                    if i < len(sentence_pred_list):
                        predictions = sentence_pred_list[i]
                        row['BACKGROUND'] = str(predictions[0])
                        row['OBJECTIVES'] = str(predictions[1])
                        row['METHODS'] = str(predictions[2])
                        row['RESULTS'] = str(predictions[3])
                        row['CONCLUSIONS'] = str(predictions[4])
                        row['OTHERS'] = str(predictions[5])
                    writer.writerow(row)

            # write submission-article.csv
            # print('write submission-article.csv')
            # predict_sample_submission_article_path = Path(FLAGS.predict_sample_submission_article)
            # output_path = Path(
            #     FLAGS.eval_predict_checkpoint + FLAGS.article_predict_prefix +
            #     '-submission-article.csv'
            # )
            # with predict_sample_submission_article_path.open(
            #     encoding='utf-8', mode='r'
            # ) as f, output_path.open(encoding='utf-8', mode='w',
            #                         newline='') as out_f, tqdm(
            #                             dynamic_ncols=True,
            #                             ascii=True,
            #                             smoothing=0.1,
            #                             desc=predict_sample_submission_article_path.name,
            #                             unit='lines'
            #                         ) as t:
            #     rows = csv.DictReader(f)
            #     writer = csv.DictWriter(
            #         out_f,
            #         fieldnames=rows.fieldnames,
            #         quoting=csv.QUOTE_MINIMAL,
            #         lineterminator='\n'
            #     )
            #     writer.writeheader()
            #     for i, row in enumerate(rows):
            #         t.update()
            #         if i < len(article_pred_list):
            #             predictions = article_pred_list[i]
            #             row['THEORETICAL'] = str(predictions[0])
            #             row['ENGINEERING'] = str(predictions[1])
            #             row['EMPIRICAL'] = str(predictions[2])
            #             row['OTHERS'] = str(predictions[3])
            #         writer.writerow(row)
        elif FLAGS.job == 'train':
            while epoch_i < FLAGS.max_epochs or FLAGS.max_epochs == -1:
                tf.logging.info(f"Starting epoch {epoch_i+1}/{FLAGS.max_epochs} of run {run_i}/{FLAGS.max_runs}")
                eval_result_metrics, export_results = tf.estimator.train_and_evaluate(
                    estimator, train_spec, eval_spec
                )
                tf.logging.info(f"Finished epoch {epoch_i+1}/{FLAGS.max_epochs} of run {run_i}/{FLAGS.max_runs}")
                epoch_i += 1
            continue
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.register('type', 'list', lambda v: ast.literal_eval(v))

    parser.add_argument(
        '--model_dir',
        type=str,
        default='',
        help='Path for saving model checkpoints during training'
    )
    parser.add_argument(
        '--model_dir_prefix',
        type=str,
        default='',
        help='Path prefix for saving model checkpoints during training'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='PARSE',
        help=
        'Experiment name for logging purposes, if "PARSE", split model_dir by "_" and use the first part as the experiment name'
    )
    parser.add_argument(
        '--host_script_name',
        type=str,
        default='PARSE',
        help=
        'Host script name for logging purposes (8086K1-1.2), if "PARSE", split model_dir by "_" and use the last part as the host script name'
    )
    parser.add_argument(
        '--job',
        type=str,
        choices=['train', 'eval', 'predict', 'dataprep', 'export', 'crossover'],
        default='train',
        help='Set job type to run'
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=-1,  #
        help='Stop training and maybe start a new run after this many epochs. -1 to train forever'
    )
    parser.add_argument(
        '--max_runs',
        type=int,
        default=-1,  #
        help='Stop training after this many runs. -1 to train forever'
    )
    parser.add_argument(
        '--export_dir',
        type=str,
        default='',
        help=
        'Path for saving the exported SavedModel. Default: ${model_dir}/export'
    )
    parser.add_argument(
        '--export_checkpoint',
        type=str,
        default='',
        help=
        'Checkpoint to export to SavedModel, ex: "model_dir/step-380176". Default: latest checkpoint in model_dir'
    )
    parser.add_argument(
        '--eval_predict_checkpoint',
        type=str,
        default='',
        help=
        'Checkpoint to use for evaluation and prediction, ex: "model_dir/step-380176". Default: latest checkpoint in model_dir'
    )
    parser.add_argument(
        '--crossover_checkpoints',
        type=str,
        default='',
        help=
        'Checkpoints to use for crossover'
    )
    parser.add_argument(
        '--crossover_output_prefix',
        type=str,
        default='',
        help='string to add to crossover output filenames'
    )
    parser.add_argument(
        '--crossover_weights',
        type='list',
        default='[0,101,10]',
        help='crossover weights'
    )
    parser.add_argument(
        '--predict_prefix',
        type=str,
        default='',
        help='string to add to prediction output filenames'
    )
    parser.add_argument(
        '--article_predict_prefix',
        type=str,
        default='',
        help='string to add to article prediction output filenames'
    )
    parser.add_argument(
        '--predict_data',
        type=str,
        default='',
        help='Path to predict data (tf.Example in TFRecord format)'
    )
    parser.add_argument(
        '--predict_sample_submission',
        type=str,
        default='',
        help='Path to the task1_sample_submission.csv file.'
    )
    parser.add_argument(
        '--predict_sample_submission_article',
        type=str,
        default='',
        help='Path to the task2_sample_submission.csv file.'
    )
    parser.add_argument(
        '--predict_threshold',
        type=float,
        default=0.5,
        help='predict_threshold.'
    )
    parser.add_argument(
        '--article_predict_threshold',
        type=float,
        default=0.5,
        help='article_predict_threshold.'
    )
    parser.add_argument(
        '--eval_dir',
        type=str,
        default='',
        help=
        'Path for saving evaluation results. Default: ${model_dir}/eval-${global_step}'
    )
    parser.add_argument(
        '--eval_prefix',
        type=str,
        default='',
        help=
        'Filename prefix for evaluation results. Default: ${model_dir}@${global_step}'
    )
    parser.add_argument(
        '--eval_format',
        type=str,
        choices=['json', 'msgpack', 'msgpack.gz'],
        default='json',
        help=
        'File format of evaluation results, one of ["json", "msgpack", "msgpack.gz"]. Default: json'
    )
    parser.add_argument(
        '--eval_level',
        type=str,
        choices=['topk', 'min'],
        default='min',
        help=
        'Amount of data saved by evaluation, one of ["topk", "min"]. Default: min'
    )
    parser.add_argument(
        '--eval_precision_recall_at_equal_thresholds',
        type='bool',
        default='False',
        help='Output metrics precision_recall_at_equal_thresholds'
    )
    parser.add_argument(
        '--predict_top_k',
        type=int,
        default=3,  #
        help='Save only the top k most probable classes for each amino acid.'
    )
    parser.add_argument(
        '--training_data',
        type=str,
        # default='D:/datasets/pfam-regions-d0-s20/pfam-regions-d0-s20-train.tfrecords',
        default='D:/datasets/pfam-regions-d10-s20-train.tfrecords',
        help='Path to training data (tf.Example in TFRecord format)'
    )
    parser.add_argument(
        '--eval_data',
        type=str,
        # default='D:/datasets/pfam-regions-d0-s20/pfam-regions-d0-s20-test.tfrecords',
        default='D:/datasets/pfam-regions-d10-s20-test.tfrecords',
        help='Path to evaluation data (tf.Example in TFRecord format)'
    )
    parser.add_argument(
        '--data_version',
        type=str,
        default='v2',
        help=
        'Data format version of training and evaluation data. v1 uses SequenceExample, v2 uses Example'
    )
    parser.add_argument(
        '--metadata_path',
        type=str,
        # default='D:/datasets/pfam-regions-d0-s20/pfam-regions-d0-s20-test.tfrecords',
        default='',
        help='Path to metadata.json generated by prep_dataset'
    )
    parser.add_argument(
        '--max_sentences',
        type=int,
        default=50,  # current train test max is 26
        help='Maximum number of sentences per abstract.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        # default=16712 + 3, # 'PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'
        default=-1,  # 'PAD', 'NO_DOMAIN', 'UNKNOWN_DOMAIN'
        help='Number of domain classes.'
    )
    parser.add_argument(
        '--classes_file',
        type=str,
        default='',
        help='Path to a file with the classes - one class per line'
    )

    parser.add_argument(
        '--num_gpus',
        type=int,
        default=0,
        help='Number of GPUs to use, defaults to total number of gpus available.'
    )
    parser.add_argument(
        '--num_cpu_threads',
        type=int,
        default=0,
        help=
        'Number of CPU threads to use, defaults to half the number of hardware threads.'
    )
    parser.add_argument(
        '--random_seed', type=int, default=-1, help='The random seed.'
    )
    parser.add_argument(
        '--use_xla',
        type='bool',
        default='False',
        help='Whether to enable JIT XLA.'
    )
    parser.add_argument(
        '--use_fp16',
        type='bool',
        default='False',
        help='Whether to enable automatic mixed precision.'
    )
    parser.add_argument(
        '--use_tensor_ops',
        type='bool',
        default='False',
        help='Whether to use tensorcores or not.'
    )
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=100,
        help='Save summaries every this many steps.'
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100000,
        help='Save checkpoints every this many steps.'
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=30 * 60,
        help='Save checkpoints every this many seconds.'
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=1000,
        help='The maximum number of recent checkpoint files to keep.'
    )
    parser.add_argument(
        '--keep_checkpoint_every_n_hours',
        type=float,
        default=6,
        help='Keep an additional checkpoint every `N` hours.'
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=10,
        help=
        'The frequency, in number of global steps, that the global step/sec will be logged during training.'
    )
    parser.add_argument(
        '--eval_delay_secs',
        type=int,
        default=30 * 24 * 60 * 60,
        help=
        'Start distributed continuous evaluation after waiting for this many seconds. Not used in local training.'
    )
    parser.add_argument(
        '--eval_throttle_secs',
        type=int,
        default=30 * 24 * 60 * 60,
        help='Stop training and start evaluation after this many seconds.'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=0,  # 100000,
        help='Number of training steps, if 0 train forever.'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=-1,  # 100000,
        help='Number of evaluation steps, if 0, evaluates until end-of-input.'
    )

    parser.add_argument(
        '--dataset_buffer',
        type=int,
        default=256,
        help='Number of MB in the read buffer.'
    )
    parser.add_argument(
        '--dataset_parallel_reads',
        type=int,
        default=1,
        help='Number of input Datasets to interleave from in parallel.'
    )
    parser.add_argument(
        '--shuffle_buffer',
        type=int,
        default=16 * 1024,
        help='Maximum number elements that will be buffered when shuffling input.'
    )
    parser.add_argument(
        '--repeat_count',
        type=int,
        default=1,
        help='Number of times the dataset should be repeated.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help=
        'Batch size to use for longest sequence for training/evaluation. 1 if GPU Memory <= 6GB, 2 if <= 12GB'
    )
    parser.add_argument(
        '--predict_batch_size',
        type=int,
        default=128,
        help='Batch size to use for for prediction.'
    )
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=128,
        help='Max batch size for short sequences'
    )
    parser.add_argument(
        '--prefetch_buffer',
        type=int,
        default=64,
        help='Maximum number of batches that will be buffered when prefetching.'
    )

    parser.add_argument(
        '--conv_catagorical_fields',
        type='list',
        default='[0,2,3,4]',
        help=
        'Catagorical fields to feed into convolution bank, multiple choice: 0:"year", 1:"title_features", 2:"categories_features", 3:"fields_features", 4:"authors_features".'
    )
    parser.add_argument(
        '--dense_catagorical_fields',
        type='list',
        default='[0]',
        help=
        'Catagorical fields to feed into dense layer, multiple choice: 0:"year", 1:"title_features", 2:"categories_features", 3:"fields_features", 4:"authors_features".'
    )
    parser.add_argument(
        '--sent_conv_catagorical_fields',
        type='list',
        default='[0,1,2,3,4]',
        help=
        'Catagorical fields to add to sentence pooling, multiple choice: 0:"year", 1:"title_features", 2:"categories_features", 3:"fields_features", 4:"authors_features".'
    )
    parser.add_argument(
        '--segment_id_input',
        type='list',
        default='[2]',
        help=
        'Locations to concat segment_id, multiple choice: 1:"conv", 2:"rnn", 3:"dense".'
    )
    # parser.add_argument(
    #     '--hub_model',
    #     type=str,
    #     choices=[
    #         'bert_uncased_L-12_H-768_A-12', 'bert_uncased_L-24_H-1024_A-16', 'bert_cased_L-12_H-768_A-12'
    #     ],
    #     default='bert_uncased_L-12_H-768_A-12',
    #     help='Hub Model'
    # )
    parser.add_argument(
        '--init_checkpoint_root',
        type=str,
        default=r'/data12/tbrain/scibert_scivocab_uncased',
        help='Initial checkpoint (usually from a pre-trained BERT model).'
    )
    parser.add_argument(
        '--ckpt_name',
        type=str,
        default='bert_model.ckpt',
        help='checkpoint name.'
    )
    parser.add_argument(
        '--title_init_checkpoint_root',
        type=str,
        default=r'/data12/tbrain/biobert_v1.1_pubmed',
        help='Initial checkpoint (usually from a pre-trained BERT model).'
    )
    parser.add_argument(
        '--title_ckpt_name',
        type=str,
        default='model.ckpt-1000000',
        help='checkpoint name.'
    )
    parser.add_argument(
        '--title_dense_units',
        type=int,
        default=256,
        help='Number of units for authors embedding.'
    )
    parser.add_argument(
        '--authors_embed_dim',
        type=int,
        default=64,
        help='Number of units for authors embedding.'
    )
    parser.add_argument(
        '--authors_rnn_units',
        type=int,
        default=64,
        help='Number of units for authors rnn.'
    )
    parser.add_argument(
        '--categories_embed_dim',
        type=int,
        default=256,
        help='Number of units for authors embedding.'
    )
    parser.add_argument(
        '--categories_rnn_units',
        type=int,
        default=256,
        help='Number of units for authors rnn.'
    )
    parser.add_argument(
        '--fields_embed_dim',
        type=int,
        default=16,
        help='Number of units for authors embedding.'
    )
    parser.add_argument(
        '--fields_rnn_units',
        type=int,
        default=16,
        help='Number of units for authors rnn.'
    )
    parser.add_argument(
        '--input_norm',
        type=str,
        choices=['None', 'Batch', 'Layer'],
        default='Layer',
        help='Input Normalization'
    )
    parser.add_argument(
        '--rnn_norm',
        type=str,
        choices=['None', 'Batch', 'Layer'],
        default='Layer',
        help='RNN Normalization'
    )
    parser.add_argument(
        '--dense_norm',
        type=str,
        choices=['None', 'Batch', 'Layer'],
        default='None',
        help='Dense Normalization'
    )

    parser.add_argument(
        '--embed_dim', type=int, default=768, help='Embedding dimensions.'
    )
    parser.add_argument(
        '--use_xlnet_zero_seg_ids',
        type='bool',
        default='False',
        help='Use all zeros for segment_ids input.'
    )
    parser.add_argument(
        '--xlnet_dropout',
        type=float,
        default=0.1,
        help='Dropout rate used in xlnet layers.'
    )
    parser.add_argument(
        '--embedded_dropout',
        type=float,
        default=0.2,
        help='Dropout rate used after embedding layers.'
    )
    parser.add_argument(
        '--feature_dropout',
        type=float,
        default=0.4,
        help='Dropout rate used after feature rnn layers.'
    )

    parser.add_argument(
        '--conv_bank_size',
        type=int,
        default=15,
        help='Convolution bank kernal sizes 1 to bank_size.'
    )
    parser.add_argument(
        '--conv_filters',
        type=int,
        default=32,
        help='Number of convolution filters.'
    )
    parser.add_argument(
        '--conv_kernel_size',
        type=int,
        default=7,
        help='Length of the convolution filters.'
    )
    parser.add_argument(
        '--conv_strides',
        type=int,
        default=1,
        help=
        'The number of entries by which the filter is moved right at each step..'
    )
    parser.add_argument(
        '--conv_dropout',
        type=float,
        default=0.3,
        help='Dropout rate used for convolution layer outputs.'
    )
    parser.add_argument(
        '--use_conv_batch_norm',
        type='bool',
        default='True',
        help='Apply batch normalization after convolution layers.'
    )
    parser.add_argument(
        '--use_conv_residual',
        type='bool',
        default='True',
        help='Add residual connection after convolution layer 1.'
    )
    parser.add_argument(
        '--use_rnn_residual',
        type='bool',
        default='False',
        help='Add residual connection after rnn layer.'
    )
    parser.add_argument(
        '--use_dense_residual',
        type='bool',
        default='False',
        help='Add residual connection after dense layer.'
    )
    parser.add_argument(
        '--use_conv_highway',
        type='bool',
        default='True',
        help='Add a highway network after convolution layer 1.'
    )
    parser.add_argument(
        '--conv_highway_depth',
        type=int,
        default=3,
        help='Number of layers of highway network.'
    )
    parser.add_argument(
        '--conv_highway_units',
        type=int,
        default=1248,
        help='Number of units per layer of highway network.'
    )

    parser.add_argument(
        '--rnn_cell_type',
        type=str,
        choices=['LSTM', 'GRU', 'encoder', 'None'],
        default='GRU',
        help='RNN Cell Type'
    )
    parser.add_argument(
        '--rnn_num_units',
        type='list',
        default='[128]',
        help='Number of node per recurrent network layer.'
    )
    parser.add_argument(
        '--rnn_dropout',
        type=float,
        default=0.0,
        help='Dropout rate used between rnn layers.'
    )
    parser.add_argument(
        '--rnn_recurrent_dropout',
        type=float,
        default=0.0,
        help=
        'Dropout rate used between rnn timesteps, needs to be 0.0 to use cudnn.'
    )
    parser.add_argument(
        '--dense_units',
        type='list',
        # default='[256,128,64]',
        default='[None]',
        help='Number of units for the dense layers.'
    )
    parser.add_argument(
        '--dense_dropout',
        type=float,
        default=0.4,
        help='Dropout rate used after dense layers.'
    )
    parser.add_argument(
        '--attend_to',
        type=str,
        default='abstract',
        help='field to attend to'
    )
    parser.add_argument(
        '--word_num_layers',
        type=int,
        default=1,
        help='The number of transformer decoder layers.'
    )
    parser.add_argument(
        '--word_d_model',
        type=int,
        default=768,
        help='Number of units for attention v, k, q, and final linear transforms.'
    )
    parser.add_argument(
        '--word_dff_x',
        type=int,
        default=4,
        help=
        'Number of units for the Point wise feed forward network, normally 4X d_model.'
    )
    parser.add_argument(
        '--word_dropout_rate',
        type=float,
        default=0.1,
        help='Dropout rate for transformer embedding.'
    )
    parser.add_argument(
        '--sent_num_layers',
        type=int,
        default=1,
        help='The number of transformer decoder layers.'
    )
    parser.add_argument(
        '--sent_d_model',
        type=int,
        default=768,
        help='Number of units for attention v, k, q, and final linear transforms.'
    )
    parser.add_argument(
        '--sent_dff_x',
        type=int,
        default=4,
        help=
        'Number of units for the Point wise feed forward network, normally 4X d_model.'
    )
    parser.add_argument(
        '--sent_dropout_rate',
        type=float,
        default=0.1,
        help='Dropout rate for transformer embedding.'
    )
    parser.add_argument(
        '--use_transformer_positional_encoding',
        type='bool',
        default='True',
        help='Add positional encoding to transformer input.'
    )

    parser.add_argument(
        '--sent_pool_abstract_features',
        type='bool',
        default='False',
        help='Pool abstract features with word output.'
    )
    parser.add_argument(
        '--sent_conv_bank_size',
        type=int,
        default=15,
        help='Convolution bank kernal sizes 1 to bank_size.'
    )
    parser.add_argument(
        '--sent_rnn_cell_type',
        type=str,
        choices=['LSTM', 'GRU', 'encoder', 'decoder', 'None'],
        default='LSTM',
        help='RNN Cell Type'
    )
    parser.add_argument(
        '--sent_rnn_num_units',
        type='list',
        default='[128]',
        help='Number of node per recurrent network layer.'
    )
    parser.add_argument(
        '--sent_dense_units',
        type='list',
        # default='[256,128,64]',
        default='[None]',
        help='Number of units for the dense layers.'
    )
    parser.add_argument(
        '--arti_dense_units',
        type='list',
        # default='[256,128,64]',
        default='[None]',
        help='Number of units for the dense layers.'
    )

    parser.add_argument(
        '--use_crf',
        type='bool',
        default='False',
        help='Calculate loss using linear chain CRF instead of Softmax.'
    )
    parser.add_argument(
        '--use_batch_renorm',
        type='bool',
        default='True',
        help='Use Batch Renormalization.'
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        choices=['softmax', 'sigmoid'],
        default='sigmoid',
        help='Loss function'
    )
    parser.add_argument(
        '--scale_label',
        type=float,
        default=0.925,
        help='Scale labels to keep weights from exploding.'
    )
    parser.add_argument(
        '--loss_pos_weight',
        type=float,
        default=1.8,
        help=
        'A value pos_weight > 1 decreases the false negative count, hence increasing the recall. Conversely setting pos_weight < 1 decreases the false positive count and increases the precision.'
    )
    parser.add_argument(
        '--article_loss_pos_weight',
        type=float,
        default=1.8,
        help=
        'A value pos_weight > 1 decreases the false negative count, hence increasing the recall. Conversely setting pos_weight < 1 decreases the false positive count and increases the precision.'
    )

    parser.add_argument(
        '--clip_gradients_std_factor',
        type=float,
        default=2.,  # num_batches_per_epoch * num_epochs_per_decay(8)
        help=
        'If the norm exceeds `exp(mean(log(norm)) + std_factor*std(log(norm)))` then all gradients will be rescaled such that the global norm becomes `exp(mean)`.'
    )
    parser.add_argument(
        '--clip_gradients_decay',
        type=float,
        default=0.95,
        help='The smoothing factor of the moving averages.'
    )
    parser.add_argument(
        '--clip_gradients_static_max_norm',
        type=float,
        default=6.,
        help=
        'If provided, will threshold the norm to this value as an extra safety.'
    )

    parser.add_argument(
        '--no_bert_training_steps',
        type=int,
        default=224,  # num_batches_per_epoch * num_epochs_per_decay(8)
        help='Decay learning_rate by decay_rate every decay_steps.'
    )
    parser.add_argument(
        '--learning_rate_decay_fn',
        type=str,
        default='exponential_decay',
        help=
        'Learning rate decay function. One of "none", "noisy_linear_cosine_decay", "exponential_decay"'
    )
    parser.add_argument(
        '--learning_rate_decay_steps',
        type=int,
        default=2000,  # num_batches_per_epoch * num_epochs_per_decay(8)
        help='Decay learning_rate by decay_rate every decay_steps.'
    )
    parser.add_argument(
        '--learning_rate_decay_rate',
        type=float,
        default=0.7,
        help='Learning rate decay rate.'
    )
    parser.add_argument(
        '--train_seg_embed',
        type='bool',
        default='False',
        help='Whether to train bias and seg_embed layers.'
    )
    parser.add_argument(
        '--freeze_layers',
        type=int,
        default=12,  # 
        help='How many pretrained layers to freeze.'
    )
    parser.add_argument(
        '--layer_decay_rate',
        type=float,
        default=1.0,
        help='Exponetially descrease the learning rate the further up the stack you go for pretrained layers. Top layer: lr[L] = FLAGS.learning_rate. Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.'
    )
    parser.add_argument(
        '--layer_warmup_steps',
        type=int,
        default=2000,  # 200% epoch
        help='Learning rate warmup steps for pretrained layers.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate used for training.'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=250,  # 10% epoch
        help='Learning rate warmup steps needed to reach specified learning_rate.'
    )
    parser.add_argument(
        '--warmup_repeat_steps',
        type=int,
        default=0,  # 0 to disable repeat warmup
        help='Restart warmup every this many steps.'
    )
    parser.add_argument(
        '--warmup_start_lr',
        type=float,
        default=0.001,
        help='Learning rate warmup starting multiplier value.'
    )
    parser.add_argument(
        '--warmup_schedule',
        type=str,
        default='exp',
        help='Learning rate warmup schedule. One of "exp", "linear", "none"'
    )
    # learning rate defaults
    # Adagrad: 0.01
    # Adam: 0.001
    # RMSProp: 0.001
    # :
    # Nadam: 0.002
    # SGD: 0.01
    # Adamax: 0.002
    # Adadelta: 1.0
    parser.add_argument(
        '--optimizer',
        type=str,
        default='bertadam',
        help=
        'Optimizer to use. One of "Adam", "bertadam", "adamw", "Momentum", "Adagrad", "Ftrl", "RMSProp", "SGD"'
    )
    parser.add_argument(
        '--adam_epsilon',
        type=float,
        default=1e-08,
        help=
        'A small constant for numerical stability. This epsilon is "epsilon hat" in the Kingma and Ba paper (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.'
    )

    parser.add_argument(
        '--check_nans',
        type='bool',
        default='False',
        help=
        'Add runtime checks to spot when NaNs or other symptoms of numerical errors start occurring during training.'
    )
    parser.add_argument(
        '--trace',
        type='bool',
        default='False',
        help=
        'Captures CPU/GPU profiling information in "timeline-<step>.json", which are in Chrome Trace format.'
    )
    parser.add_argument(
        '--debug', type='bool', default='False', help='Run debugging ops.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.command = ' '.join(sys.argv)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
