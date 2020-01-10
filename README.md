# Ideas

# Implemented
* lock bert variables until epoch 3
* v3 -> v6: sentence level rnn
* scibert
  * https://github.com/allenai/scibert
* Transformers
  * with rnn

# Research
* https://www.kaggle.com/sergeykalutsky/introducing-bert-with-tensorflow
  * https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb
* https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
  * https://github.com/google-research/bert/blob/cc7051dc592802f501e8a6f71f8fb3cf9de95dc9/run_classifier_with_tfhub.py#L37
* https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=NOO3RfG1DYLo
* https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
* https://github.com/ThilinaRajapakse/simpletransformers
  * Explore pre trained models

# Setup Environment

```ps
py -3.7 -m venv $Env:USERPROFILE\venv\tf15
& "$Env:USERPROFILE\venv\tf15\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools
pip install --upgrade tensorflow-gpu==1.15
pip install --upgrade colorama coloredlogs msgpack tqdm
pip install --upgrade scikit-learn tensorflow_hub bert-tensorflow 
pip install --upgrade sentencepiece xlnet-tensorflow
```

```bash
source /opt/intel/parallel_studio_xe_2019/bin/psxevars.sh
python3.7 -m venv ~/venv/pt13
source ~/venv/pt13/bin/activate
pip install -U pip setuptools colorama coloredlogs msgpack tqdm six wheel mock 'future>=0.17.1'
cd ~/build/np20190921
python setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install
cd ~
pip install torch torchvision
pip install scipy scikit-learn transformers seqeval tensorboardx simpletransformers pandas
import torch
x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()
```

```bash
cd /home/hotdogee/Dropbox/Work/competitions/aicup
source /opt/intel/parallel_studio_xe_2019/bin/psxevars.sh
source ~/venv/pt13/bin/activate
export CUDA_VISIBLE_DEVICES=0; python
export CUDA_VISIBLE_DEVICES=1; python
HOSTSCRIPT=$(basename ${BASH_SOURCE%.*})
DATADIR=/data12/tbrain/aicup1
DATASET=aicup1-v1
TFSCRIPT=aicup1-v1
TOTAL_INSTALLED_GPUS=4
CARDTYPE=2080Ti
CARDID=0
NODE=_2080Ti_W2125-1.4
MODELDIR=/data12/checkpoints/${DATASET}/${TFSCRIPT}
PYTHON=/home/hotdogee/venv/tf15/bin/python
export CUDA_VISIBLE_DEVICES=${CARDID}; ${PYTHON} ${TFSCRIPT}.py --training_data=${DATADIR}/${DATASET}-train.*.tfrecords --eval_data=${DATADIR}/${DATASET}-eval.tfrecords --metadata_path=${DATADIR}/${DATASET}-meta.json --save_summary_steps=100 --log_step_count_steps=10 --learning_rate_decay_steps=800 --learning_rate_decay_rate=0.7 --adam_epsilon=0.05 --save_checkpoints_secs=900 --keep_checkpoint_max=24 --optimizer=Momentum --learning_rate_decay_fn=exponential_decay --warmup_schedule=exp --warmup_steps=116 --warmup_repeat_steps=0 --warmup_start_lr=0.001 --batch_size=1 --rnn_num_units=[512,512,512,512] --learning_rate=0.0060 --random_seed=-1 --model_dir=${MODELDIR}/gru512x4-lr60_aicup1_${NODE}
# 6000 articles, 58 steps per epoch
# set warmup_steps = 116, decay_steps = 800
```

# Quick Explorations
```python
# checkpoint averaging
import re
from collections import OrderedDict
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
# import xlnet
from xlnet import xlnet
checkpoint_root = Path(r'E:\tbrain\aicup1\checkpoints\large-fp16-v16-xlnet-b8-24g_test\test58-lr160-f6-en-pw136-ds377-lws40-ws40-sd1f4-authors-d16384_aicup1-v15_TITANRTX_8086K1-2-00001')
checkpoint = str(checkpoint_root / 'step-1697')
vars = tf.train.list_variables(checkpoint)
reader = tf.train.load_checkpoint(checkpoint)
tensor = reader.get_tensor('global_step')
tensor = reader.get_tensor('output/dense/bias')
np.zeros([6]) + reader.get_tensor('output/dense/bias')
np.max([np.zeros([6]), reader.get_tensor('output/dense/bias')],0)
np.min([np.zeros([6]), reader.get_tensor('output/dense/bias')],0)
np.average([np.zeros([6]), reader.get_tensor('output/dense/bias')], axis=0, weights=[50,50])
[('abstract/model/transformer/layer_0/ff/LayerNorm/beta', [1024]), ('optimizer/sentence_dense/dense_0_16384/dense_10/kernel/adam_v', [1024, 16384]), ('output/dense/bias', [6]), ('output/dense/kernel', [16384, 6]), ('sent_decoder/decoder_layer/layer_normalization/beta', [1024]), ('sent_decoder/decoder_layer/layer_normalization/gamma', [1024]), ('sent_decoder/decoder_layer/layer_normalization_1/beta', [1024]), ('sent_decoder/decoder_layer/layer_normalization_1/gamma', [1024]), ('sent_decoder/decoder_layer/layer_normalization_2/beta', [1024]), ('sent_decoder/decoder_layer/layer_normalization_2/gamma', [1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense_1/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense_1/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense_2/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense_2/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense_3/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention/dense_3/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_4/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_4/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_5/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_5/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_6/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_6/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_7/bias', [1024]), ('sent_decoder/decoder_layer/multi_head_attention_1/dense_7/kernel', [1024, 1024]), ('sent_decoder/decoder_layer/sequential/dense_8/bias', [4096]), ('sent_decoder/decoder_layer/sequential/dense_8/kernel', [1024, 4096]), ('sent_decoder/decoder_layer/sequential/dense_9/bias', [1024]), ('sent_decoder/decoder_layer/sequential/dense_9/kernel', [4096, 1024]), ('sentence_dense/dense_0_16384/dense_10/bias', [16384]), ('sentence_dense/dense_0_16384/dense_10/kernel', [1024, 16384])]
```
```python
# xlnet
import re
from collections import OrderedDict
import tensorflow as tf
# tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
# import xlnet
from xlnet import xlnet

class RunConfig(object):
  def __init__(self, is_training, use_tpu, use_bfloat16, dropout, dropatt,
               init="normal", init_range=0.1, init_std=0.02, mem_len=None,
               reuse_len=None, bi_data=False, clamp_len=-1, same_length=False):
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

xlnet_config = xlnet.XLNetConfig(json_path=r'E:\tbrain\aicup1\scibert\xlnet_cased_L-12_H-768_A-12\xlnet_config.json')
xlnet_config = xlnet.XLNetConfig(json_path=r'E:\tbrain\aicup1\scibert\xlnet_cased_L-24_H-1024_A-16\xlnet_config.json')
run_config = RunConfig(
    is_training=True,
    use_tpu=False,
    use_bfloat16=False,
    dropout=0.1,
    dropatt=0.1
)
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]], tf.float32)
input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

xlnet_model = xlnet.XLNetModel(
    xlnet_config=xlnet_config,
    run_config=run_config,
    input_ids=input_ids,
    seg_ids=input_type_ids,
    input_mask=input_mask
)
output = xlnet_model.get_sequence_output()
abstract_pooled = xlnet_model.get_pooled_out('first', False)
tvars = tf.trainable_variables()
init_checkpoint_root = Path(r'E:\tbrain\aicup1\scibert\xlnet_cased_L-12_H-768_A-12')
init_checkpoint_root = Path(r'E:\tbrain\aicup1\scibert\xlnet_cased_L-24_H-1024_A-16')
init_checkpoint = str(init_checkpoint_root / 'xlnet_model.ckpt')
init_vars = tf.train.list_variables(init_checkpoint)
name_to_variable = OrderedDict()
for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
        name = m.group(1)
    name_to_variable[name] = var

initialized_variable_names = {}
assignment_map = OrderedDict()
for x in init_vars:
    (name, var) = (x[0], x[1])
    # tf.logging.info('original name: %s', name)
    if name not in name_to_variable:
        continue
    # assignment_map[name] = name
    assignment_map[name] = name_to_variable[name]
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

tf.logging.info('**** Trainable Variables ****')
for var in tvars:
    init_string = ''
    if var.name in initialized_variable_names:
        init_string = ', *INIT_FROM_CKPT*'
    tf.logging.info(
        '  name = %s, shape = %s%s', var.name, var.shape, init_string
    )

init_checkpoint_root2 = Path(r'E:\tbrain\aicup1\scibert\xlnet_cased_L-24_H-1024_A-16')
init_checkpoint2 = str(init_checkpoint_root2 / 'xlnet_model.ckpt')
init_vars2 = tf.train.list_variables(init_checkpoint2)

[<tf.Variable 'model/transformer/r_w_bias:0' shape=(12, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/r_r_bias:0' shape=(12, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/word_embedding/lookup_table:0' shape=(32000, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/r_s_bias:0' shape=(12, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/seg_embed:0' shape=(12, 2, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_0/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_1/rel_attn/q/kernel:0' shape=(768, 
rmer/layer_1/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_1/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_1/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_1/ff/layer_1/kernel:0' shape=(768, 3072) dtyp0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_1/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/rel_attn/LayerNorm/beta:0' shape=(768,) 
dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_2/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_3/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_4/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_5/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/rel_attn/o/kernel:0' shape=(768, 12, 64) 
dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_6/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, 
<tf.Variable 'model/transformer/layer_7/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_7/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/rel_attn/q/kernel:0' shape=(768, 12, 
64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_8/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_9/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_10/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/rel_attn/q/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/rel_attn/k/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/rel_attn/v/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/rel_attn/r/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/rel_attn/o/kernel:0' shape=(768, 12, 64) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/rel_attn/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/rel_attn/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/ff/layer_1/kernel:0' shape=(768, 3072) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/ff/layer_1/bias:0' shape=(3072,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/ff/layer_2/kernel:0' shape=(3072, 768) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/ff/layer_2/bias:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/ff/LayerNorm/beta:0' shape=(768,) dtype=float32_ref>, <tf.Variable 'model/transformer/layer_11/ff/LayerNorm/gamma:0' shape=(768,) dtype=float32_ref>]

[('global_step', []), ('model/lm_loss/bias', [32000]), ('model/transformer/layer_0/ff/LayerNorm/beta', [768]), ('model/transformer/layer_0/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_0/ff/layer_1/bias', [3072]), ('model/transformer/layer_0/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_0/ff/layer_2/bias', [768]), ('model/transformer/layer_0/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_0/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_0/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_0/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_0/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_0/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_0/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_0/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_1/ff/LayerNorm/beta', [768]), ('model/transformer/layer_1/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_1/ff/layer_1/bias', [3072]), ('model/transformer/layer_1/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_1/ff/layer_2/bias', [768]), ('model/transformer/layer_1/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_1/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_1/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_1/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_1/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_1/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_1/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_1/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_10/ff/LayerNorm/beta', [768]), ('model/transformer/layer_10/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_10/ff/layer_1/bias', [3072]), ('model/transformer/layer_10/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_10/ff/layer_2/bias', [768]), ('model/transformer/layer_10/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_10/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_10/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_10/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_10/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_10/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_10/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_10/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_11/ff/LayerNorm/beta', [768]), ('model/transformer/layer_11/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_11/ff/layer_1/bias', [3072]), ('model/transformer/layer_11/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_11/ff/layer_2/bias', [768]), ('model/transformer/layer_11/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_11/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_11/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_11/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_11/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_11/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_11/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_11/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_2/ff/LayerNorm/beta', [768]), ('model/transformer/layer_2/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_2/ff/layer_1/bias', [3072]), ('model/transformer/layer_2/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_2/ff/layer_2/bias', [768]), ('model/transformer/layer_2/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_2/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_2/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_2/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_2/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_2/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_2/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_2/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_3/ff/LayerNorm/beta', [768]), ('model/transformer/layer_3/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_3/ff/layer_1/bias', [3072]), ('model/transformer/layer_3/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_3/ff/layer_2/bias', [768]), ('model/transformer/layer_3/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_3/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_3/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_3/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_3/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_3/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_3/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_3/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_4/ff/LayerNorm/beta', [768]), ('model/transformer/layer_4/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_4/ff/layer_1/bias', [3072]), ('model/transformer/layer_4/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_4/ff/layer_2/bias', [768]), ('model/transformer/layer_4/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_4/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_4/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_4/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_4/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_4/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_4/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_4/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_5/ff/LayerNorm/beta', [768]), ('model/transformer/layer_5/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_5/ff/layer_1/bias', [3072]), ('model/transformer/layer_5/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_5/ff/layer_2/bias', [768]), ('model/transformer/layer_5/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_5/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_5/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_5/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_5/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_5/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_5/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_5/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_6/ff/LayerNorm/beta', [768]), ('model/transformer/layer_6/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_6/ff/layer_1/bias', [3072]), ('model/transformer/layer_6/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_6/ff/layer_2/bias', [768]), ('model/transformer/layer_6/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_6/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_6/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_6/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_6/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_6/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_6/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_6/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_7/ff/LayerNorm/beta', [768]), ('model/transformer/layer_7/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_7/ff/layer_1/bias', [3072]), ('model/transformer/layer_7/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_7/ff/layer_2/bias', [768]), ('model/transformer/layer_7/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_7/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_7/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_7/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_7/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_7/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_7/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_7/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_8/ff/LayerNorm/beta', [768]), ('model/transformer/layer_8/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_8/ff/layer_1/bias', [3072]), ('model/transformer/layer_8/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_8/ff/layer_2/bias', [768]), ('model/transformer/layer_8/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_8/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_8/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_8/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_8/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_8/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_8/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_8/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/layer_9/ff/LayerNorm/beta', [768]), ('model/transformer/layer_9/ff/LayerNorm/gamma', [768]), ('model/transformer/layer_9/ff/layer_1/bias', [3072]), 
('model/transformer/layer_9/ff/layer_1/kernel', [768, 3072]), ('model/transformer/layer_9/ff/layer_2/bias', [768]), ('model/transformer/layer_9/ff/layer_2/kernel', [3072, 768]), ('model/transformer/layer_9/rel_attn/LayerNorm/beta', [768]), ('model/transformer/layer_9/rel_attn/LayerNorm/gamma', [768]), ('model/transformer/layer_9/rel_attn/k/kernel', [768, 12, 64]), ('model/transformer/layer_9/rel_attn/o/kernel', [768, 12, 64]), ('model/transformer/layer_9/rel_attn/q/kernel', [768, 12, 64]), ('model/transformer/layer_9/rel_attn/r/kernel', [768, 12, 64]), ('model/transformer/layer_9/rel_attn/v/kernel', [768, 12, 64]), ('model/transformer/mask_emb/mask_emb', [1, 1, 768]), ('model/transformer/r_r_bias', [12, 12, 64]), ('model/transformer/r_s_bias', [12, 12, 64]), ('model/transformer/r_w_bias', [12, 12, 64]), ('model/transformer/seg_embed', [12, 2, 12, 64]), ('model/transformer/word_embedding/lookup_table', [32000, 768])]

do_lower_case=True
init_checkpoint_root = Path(r'E:\tbrain\aicup1\scibert\xlnet_cased_L-12_H-768_A-12')
vocab_file = str(init_checkpoint_root / 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
max_seq_length = 512
bert_config_file = str(init_checkpoint_root / 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_vars = tf.train.list_variables(init_checkpoint)
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=input_type_ids,
    use_one_hot_embeddings=False
)
tvars = tf.trainable_variables()
```
```python
# xlnet
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load(r'E:\tbrain\aicup1\scibert\xlnet_cased_L-12_H-768_A-12\spiece.model')
vocabs = [[sp.id_to_piece(id), id] for id in range(sp.get_piece_size())]
>>> vocabs[:100]
[['<unk>', 0], ['<s>', 1], ['</s>', 2], ['<cls>', 3], ['<sep>', 4], ['<pad>', 5], ['<mask>', 6], ['<eod>', 7], ['<eop>', 8], ['.', 9], ['(', 10], [')', 11], ['"', 12], ['-', 13], ['–', 14], ['£', 15], ['€', 16], ['▁', 17], ['▁the', 18], [',', 19], ['▁of', 20], ['▁and', 21], ['▁to', 22], ['s', 23], ['▁a', 24], ['▁in', 25], ["'", 26], ['▁is', 27], ['▁for', 28], ['▁that', 29], ['▁was', 30], ['▁on', 31], ['▁The', 32], ['▁with', 33], ['▁as', 34], ['▁I', 35], ['▁it', 36], ['▁by', 37], ['▁at', 38], ['▁be', 39], ['▁from', 40], ['▁are', 41], ['▁said', 42], ['▁he', 43], ['▁you', 44], ['▁his', 45], ['t', 46], ['▁have', 47], ['▁an', 48], ['▁or', 49], ['▁not', 50], ['▁has', 51], ['▁this', 52], ['▁will', 53], ['▁had', 54], ['▁were', 55], ['ing', 56], ['▁but', 57], ['▁their', 58], ['▁which', 59], [':', 60], ['▁who', 61], ['▁her', 62], ['▁they', 63], ['▁can', 64], ['▁one', 65], ['d', 66], ['▁In', 67], ['ed', 68], ['▁He', 69], ['▁more', 70], ['▁all', 71], ['▁been', 72], ['▁your', 73], ['▁would', 74], ['▁about', 75], ['▁up', 76], ['▁also', 77], ['▁out', 78], ['▁A', 79], ['▁we', 80], ['▁its', 81], ['?', 82], ['S', 83], ['▁It', 84], ['▁she', 85], ['▁other', 86], ['▁two', 87], ['re', 88], ['▁first', 89], ['▁when', 90], ['▁into', 91], ['▁time', 92], ['e', 93], ['▁my', 94], ['▁over', 95], ['I', 96], [';', 97], ['m', 98], ['▁after', 99]]
print_(u'I was born in 2000, and this is falsé.')
print_(u'ORIGINAL', sp.EncodeAsPieces(u'I was born in 2000, and this is falsé.'))
print_(u'OURS', encode_pieces(sp, u'I was born in 2000, and this is falsé.'))
print(encode_ids(sp, u'I was born in 2000, and this is falsé.'))
print_('')
prepro_func = partial(preprocess_text, lower=True)
print_(prepro_func('I was born in 2000, and this is falsé.'))
print_('ORIGINAL', sp.EncodeAsPieces(prepro_func('I was born in 2000, and this is falsé.')))
print_('OURS', encode_pieces(sp, prepro_func('I was born in 2000, and this is falsé.')))
print(encode_ids(sp, prepro_func('I was born in 2000, and this is falsé.')))
print_('')
print_('I was born in 2000, and this is falsé.')
print_('ORIGINAL', sp.EncodeAsPieces('I was born in 2000, and this is falsé.'))
print_('OURS', encode_pieces(sp, 'I was born in 2000, and this is falsé.'))
print(encode_ids(sp, 'I was born in 2000, and this is falsé.'))
print_('')
print_('I was born in 92000, and this is falsé.')
print_('ORIGINAL', sp.EncodeAsPieces('I was born in 92000, and this is falsé.'))
print_('OURS', encode_pieces(sp, 'I was born in 92000, and this is falsé.'))
print(encode_ids(sp, 'I was born in 92000, and this is falsé.'))
```
```python
# label split
import tensorflow as tf
import os
import re
import csv
import json
import math
import errno
import random
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
train1_path = verify_input_path(r'E:\tbrain\aicup1\task1_trainset.csv')
train2_path = verify_input_path(r'E:\tbrain\aicup2\task2_trainset.csv')
test_path = verify_input_path(r'E:\tbrain\aicup1\task1_public_testset.csv')
train_articles = parse_train_articles(train1_path, train2_path)
test_articles = parse_test_articles(test_path)
print(f'generate categories')
label1_count = defaultdict(int)
label2_count = defaultdict(int)
label1_idx = defaultdict(set)
label2_idx = defaultdict(set)
for i, article in enumerate(train_articles):
    for label in article['Task 1']:
        label1_count[label] += 1
        label1_idx[label].add(i)
    label2_count[article['Task 2']] += 1
    label2_idx[article['Task 2']].add(i)

task1_categories = ['PAD', 'START', 'END'] + sorted(label1_count, key=lambda k: (label1_count[k], k), reverse=True)
task1_mappings = {c: i for i, c in enumerate(task1_categories)}
print(f'task1_categories: {len(task1_categories)}')

task2_categories = ['PAD'] + sorted(label2_count, key=lambda k: (label2_count[k], k), reverse=True)
task2_mappings = {c: i for i, c in enumerate(task2_categories)}
print(f'task2_categories: {len(task2_categories)}')
print(f'label1_count: {label1_count}')
print(f'label2_count: {label2_count}')

# split training
print(f'split training')
random.seed(113)
np.random.seed(113)

label1_count = defaultdict(int)
label2_count = defaultdict(int)
label1_idx = defaultdict(set)
label2_idx = defaultdict(set)
for i, article in enumerate(train_articles):
    for label in article['Task 1']:
        label1_count[label] += 1
        label1_idx[label].add(i)
    label2_count[article['Task 2']] += 1
    label2_idx[article['Task 2']].add(i)

eval_goal_count = {}
label1_goal_count = {}
label2_goal_count = {}
for k, v in label1_count.items():
    label1_goal_count[k] = int((v+0.5)/train_split+0.5)
    eval_goal_count[k] = int((v+0.5)/train_split+0.5)

for k, v in label2_count.items():
    label2_goal_count[k] = int((v)/train_split+0.5)

eval_idx = set()
eval_count = defaultdict(int)
for label, goal in sorted(label1_goal_count.items(), key=lambda x: x[1]):
    while eval_count[label] < goal:
        for idx in random.sample(label1_idx[label] - eval_idx, goal - eval_count[label]):
            eval_idx.add(idx)
            for l in train_articles[idx]['Task 1']:
                eval_count[l] += 1

error = 0
for k, v in label1_goal_count.items():
    e = (v - eval_count[k])/(v) *100
    error += e * e

math.pow(error, 0.5)
# error = 589.0845684349014

label1_count = defaultdict(int)
label2_count = defaultdict(int)
label1_idx = defaultdict(set)
label2_idx = defaultdict(set)
for i, article in enumerate(train_articles):
    for label in article['Task 1']:
        label1_count[label] += 1
        label1_idx[label].add(i)
    label2_count[article['Task 2']] += 1
    label2_idx[article['Task 2']].add(i)

eval_goal_count = {}
label1_goal_count = {}
label2_goal_count = {}
for k, v in label1_count.items():
    label1_goal_count[k] = int((v+0.5)/train_split+0.5)
    eval_goal_count[k] = int((v+0.5)/train_split+0.5)

for k, v in label2_count.items():
    label2_goal_count[k] = int((v)/train_split+0.5)

eval_idx = set()
eval_count = defaultdict(int)
while sum([v for v in eval_goal_count.values() if v > 0]) > 0:
    label, goal = sorted([(l, g) for l, g in eval_goal_count.items() if g > 0], key=lambda x: x[1])[0]
    idx = random.choice(list(label1_idx[label] - eval_idx))
    eval_idx.add(idx)
    for l in train_articles[idx]['Task 1']:
        eval_count[l] += 1
        eval_goal_count[l] -= 1

error = 0
for k, v in label1_goal_count.items():
    e = (v - eval_count[k])/(v) *100
    error += e * e

math.pow(error, 0.5)
sorted(eval_count.items(), key=lambda x: x[1])
sorted(label1_goal_count.items(), key=lambda x: x[1])
# error = 677.3380480873988

label1_count = defaultdict(int)
label2_count = defaultdict(int)
label1_idx = defaultdict(set)
label2_idx = defaultdict(set)
for i, article in enumerate(train_articles):
    for label in article['Task 1']:
        label1_count[label] += 1
        label1_idx[label].add(i)
    label2_count[article['Task 2']] += 1
    label2_idx[article['Task 2']].add(i)

eval_goal_count = {}
label1_goal_count = {}
label2_goal_count = {}
for k, v in label1_count.items():
    label1_goal_count[k] = int((v+0.5)/train_split+0.5)
    eval_goal_count[k] = int((v+0.5)/train_split+0.5)

for k, v in label2_count.items():
    label2_goal_count[k] = int((v)/train_split+0.5)

eval_idx = set()
eval_count = defaultdict(int)
while sum([v for v in eval_goal_count.values() if v > 0]) > 0:
    label, goal = sorted([(l, g) for l, g in eval_goal_count.items() if g > 0], key=lambda x: x[1])[0]
    candidates = []
    for idx in label1_idx[label] - eval_idx:
        below_zero = 0
        above_zero = 0
        for l in train_articles[idx]['Task 1']:
            if eval_goal_count[l] <= 0:
                below_zero += 1
            else:
                above_zero -= eval_goal_count[l]
        candidates.append((idx, (below_zero, above_zero)))
    random.shuffle(candidates)
    idx, _ = sorted(candidates, key=lambda x: x[1])[0]
    eval_idx.add(idx)
    for l in train_articles[idx]['Task 1']:
        eval_count[l] += 1
        eval_goal_count[l] -= 1

error = 0
for k, v in label1_goal_count.items():
    e = (v - eval_count[k])/(v) *100
    error += e * e

math.pow(error, 0.5)
error = 0
for k, v in label2_goal_count.items():
    e = (v - eval_count[k])/(v) *100
    error += e * e

math.pow(error, 0.5)
sorted(eval_count.items(), key=lambda x: x[1])
sorted(label1_goal_count.items(), key=lambda x: x[1])
# error = 120.18304506756509

random.seed(113)
train_split=7

label1_count = defaultdict(int)
label2_count = defaultdict(int)
label1_idx = defaultdict(set)
label2_idx = defaultdict(set)
for i, article in enumerate(train_articles):
    for label in article['Task 1']:
        label1_count[label] += 1
        label1_idx[label].add(i)
    label2_count[article['Task 2']] += 1
    label2_idx[article['Task 2']].add(i)

eval_goal_count = {}
label1_goal_count = {}
label2_goal_count = {}
for k, v in label1_count.items():
    label1_goal_count[k] = int((v+0.5)/train_split+0.5)
    eval_goal_count[k] = int((v+0.5)/train_split+0.5)

for k, v in label2_count.items():
    label2_goal_count[k] = int((v)/train_split+0.5)

eval_idx = set()
eval_label1_count = defaultdict(int)
eval_label2_count = defaultdict(int)
while sum([v for v in eval_goal_count.values() if v > 0]) > 0:
    label, goal = sorted([(l, g) for l, g in eval_goal_count.items() if g > 0], key=lambda x: x[1])[0]
    candidates = []
    for idx in label1_idx[label] - eval_idx:
        below_zero = 0
        for l in train_articles[idx]['Task 1']:
            if eval_goal_count[l] <= 0:
                below_zero += 1
        candidates.append((idx, below_zero))
    random.shuffle(candidates)
    idx, _ = sorted(candidates, key=lambda x: x[1])[0]
    eval_idx.add(idx)
    for l in train_articles[idx]['Task 1']:
        eval_label1_count[l] += 1
        eval_goal_count[l] -= 1
    eval_label2_count[train_articles[idx]['Task 2']] += 1

error = 0
for k, v in label1_goal_count.items():
    e = (v - eval_label1_count[k])/(v) *100
    error += e * e

math.pow(error, 0.5)
error = 0
for k, v in label2_goal_count.items():
    e = (v - eval_label2_count[k])/(v) *100
    error += e * e

math.pow(error, 0.5)
sorted(eval_label1_count.items(), key=lambda x: x[1])
sorted(label1_goal_count.items(), key=lambda x: x[1])
len(eval_idx)

# error = 2.6364957646733784
# save ids
train_id_list = []
eval_id_list = []
for i, article in enumerate(train_articles):
    id = int(article['Id'][1:])
    if i in eval_idx:
        eval_id_list.append(id)
    else:
        train_id_list.append(id)

train_id_lists = [[] for i in range(6)]
random.seed(113)
random.shuffle(train_id_list)
for i, a in enumerate(train_id_list):
    train_id_lists[i % 6].append(a)


train_eval_id_lists = {
    'train_id_lists': train_id_lists,
    'eval_id_list': eval_id_list
}

len(train_eval_id_lists['train_id_lists'])
len(train_eval_id_lists['eval_id_list'])
p = Path(r'E:\tbrain\aicup1\train_eval_id_lists_v2.json')
with p.open('w') as f:
    json.dump(train_eval_id_lists, f, indent=2, sort_keys=False)

id_article = {}
for a in train_articles:
    id_article[int(a['Id'][1:])] = a

train_lists = [[id_article[i] for i in random.sample(l, k=len(l))] for l in train_id_lists]
eval_list = [id_article[i] for i in random.sample(eval_id_list, k=len(eval_id_list))]

train_lists.append(eval_list)

split_label1_count = [defaultdict(int) for i in range(train_split)]
split_label2_count = [defaultdict(int) for i in range(train_split)]
for i, articles in enumerate(train_lists):
    for article in articles:
        for label in article['Task 1']:
            split_label1_count[i][label] += 1
        split_label2_count[i][article['Task 2']] += 1

task1_errors = []
task2_errors = []
for i, articles in enumerate(train_lists):
    error = 0
    for k, v in label1_goal_count.items():
        e = (v - split_label1_count[i][k])/(v) *100
        error += e * e
    task1_errors.append(math.pow(error, 0.5))
    error = 0
    for k, v in label2_goal_count.items():
        e = (v - split_label2_count[i][k])/(v) *100
        error += e * e
    task2_errors.append(math.pow(error, 0.5))

best_train_lists = []
best_error = {
    'task1_min': 999,
    'task1_total': 999,
    'task2_min': 999,
    'task2_total': 999
}
for j in range(10000):
    train_lists = [[] for i in range(train_split)]
    np.random.shuffle(train_articles)
    for i, a in enumerate(train_articles):
        train_lists[i % train_split].append(a)
    train_list_ids = [sorted([int(a['Id'][1:]) for a in l]) for l in train_lists]
    split_label1_count = [defaultdict(int) for i in range(train_split)]
    split_label2_count = [defaultdict(int) for i in range(train_split)]
    for i, articles in enumerate(train_lists):
        for article in articles:
            for label in article['Task 1']:
                split_label1_count[i][label] += 1
            split_label2_count[i][article['Task 2']] += 1
    task1_errors = []
    task2_errors = []
    for i, articles in enumerate(train_lists):
        error = 0
        for k, v in label1_goal_count.items():
            e = (v - split_label1_count[i][k])/(v) *100
            error += e * e
        task1_errors.append(math.pow(error, 0.5))
        error = 0
        for k, v in label2_goal_count.items():
            e = (v - split_label2_count[i][k])/(v) *100
            error += e * e
        task2_errors.append(math.pow(error, 0.5))
    task1_min = min(task1_errors)
    task1_total = sum(task1_errors)
    task2_min = min(task2_errors)
    task2_total = sum(task2_errors)
    if (task1_min, task2_min, task1_total, task2_total) < (best_error['task1_min'], best_error['task2_min'], best_error['task1_total'], best_error['task2_total']):
        best_error['task1_min'] = task1_min
        best_error['task1_total'] = task1_total
        best_error['task2_min'] = task2_min
        best_error['task2_total'] = task2_total
        best_error['split_label1_count'] = split_label1_count
        best_error['task1_errors'] = task1_errors
        print(j, best_error)
        best_train_lists = train_lists
# best error = 130
train_list_ids = [sorted([int(a['Id'][1:]) for a in l]) for l in best_train_lists]
split_label1_count = [defaultdict(int) for i in range(train_split)]
split_label2_count = [defaultdict(int) for i in range(train_split)]
for i, articles in enumerate(best_train_lists):
    for article in articles:
        for label in article['Task 1']:
            split_label1_count[i][label] += 1
        split_label2_count[i][article['Task 2']] += 1

task1_errors = []
task2_errors = []
for i, articles in enumerate(best_train_lists):
    error = 0
    for k, v in label1_goal_count.items():
        e = (v - split_label1_count[i][k])/(v) *100
        error += e * e
    task1_errors.append(math.pow(error, 0.5))
    error = 0
    for k, v in label2_goal_count.items():
        e = (v - split_label2_count[i][k])/(v) *100
        error += e * e
    task2_errors.append(math.pow(error, 0.5))

task1_min = min(task1_errors)
task1_total = sum(task1_errors)
task2_min = min(task2_errors)
task2_total = sum(task2_errors)
(task1_min, task2_min, task1_total, task2_total)
train_eval_id_lists = {
    'train_id_lists': train_list_ids[:3] + train_list_ids[4:],
    'eval_id_list': train_list_ids[3]
}
len(best_train_lists[:3])
len(best_train_lists[4:])
len(best_train_lists)
len(train_eval_list['train_list'])
len(train_eval_list['eval_list'])
p = Path(r'E:\tbrain\aicup1\train_eval_id_lists_v1.json')
with p.open('w') as f:
    json.dump(train_eval_id_lists, f, indent=2, sort_keys=False)
```
```python
# model_dir and checkpoint
import tensorflow as tf
import re
checkpoint_dir = r'/data12/checkpoints/aicup1-v3/aicup1-v3/bertadam-b1024-v10-aicupbert4_test/test29-sl80-pw1420-lr80-b8-wd1f8-sd1f8-d32768_aicup1-v8_TITANRTX_8086K1-2.2'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir, latest_filename='epoch.latest')
epoch_checkpoint_re = re.compile(r'.*epoch-(\d+)-\d+$')
int(epoch_checkpoint_re.match(latest_checkpoint).group(1))
```
```python
# tokenizer exploration
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
import bert
from bert import tokenization
from bert import modeling
init_checkpoint_root = Path(r'E:\tbrain\aicup1\scibert\biobert_v1.1_pubmed')
do_lower_case = False
vocab_file = init_checkpoint_root / 'vocab.txt'
tokenizer = tokenization.FullTokenizer(str(vocab_file), do_lower_case)
sentences = "High quality upsampling of sparse 3D point clouds is critically useful for a wide range of geometric operations such as reconstruction, rendering, meshing, and analysis.$$$In this paper, we propose a data-driven algorithm that enables an upsampling of 3D point clouds without the need for hard-coded rules.$$$Our approach uses a deep network with Chamfer distance as the loss function, capable of learning the latent features in point clouds belonging to different object categories.$$$We evaluate our algorithm across different amplification factors, with upsampling learned and performed on objects belonging to the same category as well as different categories.$$$We also explore the desirable characteristics of input point clouds as a function of the distribution of the point samples.$$$Finally, we demonstrate the performance of our algorithm in single-category training versus multi-category training scenarios.$$$The final proposed model is compared against a baseline, optimization-based upsampling method.$$$Results indicate that our algorithm is capable of generating more uniform and accurate upsamplings.".split('$$$')
tokens_list = [tokenizer.tokenize(sent) for sent in sentences]
total_length = sum([len(tokens) for tokens in tokens_list])
```
```python
# bert from ckpt inside variable_scope or name_scope
import tensorflow as tf
# tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
import bert
from bert import tokenization
from bert import modeling

init_checkpoint_root = Path(r'E:\tbrain\aicup1\scibert\scibert_scivocab_uncased')
bert_config_file = str(init_checkpoint_root / 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
with tf.name_scope("abstract1"):
    model1 = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False
    )
tvars = tf.trainable_variables()
# <tf.Variable 'bert/embeddings/word_embeddings:0' shape=(31090, 768) dtype=float32_ref>
with tf.variable_scope("abstract2"):
    model2 = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False
    )
tvars2 = tf.trainable_variables()
# <tf.Variable 'abstract2/bert/embeddings/word_embeddings:0' shape=(31090, 768) dtype=float32_ref>
with tf.name_scope("abstract3"):
    model1 = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=False
    )
tvars3 = tf.trainable_variables()
# <tf.Variable 'bert_1/embeddings/word_embeddings:0' shape=(31090, 768) dtype=float32_ref>
```
```python
# bert from ckpt without using hub
import tensorflow as tf
# tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
import bert
from bert import tokenization
from bert import modeling
do_lower_case=True
init_checkpoint_root = Path(r'E:\tbrain\aicup1\scibert\scibert_scivocab_uncased')
vocab_file = str(init_checkpoint_root / 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
max_seq_length = 512
bert_config_file = str(init_checkpoint_root / 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = str(init_checkpoint_root / 'bert_model.ckpt')
init_vars = tf.train.list_variables(init_checkpoint)
input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
input_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=input_type_ids,
    use_one_hot_embeddings=False
)
tvars = tf.trainable_variables()
init_vars = tf.train.list_variables(init_checkpoint)
# paste in model_fn_builder, input_fn_builder, BertFeatures
model_fn = model_fn_builder(
    bert_config=bert_config,
    init_checkpoint=init_checkpoint
)
estimator = tf.estimator.Estimator(model_fn=model_fn)
title_features=[]
title_features.append(BertFeatures(
    input_ids=[31, 51, 99],
    input_mask=[1, 1, 1],
    input_type_ids=[0, 0, 1])
)
title_features.append(BertFeatures(
    input_ids=[15, 5, 0],
    input_mask=[1, 1, 0],
    input_type_ids=[0, 2, 0])
)
batch_size = 4
input_fn = input_fn_builder(features=title_features, seq_length=max_seq_length, batch_size=batch_size)
predictions = estimator.predict(
    input_fn=input_fn,
    yield_single_examples=True
)
result=[]
for i, p in enumerate(predictions):
    # print(i, p)
    result.append(p)
bert_module = hub.Module('https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1', trainable=True)
trainable_vars = bert_module.variables
trainable_vars2 = [var for var in trainable_vars if not "/cls/" in var.name]
trainable_vars3 = trainable_vars2[-10 :]
```
```python
# making target_input and target_output for seq2seq
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path

max_sentences = 5
batch_size, sequence_length, num_classes = (2, 10, 5)
lengths = tf.constant([9,6])
sentence_count = tf.constant([3,4])
max_sentence_count = tf.reduce_max(sentence_count)
sentence_lengths = tf.constant([[3,4,2,0],[2,2,1,1]])
segment_ids = tf.constant([[0,0,0,1,1,1,1,2,2,0],[0,0,1,1,2,3,0,0,0,0]])
mask = tf.sequence_mask(lengths=lengths, maxlen=sequence_length, dtype=tf.float32)
outputs = tf.random.uniform([batch_size, sequence_length, num_classes],minval=0,maxval=1,dtype=tf.float32)

new_outputs = tf.map_fn(lambda i: tf.concat([tf.slice([tf.nn.relu(tf.reduce_max(s, axis=0)) for s in tf.dynamic_partition(data=tf.slice(outputs[i], [0,0], [lengths[i],tf.shape(outputs)[-1]]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=max_sentences)], [0,0], [max_sentence_count,tf.shape(outputs)[-1]])],axis=1), tf.range(batch_size), tf.float32)

with tf.Session() as sess:
    print(sess.run([new_outputs]))
```
```python
# make batch of sentences from batch of words
# outputs shape=(batch_size, max_tokens, dim), dtype=float32
# new_outputs shape=(batch_size, max_sentences, max_tokens_per_sentence, dim), dtype=float32
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
metadata_path = Path(r'E:\tbrain\aicup1\aicup1-v9-meta.json')
with metadata_path.open('r') as f:
    metadata = json.load(f)

task1_embeddings = tf.constant(metadata['task1_embeddings'])
max_sentences = 5
batch_size, sequence_length, num_classes = (2, 10, 5)
lengths = tf.constant([9,6])
sentence_count = tf.constant([3,4])
max_sentence_count = tf.reduce_max(sentence_count)
sentence_lengths = tf.constant([[3,4,2,0],[2,2,1,1]])
segment_ids = tf.constant([[0,0,0,1,1,1,1,2,2,0],[0,0,1,1,2,3,0,0,0,0]])
sentence_labels = tf.constant([[3,4,2,0],[3,1,1,1]])
sentence_labels_flat = tf.RaggedTensor.from_tensor(sentence_labels, padding=0).flat_values
sentence_classes = tf.nn.embedding_lookup(task1_embeddings, sentence_labels_flat)
mask = tf.sequence_mask(lengths=lengths, maxlen=sequence_length, dtype=tf.float32)
predicted_labels = tf.constant([[3,3,2,4,4,4,3,2,2,1],[3,3,2,1,4,1,0,0,0,0]])
outputs = tf.random.uniform([batch_size, sequence_length, num_classes],minval=0,maxval=1,dtype=tf.float32)

new_outputs = tf.map_fn(lambda i: tf.concat([tf.slice([tf.nn.relu(tf.reduce_max(s, axis=0)) for s in tf.dynamic_partition(data=tf.slice(outputs[i], [0,0], [lengths[i],tf.shape(outputs)[-1]]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=max_sentences)], [0,0], [max_sentence_count,tf.shape(outputs)[-1]])],axis=1), tf.range(batch_size), tf.float32)




data_len = tf.shape(data_ph)[0]
out_dim0 = tf.shape(lengths_ph)[0]
out_dim1 = tf.reduce_max(lengths_ph)
out_dim2 = tf.shape(data_ph)[-1]
# create a [[x,y,z], ...] tensor, where x=start_idx, y=length, z=pad_size
start_idxs = tf.concat([[0], tf.cumsum(lengths_ph)], 0)[:-1]
pads = tf.fill([out_dim0], out_dim1)-lengths_ph
reconstruction_metadata = tf.stack([start_idxs, lengths_ph, pads], axis=1)
# pass the xyz tensor to map_fn to create a tensor with the proper indexes.
# then gather the indexes from data_ph and reshape
reconstruction_data = tf.map_fn(lambda x: tf.concat([tf.range(x[0],x[0]+x[1]),tf.fill([x[2]], data_len)],0), reconstruction_metadata)
output = tf.gather(tf.concat([data_ph, tf.zeros((1,out_dim2),tf.int32)], 0),tf.reshape(reconstruction_data, [out_dim0*out_dim1]))
output2 = tf.reshape(output, [out_dim0, out_dim1*out_dim2])
# graph interface to access input and output nodes from outside
self.data_ph = data_ph
self.lengths_ph = lengths_ph
self.output = output
```
```python
# horizontally concat input
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
metadata_path = Path(r'E:\tbrain\aicup1\aicup1-v9-meta.json')
with metadata_path.open('r') as f:
    metadata = json.load(f)

task1_embeddings = tf.constant(metadata['task1_embeddings'])
max_sentences = 5
batch_size, sequence_length, num_classes = (2, 10, len(metadata['task1_embeddings']))
abstract_features = tf.random.uniform([batch_size, sequence_length, num_classes],minval=0,maxval=1,dtype=tf.float32)
title_features = tf.random.uniform([batch_size, num_classes],minval=0,maxval=1,dtype=tf.float32)
input = tf.concat([tf.expand_dims(title_features, 1), abstract_features], 1)
```
```python
# calculate scores from seq2seq logits
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
metadata_path = Path(r'E:\tbrain\aicup1\aicup1-v9-meta.json')
with metadata_path.open('r') as f:
    metadata = json.load(f)

task1_embeddings = tf.constant(metadata['task1_embeddings'])
max_sentences = 5
batch_size, sequence_length, num_classes = (2, 10, len(metadata['task1_embeddings']))
sentence_lengths = tf.constant([[3,4,3,0],[2,2,1,1]])
segment_ids = tf.constant([[0,0,0,1,1,1,1,2,2,2],[0,0,1,1,2,3,0,0,0,0]])
sentence_labels = tf.constant([[3,4,2,0],[3,1,1,1]])
sentence_labels_flat = tf.RaggedTensor.from_tensor(sentence_labels, padding=0).flat_values
sentence_classes = tf.nn.embedding_lookup(task1_embeddings, sentence_labels_flat)
lengths = tf.constant([10,6])
sentence_count = tf.constant([9,6])
mask = tf.sequence_mask(lengths=lengths, maxlen=sequence_length, dtype=tf.float32)
predicted_labels = tf.constant([[3,3,2,4,4,4,3,2,2,1],[3,3,2,1,4,1,0,0,0,0]])
logits = tf.random.uniform([batch_size, sequence_length, num_classes],minval=0,maxval=1,dtype=tf.float32)
all_probs_mask = tf.expand_dims(tf.sequence_mask(lengths=sentence_count,maxlen=sequence_length,dtype=tf.float32),-1)
all_probs = tf.nn.softmax(logits=logits, axis=-1) * all_probs_mask
all_probs_flat = tf.RaggedTensor.from_tensor(all_probs, lengths=sentence_count, ragged_rank=1).flat_values
predicted_sentence_labels_mask = tf.sequence_mask(lengths=sentence_count,maxlen=sequence_length,dtype=tf.int32)
predicted_sentence_labels = tf.argmax(input=logits, axis=-1, output_type=tf.int32) * predicted_sentence_labels_mask
predicted_sentence_labels_flat = tf.RaggedTensor.from_tensor(predicted_sentence_labels, padding=0).flat_values
predicted_sentence_classes = tf.nn.embedding_lookup(task1_embeddings, predicted_sentence_labels_flat)
tf.math.multiply(tf.tile(tf.expand_dims(all_probs_flat,-1),[1,1,6]),tf.cast(task1_embeddings,tf.float32))
predicted_sentence_class_scores = tf.math.reduce_sum(tf.math.multiply(tf.tile(tf.expand_dims(all_probs_flat,-1),[1,1,6]),tf.cast(task1_embeddings,tf.float32)), 1)
predicted_sentence_classes = tf.cast(predicted_sentence_class_scores > 0.5, tf.int32)
```
```python
# making target_input and target_output for seq2seq
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
sentence_labels = tf.constant([3,1,1,1])
target_input = tf.concat([[1],sentence_labels],0)
target_output = tf.concat([sentence_labels,[2]],0)
test = tf.concat([[1],sentence_labels,[2]],0)

with tf.Session() as sess:
    print(sess.run([target_input, target_output]))
```
```python
# predictions with logits, loss_type == 'sigmoid'
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
metadata_path = Path(r'E:\tbrain\aicup1\aicup1-v1-meta.json')
with metadata_path.open('r') as f:
    metadata = json.load(f)

task1_embeddings = tf.constant(metadata['task1_embeddings'])
max_sentences = 5
batch_size, sequence_length, num_classes = (2, 10, 6)
sentence_lengths = tf.constant([[3,4,3,0],[2,2,1,1]])
segment_ids = tf.constant([[0,0,0,1,1,1,1,2,2,2],[0,0,1,1,2,3,0,0,0,0]])
sentence_labels = tf.constant([[3,4,2,0],[3,1,1,1]])
sentence_labels_flat = tf.RaggedTensor.from_tensor(sentence_labels, padding=0).flat_values
sentence_classes = tf.nn.embedding_lookup(task1_embeddings, sentence_labels_flat)
lengths = tf.constant([10,6])
mask = tf.sequence_mask(lengths=lengths, maxlen=sequence_length, dtype=tf.float32)
predicted_labels = tf.constant([[3,3,2,4,4,4,3,2,2,1],[3,3,2,1,4,1,0,0,0,0]])
logits = tf.random.uniform([batch_size, sequence_length, num_classes],minval=0,maxval=1,dtype=tf.float32)

i=0
tf.slice(predicted_labels[i], [0], [lengths[i]])
tf.slice(segment_ids[i], [0], [lengths[i]])
tf.slice(logits[i], [0,0], [lengths[i],num_classes])
tf.dynamic_partition(data=tf.slice(predicted_labels[i], [0], [lengths[i]]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=max_sentences)
tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=max_sentences)
[tf.reduce_mean(s, axis=0) for s in tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=max_sentences) if len(s) > 0]
scores_flat = tf.concat([tf.concat([[tf.reduce_mean(s, axis=0) for s in tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=max_sentences) if len(s) > 0]],1) for i in range(batch_size)],0)
predicted_classes = tf.cast(scores_flat > 0.5, tf.int32)
```

```python
# predictions with logits
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
batch_size, sequence_length, num_classes = (2, 10, 6)
sentence_lengths = tf.constant([[3,4,3,0],[2,2,1,1]])
segment_ids = tf.constant([[0,0,0,1,1,1,1,2,2,2],[0,0,1,1,2,3,0,0,0,0]])
lengths = tf.constant([10,6])
mask = tf.sequence_mask(lengths=lengths, maxlen=sequence_length, dtype=tf.float32)
predicted_labels = tf.constant([[3,3,2,4,4,4,3,2,2,1],[3,3,2,1,4,1,0,0,0,0]])
logits = tf.random.uniform([batch_size, sequence_length, num_classes],minval=0,maxval=1,dtype=tf.float32)

i=0
tf.slice(predicted_labels[i], [0], [lengths[i]])
tf.slice(segment_ids[i], [0], [lengths[i]])
tf.slice(logits[i], [0,0], [lengths[i],num_classes])
tf.dynamic_partition(data=tf.slice(predicted_labels[i], [0], [lengths[i]]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5)
tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5)
[tf.reduce_sum(s, axis=0) for s in tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5) if len(s) > 0]
[tf.argmax(tf.reduce_sum(s, axis=0)) for s in tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5) if len(s) > 0]
tf.concat([[tf.argmax(tf.reduce_sum(s, axis=0)) for s in tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5) if len(s) > 0]],0)
[tf.concat([[tf.argmax(tf.reduce_sum(s, axis=0)) for s in tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5) if len(s) > 0]],0) for i in range(batch_size)]
tensors = [tf.concat([[tf.argmax(tf.reduce_sum(s, axis=0)) for s in tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5) if len(s) > 0]],0) for i in range(batch_size)]
values = tf.concat(tensors, axis=0)
lens = tf.stack(
    [tf.shape(t, out_type=tf.int32)[0] for t in tensors]
)
tf.RaggedTensor.from_row_lengths(values, lens).flat_values

i=1
tf.slice(predicted_labels[i], [0], [lengths[i]])
tf.slice(segment_ids[i], [0], [lengths[i]])
tf.slice(logits[i], [0,0], [lengths[i],num_classes])
tf.dynamic_partition(data=tf.slice(predicted_labels[i], [0], [lengths[i]]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5)
tf.dynamic_partition(data=tf.slice(logits[i], [0,0], [lengths[i],num_classes]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=5)

slice =  tf.constant([[0.61672187, 0.24809551, 0.304824  , 0.39282465, 0.63652265,0.6423135 ],
    [0.71984804, 0.9921545 , 0.18937862, 0.7341274 , 0.24413383,0.75465643],
    [0.13824522, 0.49925053, 0.44687247, 0.8441725 , 0.14208174,0.05951583]])

tf.reduce_sum(slice, axis=0)

losses = tf.random.uniform([batch_size, sequence_length, 6],minval=0,maxval=1,dtype=tf.float32)
tf.reduce_sum(losses * tf.expand_dims(mask, axis=2))
```
```python
# predictions non eager
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools
import json
from pathlib import Path
batch_size, sequence_length, num_classes = (2, 6, 32)
lengths = tf.random.uniform([batch_size],minval=2,maxval=sequence_length,dtype=tf.int32)
mask = tf.sequence_mask(lengths=lengths, maxlen=sequence_length, dtype=tf.float32)
classes = tf.random.uniform([batch_size,sequence_length],maxval=num_classes,dtype=tf.int32)

dataset = tf.data.TFRecordDataset.list_files(file_pattern=r'E:\tbrain\aicup1\aicup1-v1-train.1of7.tfrecords',shuffle=True)
def tfrecord_dataset(filename):
    return tf.data.TFRecordDataset(filenames=filename,compression_type=None,buffer_size=256 * 1024 * 1024)

dataset2 = dataset.interleave(map_func=tfrecord_dataset,cycle_length=1,block_length=1,num_parallel_calls=None)

parse_fn = parse_aicup1_v1
mode = tf.estimator.ModeKeys.TRAIN
dataset3 = dataset2.map(
    functools.partial(parse_fn, mode=mode),
    num_parallel_calls=int(2)
)
padded_shapes = ({'embeddings': [None, 768], 'length': [], 'sentence_count': [], 'sentence_lengths': [None], 'sentence_labels': [None], 'segment_ids': [None]}, [None])
dataset4 = dataset3.apply(tf.data.experimental.bucket_by_sequence_length(
    element_length_func=lambda seq, dom: seq['length'],
    bucket_boundaries=[2 ** x for x in range(5, 15)],
    # [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    bucket_batch_sizes=[3 for x in range(10, -1, -1)],
    padded_shapes=padded_shapes,
    padding_values=None, # Defaults to padding with 0.
    pad_to_bucket_boundary=False
))
dataset5 = dataset4.prefetch(buffer_size=64)
iter = dataset5.make_initializable_iterator()
features, labels = iter.get_next()
embeddings = features['embeddings']
lengths = features['length']
sentence_count = features['sentence_count']
sentence_lengths = features['sentence_lengths']
sentence_labels = features['sentence_labels']
segment_ids = features['segment_ids']
batch_size = tf.shape(lengths)[0]
mask = tf.sequence_mask(lengths=lengths, maxlen=tf.shape(labels)[1], dtype=tf.float32)
masked_labels = labels * tf.cast(mask, tf.int32)
predicted_labels = labels * tf.cast(mask, tf.int32)
predicted_labels_r = tf.RaggedTensor.from_tensor(predicted_labels, padding=0)
sentence_lengths_r = tf.RaggedTensor.from_tensor(sentence_lengths, padding=0)

test=tf.dynamic_partition(data=predicted_labels[0],partitions=segment_ids[0],num_partitions=50)

tf.slice(predicted_labels, [0,0], [1,5])
tf.map_fn(lambda i: predicted_labels[i], tf.range(batch_size))

def ragged(tensors):
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int32)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)

def make_predicted_sentence_labels_flat(predicted_labels, segment_ids, batch_size, lengths):
    tensors = [tf.concat([[tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])] for s in tf.dynamic_partition(data=tf.slice(predicted_labels[i], [0], [lengths[i]]),partitions=tf.slice(segment_ids[i], [0], [lengths[i]]),num_partitions=50) if len(s) > 0]],0) for i in range(batch_size)]
    # tensors = [s for s in tf.dynamic_partition(data=tf.slice(predicted_labels[1], [0], [lengths[1]]),partitions=tf.slice(segment_ids[1], [0], [lengths[1]]),num_partitions=50) if len(s) > 0]
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int32)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens).flat_values
    # return tensors

predicted_sentence_labels_flat = tf.py_function(func=make_predicted_sentence_labels_flat, inp=[predicted_labels, segment_ids, batch_size, lengths], Tout=tf.int32)
metadata_path = Path(r'E:\tbrain\aicup1\aicup1-v1-meta.json')
with metadata_path.open('r') as f:
    metadata = json.load(f)

task1_embeddings = tf.constant(metadata['task1_embeddings'])
predicted_classes = tf.nn.embedding_lookup(task1_embeddings, predicted_sentence_labels_flat)

with tf.Session() as sess:
    sess.run(iter.initializer)
    print(sess.run([predicted_sentence_labels_flat, predicted_classes]))
    print(sess.run([predicted_labels[1], segment_ids[1], sentence_labels, batch_size, lengths, predicted_sentence_labels_flat]))

    print(sess.run([sentence_count,y]))
    print(sess.run(sentence_lengths))
    print(sess.run(test))
```

```python
# predictions
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import functools

batch_size, sequence_length, num_classes = (2, 6, 32)
lengths = tf.random.uniform([batch_size],minval=2,maxval=sequence_length,dtype=tf.int32)
mask = tf.sequence_mask(lengths=lengths, maxlen=sequence_length, dtype=tf.float32)
classes = tf.random.uniform([batch_size,sequence_length],maxval=num_classes,dtype=tf.int32)

dataset = tf.data.TFRecordDataset.list_files(file_pattern=r'E:\tbrain\aicup1\aicup1-v1-train.1of7.tfrecords',shuffle=True)
def tfrecord_dataset(filename):
    return tf.data.TFRecordDataset(filenames=filename,compression_type=None,buffer_size=256 * 1024 * 1024)

dataset2 = dataset.interleave(map_func=tfrecord_dataset,cycle_length=1,block_length=1,num_parallel_calls=None)

parse_fn = parse_aicup1_v1
mode = tf.estimator.ModeKeys.TRAIN
dataset3 = dataset2.map(
    functools.partial(parse_fn, mode=mode),
    num_parallel_calls=int(2)
)
padded_shapes = ({'embeddings': [None, 768], 'length': [], 'sentence_count': [], 'sentence_lengths': [None], 'sentence_labels': [None], 'segment_ids': [None]}, [None])
dataset4 = dataset3.apply(tf.data.experimental.bucket_by_sequence_length(
    element_length_func=lambda seq, dom: seq['length'],
    bucket_boundaries=[2 ** x for x in range(5, 15)],
    # [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    bucket_batch_sizes=[3 for x in range(10, -1, -1)],
    padded_shapes=padded_shapes,
    padding_values=None, # Defaults to padding with 0.
    pad_to_bucket_boundary=False
))
dataset5 = dataset4.prefetch(buffer_size=64)
features, labels = dataset5.__iter__().__next__()
embeddings = features['embeddings']
lengths = features['length']
sentence_count = features['sentence_count']
sentence_lengths = features['sentence_lengths']
sentence_labels = features['sentence_labels']
segment_ids = features['segment_ids']
batch_size = tf.shape(lengths)[0]
embeddings.shape
# TensorShape([Dimension(3), Dimension(249), Dimension(768)])
lengths.shape
# TensorShape([Dimension(3)])
sentence_count.shape
# TensorShape([Dimension(3)])
sentence_lengths.shape
# TensorShape([Dimension(3), Dimension(9)])
sentence_labels.shape
# TensorShape([Dimension(3), Dimension(9)])
mask = tf.sequence_mask(lengths=lengths, maxlen=tf.shape(labels)[1], dtype=tf.float32)
masked_labels = labels * tf.cast(mask, tf.int32)
predicted_labels = labels * tf.cast(mask, tf.int32)
tf.map_fn(lambda i: tf.split(predicted_labels[i],sentence_lengths[i]), tf.range(batch_size))
# ValueError: Cannot infer num from shape Tensor("map_3/while/strided_slice_1:0", shape=(?,), dtype=int32)

i = tf.constant(0)
a = []
c = lambda i, a: tf.less(i, batch_size)
b = lambda i, a: a.append(tf.split(predicted_labels[i],sentence_lengths[i]))
r = tf.while_loop(c, b, [i, a])
# ValueError: Cannot infer num from shape Tensor("map_3/while/strided_slice_1:0", shape=(?,), dtype=int32)
tf.split(predicted_labels[0],sentence_lengths[0])
# ValueError: Cannot infer num from shape Tensor("map_3/while/strided_slice_1:0", shape=(?,), dtype=int32)


predicted_labels_r = tf.RaggedTensor.from_tensor(labels * tf.cast(mask, tf.int32), padding=0)
labels_r = tf.RaggedTensor.from_tensor(labels * tf.cast(mask, tf.int32), padding=0)
sentence_lengths_r = tf.RaggedTensor.from_tensor(sentence_lengths, padding=0)
sentence_labels_r = tf.RaggedTensor.from_tensor(sentence_labels, padding=0)
@tf.function
def ragged(tensors):
    values = tf.concat(tensors, axis=0)
    lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
    return tf.RaggedTensor.from_row_lengths(values, lens)

@tf.function
def get_sentence_labels():
    return ragged([tf.concat([[tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])] for s in tf.split(predicted_labels_r[i],sentence_lengths_r[i])]],0) for i in range(batch_size)])

i = tf.constant(0)
a = []
c = lambda i, a: tf.less(i, batch_size)
b = lambda i, a: a.append(tf.concat([[tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])] for s in tf.split(predicted_labels_r[i],sentence_lengths_r[i])]],0))
r = tf.while_loop(c, b, [i, a])
predicted_sentence_labels_r = get_sentence_labels()

task1_embeddings = tf.constant([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 1, 0], [1, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 0], [1, 0, 1, 0, 1, 0]])

classes = tf.nn.embedding_lookup(task1_embeddings,sentence_labels_r.flat_values)
predicted_classes = tf.nn.embedding_lookup(task1_embeddings,predicted_sentence_labels_r.flat_values)

a=[t.numpy() for t in predicted_sentence_labels_r]
a[0][0] = 2
a_r=tf.ragged.constant(a)
predicted_classes = tf.nn.embedding_lookup(task1_embeddings,a_r.flat_values)
is_correct = tf.cast(tf.equal(classes, predicted_classes), tf.float32)
num_values = tf.ones_like(is_correct)
batch_accuracy = tf.math.divide(tf.reduce_sum(is_correct), tf.reduce_sum(num_values))
TP = tf.cast(tf.count_nonzero(predicted_classes * classes), tf.float32)
FP = tf.cast(tf.count_nonzero(predicted_classes * (classes - 1)), tf.float32)
FN = tf.cast(tf.count_nonzero((predicted_classes - 1) * classes), tf.float32)
batch_precision = tf.math.divide_no_nan(TP, (TP + FP))
batch_recall = tf.math.divide_no_nan(TP, (TP + FN))
batch_f1 = tf.math.divide_no_nan(
    2 * batch_precision * batch_recall,
    (batch_precision + batch_recall)
)
# predicted_labels = [[tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])] for s in tf.split(labels_r[i],sentence_lengths_r[i])] for i in range(batch_size)]
# predicted_labels = [[tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])].numpy() for s in tf.split(labels_r[i],sentence_lengths_r[i])] for i in range(batch_size)]
# predicted_labels = tf.ragged.constant([tf.concat([[tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])] for s in tf.split(labels_r[i],sentence_lengths_r[i])]],0) for i in range(batch_size)])
# [tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])].numpy() for s in tf.split(labels_r[1],sentence_lengths_r[1])]
# tf.concat([[tf.unique_with_counts(s)[0][tf.argmax(tf.unique_with_counts(s)[2])] for s in tf.split(labels_r[1],sentence_lengths_r[1])]],0)
tf.compat.v1.metrics.accuracy([1,1,1,1],[1,1,0,1])
ma = tf.keras.metrics.Accuracy()
ma.update_state([1,1,1,1],[1,1,0,1])
ma.result()
ma = tf.keras.metrics.Accuracy()
ma.update_state(y_true=sentence_labels_r,y_pred=predicted_sentence_labels_r)
ma.result()
ma = tf.keras.metrics.Accuracy()
ma.update_state(y_true=sentence_labels_r,y_pred=a_r)
ma.result()
ma = tf.keras.metrics.Accuracy()
ma.update_state(y_true=sentence_labels_r,y_pred=a_r)
ma.result()
mfp = tf.keras.metrics.FalsePositives()
mfp.update_state(y_true=sentence_labels_r,y_pred=a_r)
mfp.result()
```

```python
import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np

batch_size, sequence_length, num_categorical, num_numerics = (2, 6, 5, 3)
numerics=tf.random.uniform([batch_size,sequence_length,num_numerics],maxval=10,dtype=tf.float32)
categorical=tf.random.uniform([batch_size,sequence_length,num_categorical],maxval=10,dtype=tf.float32)
input_layers=[numerics,categorical]
input = tf.concat(values=input_layers,axis=-1)

tf_record = r'E:\tbrain\aicup1\aicup1-v1-train.1of7.tfrecords'
e = tf.python_io.tf_record_iterator(tf_record).__next__()
features = {
    'length': tf.FixedLenFeature(shape=(), dtype=tf.string),
    'labels': tf.FixedLenFeature(shape=(), dtype=tf.string),
    'numerics': tf.FixedLenFeature(shape=(), dtype=tf.string),
    'categorical': tf.FixedLenFeature(shape=(), dtype=tf.string)
}
parsed = tf.parse_single_example(
    serialized=e,
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
features['length'] = tf.decode_raw(
    parsed['length'], out_type=tf.int32, little_endian=True, name=None
)[0]
features['numerics'] = tf.reshape(tf.decode_raw(
    parsed['numerics'], out_type=tf.float32, little_endian=True, name=None
), [features['length'], -1])
features['categorical'] = tf.reshape(tf.decode_raw(
    parsed['categorical'], out_type=tf.int32, little_endian=True, name=None
), [features['length'], -1])

logits=tf.random.uniform([batch_size,sequence_length,1],maxval=10,dtype=tf.float32)
```
```python
train_articles = parse_into_articles(Path(r'E:\tbrain\aicup1\task1_trainset.csv'))
test_articles = parse_into_articles(Path(r'E:\tbrain\aicup1\task1_public_testset.csv'))
compute_embeddings(train_articles, 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1')
compute_embeddings(test_articles, 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1')

print(f'generate categories')
count = defaultdict(int)
for article in train_articles:
    for label in article['Task 1']:
        count[label] += 1


task1_categories = ['PAD'] + sorted(count, key=lambda k: (count[k], k), reverse=True)
task1_mappings = { c: i for i, c in enumerate(task1_categories)}
print(f'generate labels')
for article in train_articles:
    labels = []
    acc = 0
    for i, label in enumerate(article['Task 1']):
        labels += [task1_mappings[label]] * (article['end_coords'][i] - acc)
        acc = article['end_coords'][i]
    if len(labels) != article['length']:
        print(f"===ERROR: len(labels)({len(labels)}) != article['length']({article['length']})")
        print(f"===ERROR: article['input_ids']({len(article['input_ids'])}): {article['input_ids']}")
        print(f"===ERROR: article['input_mask']({len(article['input_mask'])}): {article['input_mask']}")
        print(f"===ERROR: article['segment_ids']({len(article['segment_ids'])}): {article['segment_ids']}")
        print(f"===ERROR: acc: {acc}")
        print(f"===ERROR: article['Task 1']: {article['Task 1']}")
        print(f"===ERROR: article['end_coords']: {article['end_coords']}")
        print(f"===ERROR: labels: {labels}")
        break
    article['labels'] = labels


import tensorflow as tf
tf.enable_eager_execution()
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
import numpy as np
import tensorflow_hub as hub
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
example = bert.run_classifier.InputExample(guid=None,text_a="A long sentence.")
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
max_seq_length = 512
input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
tokens = ['[CLS]'] + tokenizer.tokenize("A long sentence.") + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
lengths = len(input_ids)
input_ids += [0] * (512 - lengths)
input_mask = [1] * lengths + [0] * (512 - lengths)
segment_ids = [0] * 512
pooled_output, sequence_output = model.predict([[input_ids], [input_mask], [segment_ids]])
embeddings = sequence_output[0,:lengths,:]
embeddings.shape == (6, 768)

label_list = [None, 0, 1]
features = bert.run_classifier.convert_examples_to_features([example], label_list, max_seq_length, tokenizer)
pooled_output, sequence_output = model.predict([[features[0].input_ids], [features[0].input_mask], [features[0].segment_ids]])
hub_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
embed = hub.KerasLayer(hub_url)
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)
bert_embeddings = bert_layer(["A long sentence.", "single-word", "http://example.com"])
print(bert_embeddings.shape, bert_embeddings.dtype)
predict_input_fn = run_classifier.input_fn_builder(features=features, seq_length=max_seq_length, is_training=False, drop_remainder=False)
pooled_output, sequence_output = bert_layer(train_features[0].input_ids, train_features[0].input_mask, train_features[0].segment_ids)
pooled_output, sequence_output = bert_layer(predict_input_fn)
tokens: [CLS] a long sentence . [SEP]
input_ids: 101 1037 2146 6251 1012 102
input_mask: 1 1 1 1 1 1
```