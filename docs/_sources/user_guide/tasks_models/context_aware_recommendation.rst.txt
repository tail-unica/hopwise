Context-aware Recommendation
============================================

Context-aware Recommendation
-------------------------------
Context-aware recommendation can be seen as an extension of click-through rate prediction. All the model in this
class can be used for CTR prediction. Usually, the dataset is explicit and contains label field. Other feature fields
are also support for these models. And evaluation is always conducted in the way of binary classification.

.. toctree::
   :maxdepth: 1

   model/context_aware/lr
   model/context_aware/fm
   model/context_aware/nfm
   model/context_aware/deepfm
   model/context_aware/xdeepfm
   model/context_aware/afm
   model/context_aware/ffm
   model/context_aware/fwfm
   model/context_aware/fnn
   model/context_aware/pnn
   model/context_aware/dssm
   model/context_aware/widedeep
   model/context_aware/din
   model/context_aware/dien
   model/context_aware/dcn
   model/context_aware/dcnv2
   model/context_aware/autoint
   model/context_aware/xgboost
   model/context_aware/lightgbm
   model/context_aware/kd_dagfm
   model/context_aware/fignn
   model/context_aware/eulernet


For context-aware recommendation, we choose **LR** model to show you how to
train and test it on the **ml-100k** dataset from both **API** and
**source code**.

Quick-start From API
---------------------

1. Prepare your data:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Before running a model, firstly you need to prepare and load data. To
help users quickly get start, RecBole has a build-in dataset **ml-100k**
and you can directly use it. However, if you want to use other datasets,
you can read :doc:`/user_guide/usage/running_new_dataset` for more information.

Then, you need to set data config for data loading. You can create a
yaml file called test.yaml and write the following settings:

.. code:: yaml

   # dataset config : Context-aware Recommendation
   load_col:
       inter: ['user_id', 'item_id', 'rating', 'timestamp']
       user: ['user_id', 'age', 'gender', 'occupation']
       item: ['item_id', 'release_year', 'class']
   threshold: {'rating': 4}
   normalize_all: True

Generally, context-aware recommendation models utilize the features of
users, items and interactions to make CTR predictions, so it needs to
load the used features. And context-aware recommendation models are
mainly used in explicit feedback scenes, so your data should have
explicit feedback information and you need to set label for them. Here
we set ``rating=4`` as threshold to label the interaction. For more
information about label setting, please read the :doc:`/user_guide/datasets/label_of_data`.

2. Choose a model:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

You can choose a model from our :doc:`/user_guide/tasks_models_intro`
Here we choose LR model to demonstrate how to train and test the
context-aware recommendation model.

Then, you need to set the parameter for LR model. You can check the
:doc:`model/context_aware/lr` and add the model settings into the test.yaml, like:

.. code:: yaml

   # model config
   embedding_size: 10

If you want to run different models, you can read :doc:`/user_guide/usage/running_different_models`
for more information.

3. Set training and evaluation config:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

In RecBole, we support multiple training and evaluation methods. You can
choose how to train and test model by simply setting the config.

Here we want to train and test the LR model in training-validation-test
method (optimize model parameters on the training set, do parameter
selection according to the results on the validation set, and finally
report the results on the test set) and evaluate the model performance
by full ranking with all item candidates, so we can add the following
settings into the test.yaml.

.. code:: yaml

   # Training and evaluation config
   epochs: 500
   train_batch_size: 4096
   eval_batch_size: 4096
   eval_args:
     split: {'RS':[0.8,0.1,0.1]}
     order: RO
     group_by: ~
     mode: labeled
   train_neg_sample_args: ~
   metrics: ['AUC', 'LogLoss']
   valid_metric: AUC

Note that RecBole also supports to evaluate the context-aware
recommendation models by full-ranking like general recommendation
models, but you need to make sure that your ``.inter`` file can not load
any other context information column. For more details of training and
evaluation config, please refer to :doc:`/user_guide/config/training_settings`
and :doc:`/user_guide/config/evaluation_settings`.

.. _header-n19:

4. Run the model and collect the result
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Now you have finished all the preparations, it’s time to run the model!

You can create a new python file (e.g., run.py), and write the following
code:

.. code:: python

   from recbole.quick_start import run_recbole
   run_recbole(model='LR', dataset='ml-100k', config_file_list=['test.yaml'])

Then run the following command:

.. code:: python

   python run.py

And you will obtain the output like:

.. code::

   16 Jul 20:12    INFO  ml-100k
   The number of users: 944
   Average actions of users: 106.04453870625663
   The number of items: 1683
   Average actions of items: 59.45303210463734
   The number of inters: 100000
   The sparsity of the dataset: 93.70575143257098%
   Remain Fields: ['user_id', 'item_id', 'timestamp', 'age', 'gender', 'occupation', 'release_year', 'class', 'label']
   16 Jul 20:12    INFO  [Training]: train_batch_size = [4096] negative sampling: [None]
   16 Jul 20:12    INFO  [Evaluation]: eval_batch_size = [4096] eval_args: [{'split': {'RS': [0.8, 0.1, 0.1]}, 'order': 'RO', 'group_by': None, 'mode': 'labeled'}]
   16 Jul 20:12    INFO  LR(
     (token_embedding_table): FMEmbedding(
       (embedding): Embedding(2788, 10)
     )
     (float_embedding_table): Embedding(1, 10)
     (token_seq_embedding_table): ModuleList(
       (0): Embedding(20, 10)
     )
     (first_order_linear): FMFirstOrderLinear(
       (token_embedding_table): FMEmbedding(
         (embedding): Embedding(2788, 1)
       )
       (float_embedding_table): Embedding(1, 1)
       (token_seq_embedding_table): ModuleList(
         (0): Embedding(20, 1)
       )
     )
     (sigmoid): Sigmoid()
     (loss): BCELoss()
   )
   Trainable parameters: 30900
   Train     0: 100%|█████████████████████████████████████████████████| 20/20 [00:00<00:00, 165.41it/s]
   16 Jul 20:12    INFO  epoch 0 training [time: 0.12s, train loss: 14.3632]
   Evaluate   : 100%|███████████████████████████████████████████████████| 3/3 [00:00<00:00, 373.46it/s]
   16 Jul 20:12    INFO  epoch 0 evaluating [time: 0.01s, valid_score: 0.476300]
   16 Jul 20:12    INFO  valid result:
   auc : 0.4763    logloss : 0.7162
   16 Jul 20:12    INFO  Saving current: saved\LR-Jul-16-2022_20-12-38.pth
   Train     1: 100%|█████████████████████████████████████████████████| 20/20 [00:00<00:00, 165.49it/s]
   16 Jul 20:12    INFO  epoch 1 training [time: 0.12s, train loss: 14.1432]
   Evaluate   : 100%|███████████████████████████████████████████████████| 3/3 [00:00<00:00, 372.51it/s]
   16 Jul 20:12    INFO  epoch 1 evaluating [time: 0.01s, valid_score: 0.497500]
   ......
   Train   253: 100%|█████████████████████████████████████████████████| 20/20 [00:00<00:00, 165.77it/s]
   16 Jul 20:13    INFO  epoch 253 training [time: 0.12s, train loss: 10.7201]
   Evaluate   : 100%|███████████████████████████████████████████████████| 3/3 [00:00<00:00, 374.20it/s]
   16 Jul 20:13    INFO  epoch 253 evaluating [time: 0.01s, valid_score: 0.774400]
   16 Jul 20:13    INFO  valid result:
   auc : 0.7744    logloss : 0.5654
   16 Jul 20:13    INFO  Finished training, best eval result in epoch 242
   16 Jul 20:13    INFO  Loading model structure and parameters from saved\LR-Jul-16-2022_20-12-38.pth
   Evaluate   : 100%|███████████████████████████████████████████████████| 3/3 [00:00<00:00, 298.71it/s]
   16 Jul 20:13    INFO  best valid : OrderedDict([('auc', 0.7745), ('logloss', 0.5651)])
   16 Jul 20:13    INFO  test result: OrderedDict([('auc', 0.7765), ('logloss', 0.562)])


Finally you will get the model’s performance on the test set and the
model file will be saved under the /saved. Besides, RecBole allows
tracking and visualizing train loss and valid score with TensorBoard,
please read the :doc:`/user_guide/usage/use_tensorboard` for more details.

The above is the whole process of running a model in RecBole, and you
can read other docs for depth usage.

.. _header-n29:

Quick-start From Source
-------------------------

Besides using API, you can also directly run the source code of
`hopwise <https://github.com/tail-unica/hopwise>`__. The whole process is
similar to Quick-start From API. You can create a yaml file called
test.yaml and set all the config as follow:

.. code:: yaml

   # dataset config : Context-aware Recommendation
   load_col:
       inter: ['user_id', 'item_id', 'rating', 'timestamp']
       user: ['user_id', 'age', 'gender', 'occupation']
       item: ['item_id', 'release_year', 'class']
   threshold: {'rating': 4}

   # model config
   embedding_size: 10

   # Training and evaluation config
   epochs: 500
   train_batch_size: 4096
   eval_batch_size: 4096
   eval_args:
     split: {'RS':[0.8,0.1,0.1]}
     order: RO
     group_by: ~
     mode: labeled
   train_neg_sample_args: ~
   metrics: ['AUC', 'LogLoss']
   valid_metric: AUC

Then run the following command:

.. code:: python

   python run_recbole.py --model=LR --dataset=ml-100k --config_files=test.yaml

And you will get the output of running the LR model on the ml-100k
dataset.

If you want to change the parameters, such as ``embedding_size``, just
set the additional command parameters as you need:

.. code:: python

   python run_recbole.py --model=LR --dataset=ml-100k --config_files=test.yaml --embedding_size=100
