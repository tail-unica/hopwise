hopwise.evaluator.utils
=======================

.. py:module:: hopwise.evaluator.utils

.. autoapi-nested-parse::

   hopwise.evaluator.utils
   ################################



Functions
---------

.. autoapisummary::

   hopwise.evaluator.utils.pad_sequence
   hopwise.evaluator.utils.trunc
   hopwise.evaluator.utils.cutoff
   hopwise.evaluator.utils._binary_clf_curve
   hopwise.evaluator.utils.plot_tsne_embeddings
   hopwise.evaluator.utils.train_tsne


Module Contents
---------------

.. py:function:: pad_sequence(sequences, len_list, pad_to=None, padding_value=0)

   Pad sequences to a matrix

   :param sequences: list of variable length sequences.
   :type sequences: list
   :param len_list: the length of the tensors in the sequences
   :type len_list: list
   :param pad_to: if pad_to is not None, the sequences will pad to the length you set,
                  else the sequence will pad to the max length of the sequences.
   :type pad_to: int, optional
   :param padding_value: value for padded elements. Default: 0.
   :type padding_value: int, optional

   :returns: [seq_num, max_len] or [seq_num, pad_to]
   :rtype: torch.Tensor


.. py:function:: trunc(scores, method)

   Round the scores by using the given method

   :param scores: scores
   :type scores: numpy.ndarray
   :param method: one of ['ceil', 'floor', 'around']
   :type method: str

   :raises NotImplementedError: method error

   :returns: processed scores
   :rtype: numpy.ndarray


.. py:function:: cutoff(scores, threshold)

   Cut of the scores based on threshold

   :param scores: scores
   :type scores: numpy.ndarray
   :param threshold: between 0 and 1
   :type threshold: float

   :returns: processed scores
   :rtype: numpy.ndarray


.. py:function:: _binary_clf_curve(trues, preds)

   Calculate true and false positives per binary classification threshold

   :param trues: the true scores' list
   :type trues: numpy.ndarray
   :param preds: the predict scores' list
   :type preds: numpy.ndarray

   :returns: A count of false positives, at index i being the number of negative
             samples assigned a score >= thresholds[i]
             preds (numpy.ndarray): An increasing count of true positives, at index i being the number
             of positive samples assigned a score >= thresholds[i].
   :rtype: fps (numpy.ndarray)

   .. note::

      To improve efficiency, we referred to the source code(which is available at sklearn.metrics.roc_curve)
      in SkLearn and made some optimizations.


.. py:function:: plot_tsne_embeddings(model, **kwargs)

.. py:function:: train_tsne(model, config, load_best_model)

