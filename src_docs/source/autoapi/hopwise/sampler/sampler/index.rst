hopwise.sampler.sampler
=======================

.. py:module:: hopwise.sampler.sampler

.. autoapi-nested-parse::

   hopwise.sampler
   ########################



Classes
-------

.. autoapisummary::

   hopwise.sampler.sampler.AbstractSampler
   hopwise.sampler.sampler.Sampler
   hopwise.sampler.sampler.KGSampler
   hopwise.sampler.sampler.RepeatableSampler
   hopwise.sampler.sampler.SeqSampler


Module Contents
---------------

.. py:class:: AbstractSampler(distribution, alpha)

   :class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports
   returning a certain number of random value_ids according to the input key_id, and it also supports
   to prohibit certain key-value pairs by setting used_ids.

   :param distribution: The string of distribution, which is used for subclass.
   :type distribution: str

   .. attribute:: used_ids

      The result of :meth:`get_used_ids`.

      :type: numpy.ndarray


   .. py:attribute:: distribution
      :value: ''



   .. py:attribute:: alpha


   .. py:attribute:: used_ids


   .. py:method:: set_distribution(distribution)

      Set the distribution of sampler.

      :param distribution: Distribution of the negative items.
      :type distribution: str



   .. py:method:: _uni_sampling(sample_num)
      :abstractmethod:


      Sample [sample_num] items in the uniform distribution.

      :param sample_num: the number of samples.
      :type sample_num: int

      :returns: a list of samples.
      :rtype: sample_list (np.array)



   .. py:method:: _get_candidates_list()
      :abstractmethod:


      Get sample candidates list for _pop_sampling()

      :returns: a list of candidates id.
      :rtype: candidates_list (list)



   .. py:method:: _build_alias_table()

      Build alias table for popularity_biased sampling.



   .. py:method:: _pop_sampling(sample_num)

      Sample [sample_num] items in the popularity-biased distribution.

      :param sample_num: the number of samples.
      :type sample_num: int

      :returns: a list of samples.
      :rtype: sample_list (np.array)



   .. py:method:: sampling(sample_num)

      Sampling [sample_num] item_ids.

      :param sample_num: the number of samples.
      :type sample_num: int

      :returns: a list of samples and the len is [sample_num].
      :rtype: sample_list (np.array)



   .. py:method:: get_used_ids()
      :abstractmethod:


      Returns:
      numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.



   .. py:method:: sample_by_key_ids(key_ids, num)

      Sampling by key_ids.

      :param key_ids: Input key_ids.
      :type key_ids: numpy.ndarray or list
      :param num: Number of sampled value_ids for each key_id.
      :type num: int

      :returns: Sampled value_ids.
                value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
                is sampled for key_ids[0];
                value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
                value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
      :rtype: torch.tensor



.. py:class:: Sampler(phases, datasets, distribution='uniform', alpha=1.0)

   Bases: :py:obj:`AbstractSampler`


   :class:`Sampler` is used to sample negative items for each input user. In order to avoid positive items
   in train-phase to be sampled in valid-phase, and positive items in train-phase or valid-phase to be sampled
   in test-phase, we need to input the datasets of all phases for pre-processing. And, before using this sampler,
   it is needed to call :meth:`set_phase` to get the sampler of corresponding phase.

   :param phases: All the phases of input.
   :type phases: str or list of str
   :param datasets: All the dataset for each phase.
   :type datasets: Dataset or list of Dataset
   :param distribution: Distribution of the negative items. Defaults to 'uniform'.
   :type distribution: str, optional

   .. attribute:: phase

      the phase of sampler. It will not be set until :meth:`set_phase` is called.

      :type: str


   .. py:attribute:: phases


   .. py:attribute:: datasets


   .. py:attribute:: uid_field


   .. py:attribute:: iid_field


   .. py:attribute:: user_num


   .. py:attribute:: item_num


   .. py:method:: _get_candidates_list()

      Get sample candidates list for _pop_sampling()

      :returns: a list of candidates id.
      :rtype: candidates_list (list)



   .. py:method:: _uni_sampling(sample_num)

      Sample [sample_num] items in the uniform distribution.

      :param sample_num: the number of samples.
      :type sample_num: int

      :returns: a list of samples.
      :rtype: sample_list (np.array)



   .. py:method:: get_used_ids()

      Returns:
      dict: Used item_ids is the same as positive item_ids.
      Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.



   .. py:method:: set_phase(phase)

      Get the sampler of corresponding phase.

      :param phase: The phase of new sampler.
      :type phase: str

      :returns: the copy of this sampler, :attr:`phase` is set the same as input phase, and :attr:`used_ids`
                is set to the value of corresponding phase.
      :rtype: Sampler



   .. py:method:: sample_by_user_ids(user_ids, item_ids, num)

      Sampling by user_ids.

      :param user_ids: Input user_ids.
      :type user_ids: numpy.ndarray or list
      :param item_ids: Input item_ids.
      :type item_ids: numpy.ndarray or list
      :param num: Number of sampled item_ids for each user_id.
      :type num: int

      :returns: Sampled item_ids.
                item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
                is sampled for user_ids[0];
                item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
                item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
      :rtype: torch.tensor



.. py:class:: KGSampler(dataset, distribution='uniform', alpha=1.0)

   Bases: :py:obj:`AbstractSampler`


   :class:`KGSampler` is used to sample negative entities in a knowledge graph.

   :param dataset: The knowledge graph dataset, which contains triplets in a knowledge graph.
   :type dataset: Dataset
   :param distribution: Distribution of the negative entities. Defaults to 'uniform'.
   :type distribution: str, optional


   .. py:attribute:: dataset


   .. py:attribute:: hid_field


   .. py:attribute:: tid_field


   .. py:attribute:: hid_list


   .. py:attribute:: tid_list


   .. py:attribute:: head_entities


   .. py:attribute:: entity_num


   .. py:method:: _uni_sampling(sample_num)

      Sample [sample_num] items in the uniform distribution.

      :param sample_num: the number of samples.
      :type sample_num: int

      :returns: a list of samples.
      :rtype: sample_list (np.array)



   .. py:method:: _get_candidates_list()

      Get sample candidates list for _pop_sampling()

      :returns: a list of candidates id.
      :rtype: candidates_list (list)



   .. py:method:: get_used_ids()

      Returns:
      numpy.ndarray: Used entity_ids is the same as tail_entity_ids in knowledge graph.
      Index is head_entity_id, and element is a set of tail_entity_ids.



   .. py:method:: sample_by_entity_ids(head_entity_ids, num=1)

      Sampling by head_entity_ids.

      :param head_entity_ids: Input head_entity_ids.
      :type head_entity_ids: numpy.ndarray or list
      :param num: Number of sampled entity_ids for each head_entity_id. Defaults to ``1``.
      :type num: int, optional

      :returns: Sampled entity_ids.
                entity_ids[0], entity_ids[len(head_entity_ids)], entity_ids[len(head_entity_ids) * 2], ...,
                entity_id[len(head_entity_ids) * (num - 1)] is sampled for head_entity_ids[0];
                entity_ids[1], entity_ids[len(head_entity_ids) + 1], entity_ids[len(head_entity_ids) * 2 + 1], ...,
                entity_id[len(head_entity_ids) * (num - 1) + 1] is sampled for head_entity_ids[1]; ...; and so on.
      :rtype: torch.tensor



.. py:class:: RepeatableSampler(phases, dataset, distribution='uniform', alpha=1.0)

   Bases: :py:obj:`AbstractSampler`


   :class:`RepeatableSampler` is used to sample negative items for each input user. The difference from
   :class:`Sampler` is it can only sampling the items that have not appeared at all phases.

   :param phases: All the phases of input.
   :type phases: str or list of str
   :param dataset: The union of all datasets for each phase.
   :type dataset: Dataset
   :param distribution: Distribution of the negative items. Defaults to 'uniform'.
   :type distribution: str, optional

   .. attribute:: phase

      the phase of sampler. It will not be set until :meth:`set_phase` is called.

      :type: str


   .. py:attribute:: phases


   .. py:attribute:: dataset


   .. py:attribute:: iid_field


   .. py:attribute:: user_num


   .. py:attribute:: item_num


   .. py:method:: _uni_sampling(sample_num)

      Sample [sample_num] items in the uniform distribution.

      :param sample_num: the number of samples.
      :type sample_num: int

      :returns: a list of samples.
      :rtype: sample_list (np.array)



   .. py:method:: _get_candidates_list()

      Get sample candidates list for _pop_sampling()

      :returns: a list of candidates id.
      :rtype: candidates_list (list)



   .. py:method:: get_used_ids()

      Returns:
      numpy.ndarray: Used item_ids is the same as positive item_ids.
      Index is user_id, and element is a set of item_ids.



   .. py:method:: sample_by_user_ids(user_ids, item_ids, num)

      Sampling by user_ids.

      :param user_ids: Input user_ids.
      :type user_ids: numpy.ndarray or list
      :param item_ids: Input item_ids.
      :type item_ids: numpy.ndarray or list
      :param num: Number of sampled item_ids for each user_id.
      :type num: int

      :returns: Sampled item_ids.
                item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
                is sampled for user_ids[0];
                item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
                item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
      :rtype: torch.tensor



   .. py:method:: set_phase(phase)

      Get the sampler of corresponding phase.

      :param phase: The phase of new sampler.
      :type phase: str

      :returns: the copy of this sampler, and :attr:`phase` is set the same as input phase.
      :rtype: Sampler



.. py:class:: SeqSampler(dataset, distribution='uniform', alpha=1.0)

   Bases: :py:obj:`AbstractSampler`


   :class:`SeqSampler` is used to sample negative item sequence.

   :param datasets: All the dataset for each phase.
   :type datasets: Dataset or list of Dataset
   :param distribution: Distribution of the negative items. Defaults to 'uniform'.
   :type distribution: str, optional


   .. py:attribute:: dataset


   .. py:attribute:: iid_field


   .. py:attribute:: user_num


   .. py:attribute:: item_num


   .. py:method:: _uni_sampling(sample_num)

      Sample [sample_num] items in the uniform distribution.

      :param sample_num: the number of samples.
      :type sample_num: int

      :returns: a list of samples.
      :rtype: sample_list (np.array)



   .. py:method:: get_used_ids()

      Returns:
      numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.



   .. py:method:: sample_neg_sequence(pos_sequence)

      For each moment, sampling one item from all the items except the one the user clicked on at that moment.

      :param pos_sequence: all users' item history sequence, with the shape of `(N, )`.
      :type pos_sequence: torch.Tensor

      :returns: all users' negative item history sequence.
      :rtype: torch.tensor



