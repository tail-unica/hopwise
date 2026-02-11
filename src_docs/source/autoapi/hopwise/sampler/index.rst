hopwise.sampler
===============

.. py:module:: hopwise.sampler


Submodules
----------

.. toctree::
   :maxdepth: 1

   /autoapi/hopwise/sampler/sampler/index


Classes
-------

.. autoapisummary::

   hopwise.sampler.KGSampler
   hopwise.sampler.RepeatableSampler
   hopwise.sampler.Sampler
   hopwise.sampler.SeqSampler


Package Contents
----------------

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



