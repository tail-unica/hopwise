hopwise.model.init
==================

.. py:module:: hopwise.model.init

.. autoapi-nested-parse::

   hopwise.model.init
   ########################



Functions
---------

.. autoapisummary::

   hopwise.model.init.xavier_normal_initialization
   hopwise.model.init.xavier_uniform_initialization


Module Contents
---------------

.. py:function:: xavier_normal_initialization(module)

   Using `xavier_normal_`_ in PyTorch to initialize the parameters in
   nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
   using constant 0 to initialize.

   .. _`xavier_normal_`:
       https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

   .. rubric:: Examples

   >>> self.apply(xavier_normal_initialization)


.. py:function:: xavier_uniform_initialization(module)

   Using `xavier_uniform_`_ in PyTorch to initialize the parameters in
   nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
   using constant 0 to initialize.

   .. _`xavier_uniform_`:
       https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

   .. rubric:: Examples

   >>> self.apply(xavier_uniform_initialization)


