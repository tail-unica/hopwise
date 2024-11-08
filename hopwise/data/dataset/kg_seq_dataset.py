# @Time   : 2020/9/23
# @Author : Xingyu Pan
# @Email  : panxingyu@ruc.edu.cn

"""hopwise.data.kg_seq_dataset
#############################
"""

from hopwise.data.dataset import KnowledgeBasedDataset, SequentialDataset


class KGSeqDataset(SequentialDataset, KnowledgeBasedDataset):
    """Containing both processing of Sequential Models and Knowledge-based Models.

    Inherit from :class:`~hopwise.data.dataset.sequential_dataset.SequentialDataset` and
    :class:`~hopwise.data.dataset.kg_dataset.KnowledgeBasedDataset`.
    """

    def __init__(self, config):
        super().__init__(config)
