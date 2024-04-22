from typing import List
from vllm.sequence import SequenceGroup


class Policy:
    def get_priority(self, now: float, seq_group: SequenceGroup,) -> float:    
        raise NotImplementedError

    def sort_by_priority(self, now: float, seq_groups: List[SequenceGroup], ) -> List[SequenceGroup]:
        return sorted(seq_groups, key=lambda seq_group: self.get_priority(now, seq_group), reverse=True,)

    def sort_by_generation_length(self, seq_groups: List[SequenceGroup],)-> List[SequenceGroup]:
         return sorted(seq_groups, key=lambda seq_group: self.get_priority(seq_group), reverse=True,)


class FCFS(Policy):
    def get_priority(self, now: float, seq_group: SequenceGroup,) -> float:
        return now - seq_group.arrival_time

class Prediction(Policy):
    def get_priority(self, seq_group: SequenceGroup)-> float:
        all_seqs = seq_group.get_seqs()
        return (seq_group.sampling_params.max_tokens - all_seqs[0].get_output_len()) / float(all_seqs[0].get_output_len()+all_seqs[0].get_prompt_len())


class PolicyFactory:
    _POLICY_REGISTRY = {'fcfs': FCFS, 'prediction': Prediction}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
