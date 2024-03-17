class AsySubmission:
    def __init__(self):
        self.pending_length = 4

    def get_pending_length(self):
        return self.pending_length

class CostModel:
    def __init__(self):
        self.spillover_portion = 0.2
        self.offload_portion = 0.5
        self.auxilary_portion = self.spillover_portion + self.offload_portion
        self.primary_portion = 1 - self.auxilary_portion

    def compute_offload_proportion(self):
        pass


    def get_auxilary_queue_length(self):
        return (self.offload_portion/self.primary_portion)