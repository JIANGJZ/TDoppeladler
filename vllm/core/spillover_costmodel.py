class CostModel:
    def __init__(self, block_manager):
        self.spillover_portion = 0.2
        self.offload_portion = 0.28  #A10 vicuna 0.45  3090 0.4
        self.auxilary_portion = self.spillover_portion + self.offload_portion
        self.primary_portion = 1 - self.auxilary_portion

        self.PCIe_bandwidth = 16 #16GB
        self.primary_peakflops = 0
        self.auxilary_peakflops = 0
        self.primary_bandwidth = 0
        self.auxilary_bandwidth  = 0
        self.compute_flops = 0

        self.block_manager = block_manager
        self.offload_seq_per_time = 2


    def compute_offload_portion(self):
        pass


    def get_auxilary_queue_ratio(self):
        aux_memory_usage = self.block_manager.get_aux_memory_usage()
        ratio = ((aux_memory_usage * self.offload_portion) + self.spillover_portion)/self.primary_portion
        # print ("queue ratio = {}".format(ratio))
        return ratio
        #vincuna=1.2, baichuang=1.5, aquiq=1.5