class CostModel:
    def __init__(self):
        self.spillover_portion = 0.2
        self.offload_portion = 0.38  #A10 vicuna 0.45  3090 0.4
        self.auxilary_portion = self.spillover_portion + self.offload_portion
        self.primary_portion = 1 - self.auxilary_portion

        self.PCIe_bandwidth = 16 #16GB
        self.primary_peakflops = 0
        self.auxilary_peakflops = 0
        self.primary_bandwidth = 0
        self.auxilary_bandwidth  = 0
        self.compute_flops = 0

    def compute_offload_portion(self):
        pass


    def get_auxilary_queue_length(self):
        return (self.offload_portion/self.primary_portion)
        #vincuna=1.2, baichuang=1.5, aquiq=1.5