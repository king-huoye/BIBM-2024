import torch
# 内存监控

class MemMonica:
    def __init__(self):
        self.max_allocated_memory = 0
        self.max_reserved_memory = 0
        self.total_memory = torch.cuda.get_device_properties("cuda").total_memory  # 总共的
        self.cumulate_print = 0
        self.check_time = 0

    def check(self, status=0, **kwargs):
        self.check_time += 1
        allocated_memory = torch.cuda.memory_allocated("cuda")  # 分配的
        reserved_memory = torch.cuda.memory_reserved("cuda")  # 占用的
        to_print = False
        if allocated_memory > self.max_allocated_memory:
            self.max_allocated_memory = allocated_memory
            to_print = True
        if reserved_memory > self.max_reserved_memory:
            self.max_reserved_memory = reserved_memory
            to_print = True
        if to_print:
            self.cumulate_print += 1
        if to_print or self.check_time % 3000 == 0:
        # if to_print:
            print(status, self.check_time,
                  f"{allocated_memory / (1024 ** 2):.2f} MB.",  # Allocated
                  f"{reserved_memory / (1024 ** 2):.2f} MB",  # Reserved
                  f"{self.total_memory / (1024 ** 2):.2f} MB",  # Total
                  f"{(self.total_memory - reserved_memory) / (1024 ** 2):.2f} MB")  # Free


# monica = MemMonica()
