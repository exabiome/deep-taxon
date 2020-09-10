class AbstractJob:

    def __init__(self, queue_flag, project_flag, time_flag, output_flag, error_flag,
                 gpu_flag):

    @property
    @abstractmethod
    def directive(self):
        pass

    @property
    @abstractmethod
    def time(self):
        pass

    @property
    @abstractmethod
    def project(self):
        pass

    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def time(self):
        pass

    def _write_line(self, flag, value, file):
        print(f'{self.directive} -{flag} {value}', file=file)

    def write_header(self, f):
        print(f'#!/bin/bash', file=f)
        self._write_line(self.queue_flag, self.queue)
        print(f'#BSUB -q debug', file=f)
        print(f'#BSUB -P BIF115', file=f)
        print(f'#BSUB -W 1:00', file=f)
        print(f'#BSUB -nnodes 2', file=f)
        print(f'#BSUB -alloc_flags "gpumps NVME"', file=f)
        print(f'#BSUB -J DeepIndexTrain', file=f)
        print(f'#BSUB -o DeepIndexTrain.%J', file=f)
        print(f'#BSUB -e DeepIndexTrain.%J', file=f)
