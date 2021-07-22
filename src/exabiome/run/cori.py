from hdmf.utils import docval, getargs
from .job import AbstractJob

class SlurmJob(AbstractJob):

    directive = 'SBATCH'
    queue_flag = 'q'
    project_flag = 'A'
    time_flag = 't'
    output_flag = 'o'
    error_flag = 'e'
    jobname_flag = 'J'
    nodes_flag = 'n'
    submit_cmd = 'sbatch'
    job_var = 'SLURM_JOB_ID'
    job_fmt_var = 'j'
    job_id_re = 'Submitted batch job (\d+)'

    debug_queue = 'debug'

    @docval({'name': 'queue',   'type': str, 'doc': 'queue to submit to', 'default': 'regular'},
            {'name': 'project', 'type': str, 'doc': 'project to charge to', 'default': None},
            {'name': 'time',    'type': str, 'doc': 'request job time', 'default': '1:00:00'},
            {'name': 'nodes',   'type': int, 'doc': 'number of nodes to request', 'default': 1},
            {'name': 'gpus',    'type': int, 'doc': 'number of GPUs to request', 'default': 0},
            {'name': 'jobname', 'type': str, 'doc': 'name of the job', 'default': None},
            {'name': 'output',  'type': str, 'doc': 'standard output of job', 'default': None},
            {'name': 'error',   'type': str, 'doc': 'standard error of job', 'default': None})
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        queue, project, time, nodes, jobname = getargs('queue', 'project', 'time', 'nodes', 'jobname', kwargs)
        self.queue = queue
        self.project = project
        self.time = time
        self.nodes = self.gpus * nodes # nodes will actually be the flag for Total number of tasks
        self.jobname = jobname
        if self.jobname is not None:
            self.output = f'{self.jobname}.%J'
            self.error = f'{self.jobname}.%J'

        arch = 'gpu'
        if self.gpus == 0:
            arch = 'haswell'
        self.add_addl_jobflag('C', arch)
        #self.add_addl_jobflag('G', self.gpus)
        self.add_addl_jobflag('c', 10)
        self.add_addl_jobflag('-ntasks-per-node', self.gpus)
        self.add_addl_jobflag('-gpus-per-task', 1)

        n_gpus = self.gpus
        self.use_bb = False

    def set_use_bb(self, use_bb=True):
        self.use_bb = use_bb

    def write_run(self, f, command, command_options, options):
        print(f'srun -u {command}', file=f)
