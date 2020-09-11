from .job import AbstractJob

class LSFJob(AbstractJob):

    directive = 'BSUB'
    queue_flag = 'q'
    project_flag = 'P'
    time_flag = 'W'
    output_flag = 'o'
    error_flag = 'e'
    jobname_flag = 'J'
    nodes_flag = 'nnodes'
    submit_cmd = 'bsub'
    job_var = 'LSB_JOBID'
    job_fmt_var = 'J'
    job_id_re = 'Job <(\d+)>'

    debug_queue = 'debug'

    def __init__(self, queue='batch', project='BIF115', time='1:00:00', nodes=2, jobname=None, output=None, error=None):
        super().__init__()
        self.queue = queue
        self.project = project
        self.time = time
        self.nodes = nodes
        self.jobname = jobname
        if self.jobname is not None:
            self.output = f'{self.jobname}.%J'
            self.error = f'{self.jobname}.%J'
        self.use_bb = False

    def set_use_bb(self, use_bb=True):
        self.use_bb = use_bb

    def write_additional(self, f, options=None):
        if options is None:
            options = dict()
        alloc_flags = ['gpumps']
        if self.use_bb:
            alloc_flags.append('NVME')
        if len(alloc_flags):
            self.write_line(f, 'alloc_flags', '"%s"' % ' '.join(alloc_flags))

    def write_run(self, f, command, command_options, options):
        print(f'ddlrun {command}', file=f)
