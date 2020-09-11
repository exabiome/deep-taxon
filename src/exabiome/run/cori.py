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

    debug_queue = 'debug'

    def __init__(self, queue='batch', project='m2865', time='1:00:00', nodes=1, jobname=None, output=None, error=None):
        super().__init__()
        self.queue = queue
        self.project = project
        self.time = time
        self.nodes = nodes
        self.jobname = jobname
        if self.jobname is not None:
            self.output = f'{self.jobname}.%J'
            self.error = f'{self.jobname}.%J'

        self.add_addl_jobflag('C', 'gpu')

    def write_run(self, f, command, command_options, options):
        print(f'srun -u {command}', file=f)
