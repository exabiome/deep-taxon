import os

from .summit import LSFJob
from .nersc import SlurmJob


def check_summit(args):
    if args.gpus is None:
        args.gpus = 6
    if args.nodes is None:
        args.nodes = 2
    if args.outdir is None:
        args.outdir = os.path.realpath(os.path.expandvars("$PROJWORK/bif115/../scratch/$USER/deep-taxon"))
    if args.conda_env is None:
        args.conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)
        if args.conda_env is None:
            raise RuntimeError("Cannot determine the conda environment to use")


def check_nersc(args):
    if args.gpus is None:
        args.gpus = 4
    if args.nodes is None:
        args.nodes = 1
    if args.queue is None:
        args.queue = 'regular'


def get_jobargs(args):
    return {k: getattr(args, k) for k in ('queue', 'nodes', 'gpus', 'jobname', 'project', 'time')}


def get_job(args):
    """
    Figure out which Job type to use and set up necessary job environment configurations.
    """
    if args.summit:
        check_summit(args)
        jobargs = get_jobargs(args)
        job = LSFJob(**jobargs)
        job.add_modules('open-ce')
        if not args.load:
            job.set_use_bb(True)
        if args.conda_env is None:
            args.conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)

        if args.conda_env != 'none':
            job.set_conda_env(args.conda_env)
        # set to none
        args.conda_env = None
    else:
        check_nersc(args)
        jobargs = get_jobargs(args)
        job = SlurmJob(**jobargs)

    job.add_command('echo "=== deep-taxon package ===" >> $LOG')
    job.add_command("pip show deep-taxon >> $LOG")
    job.add_command('echo "========================" >> $LOG')

    return job
