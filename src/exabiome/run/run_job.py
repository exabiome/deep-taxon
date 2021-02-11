import sys
import os.path
import argparse
import shutil

from .summit import LSFJob
from .cori import SlurmJob

from ..utils import get_seed, check_argv


def check_summit(args):
    if args.gpus is None:
        args.gpus = 6
    if args.nodes is None:
        args.nodes = 2
    if args.outdir is None:
        args.outdir = os.path.realpath(os.path.expandvars("$PROJWORK/bif115/../scratch/$USER/deep-index"))
    if args.conda_env is None:
        args.conda_env = 'exabiome-wml'
        #args.conda_env = 'exabiome-wml-1.7.0-3'


def check_cori(args):
    if args.gpus is None:
        args.gpus = 4
    if args.nodes is None:
        args.nodes = 1
    if args.outdir is None:
        args.outdir = os.path.abspath(os.path.expandvars("$CSCRATCH/exabiome/deep-index"))


def run_train(argv=None):
    argv = check_argv(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input file to use')
    parser.add_argument('sh', nargs='?', help='the input file to use')

    parser.add_argument('--submit',            help="submit job to queue", action='store_true', default=False)
    parser.add_argument('-O', '--outdir',      help="the output directory", default=None)
    parser.add_argument('--profile',           help="use PTL profiling", action='store_true', default=False)

    rsc_grp = parser.add_argument_group('Resource Manager Arguments')
    rsc_grp.add_argument('-t', '--time',       help='the time to run the job for', default='01:00:00')
    rsc_grp.add_argument('-n', '--nodes',      help="the number of nodes to use", default=None, type=int)
    rsc_grp.add_argument('-g', '--gpus',       help="the number of GPUs to use", default=None, type=int)
    rsc_grp.add_argument('-N', '--jobname',    help="the name of the job", default=None)
    rsc_grp.add_argument('-q', '--queue',      help="the queue to submit to", default=None)
    rsc_grp.add_argument('-P', '--project',    help="the project/account to submit under", default=None)
    rsc_grp.add_argument('-a', '--arch',       help="the architecture to use, e.g., gpu or haswell (cori only)", default='gpu')

    system_grp = parser.add_argument_group('Compute system')
    grp = system_grp.add_mutually_exclusive_group()
    grp.add_argument('--cori',   help='make script for running on NERSC Cori',  action='store_true', default=False)
    grp.add_argument('--summit', help='make script for running on OLCF Summit', action='store_true', default=False)

    parser.add_argument('-r', '--rate',         help="the learning rate to use for training", default=0.001)
    parser.add_argument('-o', '--output_dims',  help="the number of dimensions to output", default=256)
    parser.add_argument('-A', '--accum',        help="the number of batches to accumulate", default=1)
    parser.add_argument('-b', '--batch_size',   help="the number of batches to accumulate", default=64)
    parser.add_argument('-W', '--window',       help="the size of chunks to use", default=4000)
    parser.add_argument('-S', '--step',         help="the chunking step size", default=4000)
    parser.add_argument('-s', '--seed',         help="the seed to use", default=None)
    parser.add_argument('-L', '--loss',         help="the loss function to use", default='M')
    parser.add_argument('-M', '--model',        help="the model name", default='roznet')
    parser.add_argument('-D', '--dataset',      help="the dataset name", default='default')
    parser.add_argument('-e', '--epochs',       help="the number of epochs to run for", default=10)
    parser.add_argument('-F', '--fwd_only',     help="use only fwd strand", action='store_true', default=False)
    parser.add_argument('-u', '--scheduler',    help="the learning rate scheduler to use", default=None)
    parser.add_argument('-c', '--checkpoint',   help="a checkpoint file to restart from", default=None)
    parser.add_argument('-E', '--experiment',   help="the experiment name to use", default=None)
    parser.add_argument('-d', '--debug',        help="run in debug mode", action='store_true', default=False)
    parser.add_argument('-l', '--load',         help="load dataset into memory", action='store_true', default=False)
    parser.add_argument('-C', '--conda_env',    help="the conda environment ot use", default=None)

    args = parser.parse_args(argv)

    if args.seed is None:
        args.seed = get_seed()

    options = dict()
    if args.summit:
        check_summit(args)
        job = LSFJob()
        job.set_conda_env(args.conda_env)
        job.add_modules('open-ce')
        if not args.load:
            job.set_use_bb(True)
    else:
        check_cori(args)
        job = SlurmJob(arch=args.arch)

    job.nodes = args.nodes
    job.time = args.time
    job.gpus = args.gpus
    job.jobname = args.jobname

    if args.queue is not None:
        job.queue = args.queue
        
    if args.project is not None:
        job.project = args.project

    args.input = os.path.abspath(args.input)

    options = ''
    if args.debug:
        job.set_debug(True)
        options = '-d'

    if args.profile:
        options += f' --profile'

    chunks = f'chunks_W{args.window}_S{args.step}'

    options = (f'{options} -{args.loss} -b {args.batch_size} -g {args.gpus} -n {args.nodes} -o {args.output_dims} '
               f'-W {args.window} -S {args.step} -r {args.rate} -A {args.accum} -e {args.epochs}')

    exp = f'n{args.nodes}_g{args.gpus}_A{args.accum}_b{args.batch_size}_r{args.rate}_o{args.output_dims}'

    if args.load:
        options += f' -l'

    if args.fwd_only:
        options += ' -F'
        chunks += '_fwd-only'

    if args.seed:
        options += f' -s {args.seed}'

    if args.scheduler:
        options += f' --lr_scheduler {args.scheduler}'
        exp += f'_{args.scheduler}'

    if args.checkpoint:
        options += f' -c {args.checkpoint}'


    if args.experiment:
        exp = args.experiment

    options += f' -E {exp}'

    expdir = f'{args.outdir}/train/datasets/{args.dataset}/{chunks}/{args.model}/{args.loss}/{exp}'
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    job.output = f'{expdir}/train.%{job.job_fmt_var}.lsf_log'
    job.error = job.output

    if args.nodes > 1:
        job.set_env_var('OMP_NUM_THREADS', 1)
        job.set_env_var('NCCL_DEBUG', 'INFO')

    job.set_env_var('OPTIONS', options)
    job.set_env_var('OUTDIR', f'{expdir}/train.$JOB')
    job.set_env_var('INPUT', args.input)
    job.set_env_var('LOG', '$OUTDIR.log')

    input_var = 'INPUT'

    train_cmd = 'deep-index train'
    if args.summit:
        train_cmd += ' --lsf'
        if job.use_bb:
            job.set_env_var('BB_INPUT', '/mnt/bb/$USER/`basename $INPUT`')
            input_var = 'BB_INPUT'
    elif args.cori:
        train_cmd += ' --slurm'

    train_cmd += f' $OPTIONS {args.model} ${input_var} $OUTDIR'

    if args.summit and job.use_bb:
        job.add_command('echo "$INPUT to $BB_INPUT"')
        job.add_command('cp $INPUT $BB_INPUT', run='jsrun -n 1')
        job.add_command('ls /mnt/bb/$USER', run='jsrun -n 1')
        job.add_command('ls $BB_INPUT', run='jsrun -n 1')

    job.set_env_var('CMD', train_cmd)

    cp_run = None
    if args.summit:
        cp_run = 'jsrun -n 1'
    job.add_command('cp $0 $OUTDIR.sh', run=cp_run)


    job.add_command('mkdir -p $OUTDIR')

    if args.summit:
        # when using regular DDP, jsrun should be called with one resource per node (-r) and
        # one rank per GPU (-a) to work with PyTorch Lightning
        jsrun = f'jsrun -g {args.gpus} -n {args.nodes} -a {args.gpus} -r 1 -c 42'
        job.add_command('$CMD > $LOG 2>&1', run=jsrun)
    else:
        job.add_command('$CMD > $LOG 2>&1', run='srun')

    if args.sh is not None:
        with open(args.sh, 'w') as out:
            job.write(out)
        if args.submit:
            job_id = job.submit_job(args.sh)
            dest = f'{expdir}/train.{job_id}.sh'
            print(f'copying submission script to {os.path.relpath(dest)}')
            logpath = os.path.relpath(f'{expdir}/train.{job_id}.log')
            print(f'logging to {logpath}')
            shutil.copyfile(args.sh, dest)

    else:
        job.write(sys.stdout)
