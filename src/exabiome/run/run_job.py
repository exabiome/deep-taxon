import argparse
from datetime import datetime
import os
import shutil
import sys
import ruamel.yaml as yaml

from .summit import LSFJob
from .cori import SlurmJob

from ..utils import get_seed, check_argv


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
        #args.conda_env = 'exabiome-wml-1.7.0-3'


def check_cori(args):
    if args.gpus is None:
        args.gpus = 4
    if args.nodes is None:
        args.nodes = 1
    if args.outdir is None:
        args.outdir = os.path.abspath(os.path.expandvars("$CSCRATCH/exabiome/deep-taxon"))
    if args.queue is None:
        args.queue = 'regular'


#def check_args(args):
#    if args.loss == 'M':
#        if args.output_dims is None:
#            args.output_dims = 256
#

def get_jobargs(args):
    return {k: getattr(args, k) for k in ('queue', 'nodes', 'gpus', 'jobname', 'project', 'time')}


def run_train(argv=None):
    argv = check_argv(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input file to use')
    parser.add_argument('config', help='the config file to use')
    parser.add_argument('sh', nargs='?', help='a path to write the shell script to')

    parser.add_argument('-o', '--outdir',      help="the output directory", default=None)
    parser.add_argument('-m', '--message',     help="message to write to log file", default=None)
    parser.add_argument('-L', '--log',         help="the log file to store run information in", default='jobs.log')
    parser.add_argument('--submit',            help="submit job to queue", action='store_true', default=False)
    parser.add_argument('--profile',           help="use PTL profiling", action='store_true', default=False)
    parser.add_argument('-a', '--chain',       help="chain jobs in submission", type=int, default=1)

    rsc_grp = parser.add_argument_group('Resource Manager Arguments')
    rsc_grp.add_argument('-T', '--time',       help='the time to run the job for', default='01:00:00')
    rsc_grp.add_argument('-n', '--nodes',      help="the number of nodes to use", default=None, type=int)
    rsc_grp.add_argument('-g', '--gpus',       help="the number of GPUs to use", default=None, type=int)
    rsc_grp.add_argument('-N', '--jobname',    help="the name of the job", default=None)
    rsc_grp.add_argument('-q', '--queue',      help="the queue to submit to", default=None)
    rsc_grp.add_argument('-P', '--project',    help="the project/account to submit under", default=None)

    system_grp = parser.add_argument_group('Compute system')
    grp = system_grp.add_mutually_exclusive_group()
    grp.add_argument('--cori',   help='make script for running on NERSC Cori',  action='store_true', default=False)
    grp.add_argument('--summit', help='make script for running on OLCF Summit', action='store_true', default=False)

    parser.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=1)
    parser.add_argument('-y', '--pin_memory', action='store_true', default=False, help='pin memory when loading data')
    parser.add_argument('-f', '--shuffle', action='store_true', default=False, help='shuffle batches when training')
    parser.add_argument('-D', '--dataset',      help="the dataset name", default='default')
    parser.add_argument('-e', '--epochs',       help="the number of epochs to run for", default=10)
    parser.add_argument('-c', '--checkpoint',   help="a checkpoint file to restart from", default=None)
    parser.add_argument('-i', '--init',         help="a checkpoint file to initialize models from", default=None)
    parser.add_argument('-F', '--features',     help="a checkpoint file for features", default=None)
    parser.add_argument('-E', '--experiment',   help="the experiment name to use", default=None)
    parser.add_argument('-d', '--debug',        help="submit to debug queue", action='store_true', default=False)
    parser.add_argument('--sanity',             help="run a small number of batches", action='store_true', default=False)
    parser.add_argument('--early_stop',         help="use PL early stopping", action='store_true', default=False)
    parser.add_argument('--swa', action='store_true', default=False, help='use stochastic weight averaging')
    parser.add_argument('-l', '--load',         help="load dataset into memory", action='store_true', default=False)
    parser.add_argument('-C', '--conda_env',    help=("the conda environment to use. use 'none' "
                                                      "if no environment loading is desired"), default=None)


    #parser.add_argument('-M', '--model',        help="the model name", default='roznet')
    #parser.add_argument('-s', '--seed',         help="the seed to use", default=None)
    #parser.add_argument('-w', '--weighted',     help='weight classes in classification',
    #                     nargs='?', const=True, default=False, choices=['ins', 'isns', 'ens'])
    #parser.add_argument('-o', '--output_dims',  help="the number of dimensions to output", default=None)
    #parser.add_argument('-A', '--accum',        help="the number of batches to accumulate", default=1)
    #parser.add_argument('-b', '--batch_size',   help="the number of batches to accumulate", default=64)
    #parser.add_argument('-W', '--window',       help="the size of chunks to use", default=4000)
    #parser.add_argument('-S', '--step',         help="the chunking step size", default=4000)
    #parser.add_argument('-s', '--seed',         help="the seed to use", default=None)
    #parser.add_argument('-L', '--loss',         help="the loss function to use", default='M')
    #parser.add_argument('-t', '--tgt_tax_lvl',  help='the taxonomic level to use', default=None)
    #parser.add_argument('-O', '--optimizer', type=str, choices=['adam', 'lamb'], help='the optimizer to use', default='adam')
    #parser.add_argument('--fwd_only',     help="use only fwd strand", action='store_true', default=False)
    #parser.add_argument('-u', '--scheduler',    help="the learning rate scheduler to use", default=None)
    #parser.add_argument('-r', '--rate',         help="the learning rate to use for training", default=0.001)

    args = parser.parse_args(argv)

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)


    if conf.get('optimizer', "").startswith('adam') and conf.get('lr_scheduler', '') == 'cyclic':
        print("Cannot use cyclic LR scheduler with Adam/AdamW", file=sys.stderr)
        exit(1)

    if conf.get('seed', None) is None:
        conf["seed"] = get_seed()

    if args.summit:
        check_summit(args)
        jobargs = get_jobargs(args)
        job = LSFJob(**jobargs)
        job.add_modules('open-ce')
        if not args.load:
            job.set_use_bb(True)
    else:
        check_cori(args)
        jobargs = get_jobargs(args)
        job = SlurmJob(**jobargs)

    if args.conda_env is None:
        args.conda_env = os.environ.get('CONDA_DEFAULT_ENV', None)

    if args.conda_env != 'none':
        job.set_conda_env(args.conda_env)

    # job.nodes = args.nodes
    # job.time = args.time
    # job.gpus = args.gpus
    # job.jobname = args.jobname

    # if args.queue is not None:
    #     job.queue = args.queue

    # if args.project is not None:
    #     job.project = args.project

    args.input = os.path.abspath(args.input)

    options = ''
    if args.debug:
        job.set_debug(True)

    if args.sanity:
        options = '--sanity'

    if args.early_stop:
        options += f' --early_stop'

    if args.swa:
        options += f' --swa'

    if args.profile:
        options += f' --profile'

    chunks = f'chunks_W{conf["window"]}_S{conf["step"]}'

    L = None
    if conf.get('manifold', False):
        L = 'M'
    elif conf.get('classify', False):
        L = 'C'
    else:
        raise ValueError("did not find 'classify' or 'manifold' in config file")

    options = (f'{options} -g {args.gpus} -n {args.nodes} -e {args.epochs} '
               f'-k {args.num_workers}')

    if "n_outputs" in conf:
        exp = f'n{args.nodes}_g{args.gpus}_A{conf["accumulate"]}_b{conf["batch_size"]}_r{conf["lr"]}_o{conf["n_outputs"]}_O{conf["optimizer"]}'
    else:
        exp = f'n{args.nodes}_g{args.gpus}_A{conf["accumulate"]}_b{conf["batch_size"]}_r{conf["lr"]}_{conf["tgt_tax_lvl"]}_O{conf["optimizer"]}'

    if args.pin_memory:
        options += ' -y'

    if args.shuffle:
        options += ' -f'

    if args.load:
        options += f' -l'

    if conf.get('lr_scheduler', None) is not None:
        exp += f'_{conf["lr_scheduler"]}'

    if args.checkpoint:
        if not args.checkpoint.endswith('.ckpt'):
            args.checkpoint = os.path.join(args.checkpoint, 'last.ckpt')
        if not os.path.exists(args.checkpoint) or os.path.isdir(args.checkpoint):
            # assume we are supposed ot wait for the job to finish
            # to get the checkpoint from
            if os.path.isdir(args.checkpoint):
                jobdir = args.checkpoint
                args.checkpoint = os.path.join(args.checkpoint, 'last.ckpt')
            else:
                jobdir = os.path.dirname(args.checkpoint)
            job_dep = jobdir[jobdir.rfind('.')+1:].strip("/")
            if args.summit:
                job_dep = f'ended({job_dep})'
            job.add_addl_jobflag(job.wait_flag, job_dep)
        job.set_env_var('CKPT', args.checkpoint)
        options += f' -c $CKPT'
    elif args.init:
        job.set_env_var('CKPT', args.init)
        options += f' -i $CKPT'
    elif args.chain > 1:
        job.set_env_var('CKPT', '')


    if args.features:
        job.set_env_var('FEATS_CKPT', args.features)
        options += f' -F $FEATS_CKPT'

    if args.experiment:
        exp = args.experiment

    options += f' -E {exp}'



    expdir = f'{args.outdir}/train/datasets/{args.dataset}/{chunks}/{conf["model"]}/{L}/{exp}'
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    job.output = f'{expdir}/train.%{job.job_fmt_var}.log'
    job.error = job.output

    job.set_env_var('OMP_NUM_THREADS', 1)
    job.set_env_var('NCCL_DEBUG', 'INFO')

    job.set_env_var('OPTIONS', options)
    job.set_env_var('OUTDIR', f'{expdir}/train.$JOB')
    job.set_env_var('CONF', f'{expdir}/train.$JOB.yml')
    job.set_env_var('INPUT', args.input)
    job.set_env_var('LOG', '$OUTDIR.log')

    input_var = 'INPUT'

    train_cmd = 'deep-taxon train'
    if args.summit:
        train_cmd += ' --lsf'
        if job.use_bb:
            job.set_env_var('BB_INPUT', '/mnt/bb/$USER/`basename $INPUT`')
            input_var = 'BB_INPUT'

            job.add_command('echo "$INPUT to $BB_INPUT" >> $LOG')
            job.add_command('cp $INPUT $BB_INPUT', run=f'jsrun -n {args.nodes} -r 1 -a 1')
            job.add_command('ls /mnt/bb/$USER', run='jsrun -n 1')
            job.add_command('ls $BB_INPUT', run='jsrun -n 1')
    elif args.cori:
        train_cmd += ' --slurm'
        if job.use_bb:
            job.set_env_var('BB_INPUT', '/tmp/`basename $INPUT`')
            input_var = 'BB_INPUT'

            job.add_command('echo "$INPUT to $BB_INPUT" >> $LOG')
            job.add_command('cp $INPUT $BB_INPUT') #, run=f'srun -n {args.nodes} -r 1 -a 1')
            job.add_command('ls /tmp') #, run='jsrun -n 1')
            job.add_command('ls $BB_INPUT') #, run='jsrun -n 1')

    train_cmd += f' $OPTIONS $CONF ${input_var} $OUTDIR'

    job.set_env_var('CMD', train_cmd)

    cp_run = None
    if args.summit:
        cp_run = 'jsrun -n 1'
    job.add_command('mkdir -p $OUTDIR')
    job.add_command('cp $0 $OUTDIR.sh', run=cp_run)

    if args.summit:
        # when using regular DDP, jsrun should be called with one resource per node (-r) and
        # one rank per GPU (-a) to work with PyTorch Lightning
        n_cores = 42
        cores_per_task = n_cores//args.gpus
        jsrun = f'jsrun -g {args.gpus} -n {args.nodes} -a {args.gpus} -r 1 -c {n_cores}'
        job.add_command('$CMD >> $LOG 2>&1', run=jsrun)
    else:
        job.add_command('$CMD >> $LOG 2>&1', run='srun')


    def submit(job, shell, message):
        job_id = job.submit_job(shell)
        if job_id is None:
            print("unable to submit job")
        else:
            jobdir = f'{expdir}/train.{job_id}'
            print(f'running job out of {jobdir}')
            cfg_path = f'{jobdir}.yml'
            print(f'writing config file to {cfg_path}')
            with open(cfg_path, 'w') as f:
                yaml.main.safe_dump(conf, f, default_flow_style=False)
            dest = f'{jobdir}.sh'
            print(f'copying submission script to {os.path.relpath(dest)}')
            logpath = os.path.relpath(f'{jobdir}.log')
            print(f'logging to {logpath}')
            shutil.copyfile(args.sh, dest)
            with open(args.log, 'a') as logout:
                if message is None:
                    message = input("please provide a message about this run:\n")
                print(f'- {message}', file=logout)
                print(f'  - date:          %s' % datetime.now().strftime("%c"), file=logout)
                print(f'  - cmd:           {" ".join(argv)}', file=logout)
                print(f'  - job directory: {jobdir}', file=logout)
                print(f'  - log file:      {logpath}', file=logout)
                print(f'  - config file:   {jobdir}.yml', file=logout)
                print(f'  - batch script:  {jobdir}.sh', file=logout)
                print('-------------------------------------------------------------------------------', file=logout)
        return job_id, jobdir

    if args.sh is not None:
        if args.submit:
            msg = args.message
            for i in range(args.chain):
                with open(args.sh, 'w') as out:
                    job.write(out)
                jobid, jobdir = submit(job, args.sh, msg)
                msg = f'{args.message}, continue from {jobid}'
                job_dep = jobid
                if args.summit:
                    job_dep = f'ended({job_dep})'
                job.add_addl_jobflag(job.wait_flag, job_dep)
                job.set_env_var('CKPT', os.path.join(jobdir, 'last.ckpt'))
                job.set_env_var('OPTIONS', options + ' -c $CKPT')
                job.set_env_var('CMD', train_cmd)
                args.message = args.message[:args.message.find("resume")].strip().strip(',')
        else:
            with open(args.sh, 'w') as out:
                job.write(out)

    else:
        job.write(sys.stdout)
