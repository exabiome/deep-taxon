import argparse
from datetime import datetime
import os
import shutil
import sys
import ruamel.yaml as yaml
import time

from .utils import get_job
from ..utils import get_seed, check_argv


def run_train(argv=None):
    argv = check_argv(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input file to use')
    parser.add_argument('config', help='the config file to use')
    parser.add_argument('sh', nargs='?', help='a path to write the shell script to')

    parser.add_argument('-o', '--outdir',      help="the output directory", default='.')
    parser.add_argument('-m', '--message',     help="message to write to log file", default=None)
    parser.add_argument('-L', '--log',         help="the log file to store run information in", default='jobs.log')
    parser.add_argument('--submit',            help="submit job to queue", action='store_true', default=False)
    prof_grp = parser.add_mutually_exclusive_group()
    prof_grp.add_argument('--profile', action='store_true', default=False, help='profile with PyTorch Lightning profile')
    prof_grp.add_argument('--cuda_profile', action='store_true', default=False, help='profile with PyTorch CUDA profiling')
    parser.add_argument('-a', '--chain',       help="chain jobs in submission", type=int, default=1)

    rsc_grp = parser.add_argument_group('Resource Manager Arguments')
    rsc_grp.add_argument('-t', '--time',       help='the time to run the job for', default='01:00:00')
    rsc_grp.add_argument('-n', '--nodes',      help="the number of nodes to use", default=None, type=int)
    rsc_grp.add_argument('-g', '--gpus',       help="the number of GPUs to use", default=None, type=int)
    rsc_grp.add_argument('-N', '--jobname',    help="the name of the job", default=None)
    rsc_grp.add_argument('-q', '--queue',      help="the queue to submit to", default=None)
    rsc_grp.add_argument('-P', '--project',    help="the project/account to submit under", default=None)
    rsc_grp.add_argument('-S', '--scratch',    help="the job require scratch", default=False, action='store_true')

    system_grp = parser.add_argument_group('Compute system')
    grp = system_grp.add_mutually_exclusive_group()
    grp.add_argument('--cori',        help='make script for running on NERSC Cori',  action='store_true', default=False)
    grp.add_argument('--perlmutter',  help='make script for running on NERSC Perlmutter',  action='store_true', default=False)
    grp.add_argument('--summit',      help='make script for running on OLCF Summit', action='store_true', default=False)

    parser.add_argument('-B', '--global_bs', type=eval, help='the global batch size. Using this option will set batch_size in the config', default=None)
    parser.add_argument('-V', '--n_val_checks', type=int, help='the number of validation checks to do per epoch', default=1)
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
    parser.add_argument('-s', '--sanity', metavar='NBAT', nargs='?', const=True, default=False,
                        help='run NBAT batches for training and NBAT//4 batches for validation. By default, NBAT=4000')
    parser.add_argument('-T', '--timed_checkpoint', metavar='MIN', nargs='?', const=True, default=False,
                        help='run a checkpoing ever MIN seconds. By default MIN=10')
    parser.add_argument('--early_stop',         help="use PL early stopping", action='store_true', default=False)
    parser.add_argument('--swa', action='store_true', default=False, help='use stochastic weight averaging')
    parser.add_argument('--csv', action='store_true', default=False, help='log to a CSV file instead of WandB')
    parser.add_argument('--apex', action='store_true', default=False, help='use Apex fused optimizers')
    parser.add_argument('--shm', action='store_true', default=False, help='copy input to shared memory before training')
    parser.add_argument('-l', '--load',         help="load dataset into memory", action='store_true', default=False)
    parser.add_argument('-C', '--conda_env',    help=("the conda environment to use. use 'none' "
                                                      "if no environment loading is desired"), default=None)
    parser.add_argument('-W', '--wandb_id', type=str, help='the WandB ID. Use this to resume previous runs', default=hex(hash(time.time()))[2:10])

    args = parser.parse_args(argv)

    job = get_job(args)

    if not os.path.exists(args.config):
        print(f"ERROR - Config file {args.config} does not exist", file=sys.stderr)
        exit(1)

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)

    if args.global_bs is not None:
        conf['batch_size'] = args.global_bs // (args.nodes * args.gpus)

    if conf.get('optimizer', "").startswith('adam') and conf.get('lr_scheduler', '') == 'cyclic':
        print("ERROR - Cannot use cyclic LR scheduler with Adam/AdamW", file=sys.stderr)
        exit(1)

    if conf.get('seed', None) is None:
        conf["seed"] = get_seed()


    args.input = os.path.abspath(args.input)
    if not os.path.exists(args.input):
        print(f"ERROR - Input file {args.input} does not exist", file=sys.stderr)
        exit(1)

    options = ''
    if args.debug:
        job.set_debug(True)

    if args.timed_checkpoint:
        options += ' -T'
        if isinstance(args.timed_checkpoint, str):
            options += f' {args.timed_checkpoint}'

    if args.sanity:
        options += ' -s'
        if isinstance(args.sanity, str):
            options += f' {args.sanity}'

    if args.early_stop:
        options += f' --early_stop'

    if args.swa:
        options += f' --swa'

    if args.profile:
        options += f' --profile'
    elif args.cuda_profile:
        options += f' --cuda_profile'
        args.csv = True
        job.add_modules('cudatoolkit')

    if args.csv:
        options += f' --csv'
    else:
        if args.wandb_id is not None:
            options += f' -W {args.wandb_id}'

    if args.apex:
        options += f' --apex'

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

    job.set_env_var('OPTIONS', options)
    job.set_env_var('OUTDIR', f'{expdir}/train.$JOB')
    job.set_env_var('CONF', f'{expdir}/train.$JOB.yml')
    job.set_env_var('INPUT', args.input)
    job.set_env_var('LOG', '$OUTDIR.log')


    if args.cuda_profile:
        job.set_env_var('NCCL_DEBUG', 'TRACE', export=True)
        job.set_env_var('NCCL_DEBUG_SUBSYS', 'ALL', export=True)
        job.set_env_var('NCCL_GRAPH_DUMP_FILE', '$OUTDIR/topology.xml', export=True)
        job.set_env_var('NCCL_DEBUG_FILE', '$OUTDIR/nccl_trace_tag.%h.%p.txt', export=True)

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
    else:
        train_cmd += ' --slurm'
        if job.use_bb:
            job.set_env_var('BB_INPUT', '/dev/shm/`basename $INPUT`')
            input_var = 'BB_INPUT'

            job.add_command('echo "$INPUT to $BB_INPUT" >> $LOG')
            job.add_command('cp $INPUT $BB_INPUT') #, run=f'srun -n {args.nodes} -r 1 -a 1')
            job.add_command('ls /tmp') #, run='jsrun -n 1')
            job.add_command('ls $BB_INPUT') #, run='jsrun -n 1')
        elif args.shm:
            job.set_env_var('SHM_INPUT', '/dev/shm/`basename $INPUT`')
            input_var = 'SHM_INPUT'
            job.add_command(f"srun --ntasks {args.nodes} --ntasks-per-node 1 cp $INPUT $SHM_INPUT")


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
        srun = f'srun -n {job.nodes}'
        if args.cuda_profile:
            srun += ' nsys profile -t cuda,cudnn,nvtx,osrt --output=$OUTDIR/nsys_report.%h.%p --stats=true'
        job.add_command('$CMD >> $LOG 2>&1', run=srun)

    for i in range(len(argv)):
        if " " in argv[i]:
            argv[i] = f'"{argv[i]}"'

    if args.scratch:
        job.add_addl_jobflag('L', 'scratch')

    def submit(job, shell, message):
        job_id = job.submit_job(shell, conda_env=args.conda_env)
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
        return job_id, jobdir, message

    continue_from = ""
    if args.sh is not None:
        if args.submit:
            msg = args.message
            if msg is None:
                msg = input(f"Please provide a log message for this job (logging to {args.log}): ")
            for i in range(args.chain):
                with open(args.sh, 'w') as out:
                    job.write(out)
                jobid, jobdir, args.message = submit(job, args.sh, msg + continue_from)
                continue_from = f', continue from {jobid}'
                job_dep = jobid
                if args.summit:
                    job_dep = f'ended({job_dep})'
                job.add_addl_jobflag(job.wait_flag, job_dep)
                job.set_env_var('CKPT', os.path.join(jobdir, 'last.ckpt'))
                if '-i' in options:
                    i = options.find('-i')
                    options = options[:i+1] + 'c' + options[i+2:]
                if '-c' not in options:
                    job.set_env_var('OPTIONS', options + ' -c $CKPT')
                job.set_env_var('CMD', train_cmd)
                args.message = args.message[:args.message.find("resume")].strip().strip(',')
        else:
            with open(args.sh, 'w') as out:
                job.write(out)

    else:
        job.write(sys.stdout)
