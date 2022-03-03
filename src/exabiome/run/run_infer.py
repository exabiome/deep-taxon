import argparse
from datetime import datetime
import os
import shutil
import sys
import ruamel.yaml as yaml

from .utils import get_job
from ..utils import get_seed, check_argv


def run_inference(argv=None):
    argv = check_argv(argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='the input file to use')
    parser.add_argument('checkpoint', help='the model checkpoint to use')
    parser.add_argument('sh', nargs='?', help='a path to write the shell script to')

    parser.add_argument('-F', '--config',      help='the config file to use', default=None)
    parser.add_argument('-m', '--message',     help="message to write to log file", default=None)
    parser.add_argument('-L', '--log',         help="the log file to store run information in", default='jobs.log')
    parser.add_argument('--submit',            help="submit job to queue", action='store_true', default=False)
    #parser.add_argument('-a', '--chain',       help="chain jobs in submission", type=int, default=1)

    rsc_grp = parser.add_argument_group('Resource Manager Arguments')
    rsc_grp.add_argument('-T', '--time',       help='the time to run the job for', default='01:00:00')
    rsc_grp.add_argument('-n', '--nodes',      help="the number of nodes to use", default=None, type=int)
    rsc_grp.add_argument('-g', '--gpus',       help="the number of GPUs to use", default=None, type=int)
    rsc_grp.add_argument('-N', '--jobname',    help="the name of the job", default=None)
    rsc_grp.add_argument('-q', '--queue',      help="the queue to submit to", default=None)
    rsc_grp.add_argument('-P', '--project',    help="the project/account to submit under", default=None)

    system_grp = parser.add_argument_group('Compute system')
    grp = system_grp.add_mutually_exclusive_group()
    grp.add_argument('--cori',        help='make script for running on NERSC Cori',  action='store_true', default=False)
    grp.add_argument('--perlmutter',  help='make script for running on NERSC Perlmutter',  action='store_true', default=False)
    grp.add_argument('--summit',      help='make script for running on OLCF Summit', action='store_true', default=False)

    parser.add_argument('--nonrep', action='store_true', default=False, help='the dataset is nonrepresentative species')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('-k', '--num_workers', type=int, help='the number of workers to load data with', default=1)
    parser.add_argument('-M', '--in_memory', default=False, action='store_true', help='collect all batches in memory before writing to disk')
    parser.add_argument('-B', '--n_batches', type=int, default=100, help='the number of batches to accumulate between each write to disk')
    parser.add_argument('-s', '--start', type=int, help='sample index to start at', default=None)
    parser.add_argument('-S', '--n_seqs', type=int, default=500, help='the number of sequences to aggregate chunks for between each write to disk')
    parser.add_argument('-p', '--maxprob', metavar='TOPN', nargs='?', const=1, default=0, type=int,
                        help='store the top TOPN probablities of each output. By default, TOPN=1')
    parser.add_argument('-c', '--save_chunks', action='store_true', help='save network outputs for all chunks', default=False)

    parser.add_argument('-d', '--debug',        help="submit to debug queue", action='store_true', default=False)

    parser.add_argument('-C', '--conda_env',    help=("the conda environment to use. use 'none' "
                                                      "if no environment loading is desired"), default=None)

    args = parser.parse_args(argv)


    job = get_job(args)

    outdir = os.path.splitext(args.checkpoint)[0] + ('.nonrep' if args.nonrep else '.rep')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if args.config is None:
        args.config = os.path.dirname(args.checkpoint) + ".yml"

    options = ''

    options = f'{options} -b {args.batch_size} -B {args.n_batches} -k {args.num_workers} -S {args.n_seqs} '

    if args.maxprob > 0:
        options += f'-p {args.maxprob} '

    if args.start != None:
        options += f'-s {args.start} '

    if args.save_chunks:
        options += f'-c '

    job.output = f'{outdir}/infer.%{job.job_fmt_var}.log'
    job.error = job.output

    job.set_env_var('OMP_NUM_THREADS', 1)

    job.set_env_var('OUTDIR', f'{outdir}/infer.$JOB')
    job.set_env_var('CONF', args.config)
    job.set_env_var('INPUT', args.input)
    job.set_env_var('CKPT', args.checkpoint)
    job.set_env_var('LOG', '$OUTDIR.log')
    job.set_env_var('OUTPUT', '$OUTDIR/outputs.h5')
    options += '-o $OUTPUT'
    job.set_env_var('OPTIONS', options)

    input_var = 'INPUT'

    train_cmd = 'deep-taxon infer'
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
            job.set_env_var('BB_INPUT', '/tmp/`basename $INPUT`')
            input_var = 'BB_INPUT'

            job.add_command('echo "$INPUT to $BB_INPUT" >> $LOG')
            job.add_command('cp $INPUT $BB_INPUT') #, run=f'srun -n {args.nodes} -r 1 -a 1')
            job.add_command('ls /tmp') #, run='jsrun -n 1')
            job.add_command('ls $BB_INPUT') #, run='jsrun -n 1')


    train_cmd += f' $OPTIONS $CONF ${input_var} $CKPT'

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
        srun = 'srun'
        job.add_command('$CMD >> $LOG 2>&1', run=srun)

    job.add_command('echo "==== Computing taxonomic accuracy ====" >> $LOG')
    job.add_command('deep-taxon tax-acc $OUTPUT $INPUT $OUTDIR/tax_acc.csv >> $LOG 2>&1')

    def submit(job, shell, message):
        job_id = job.submit_job(shell, conda_env=args.conda_env)
        if job_id is None:
            print("unable to submit job")
        else:
            jobdir = f'{outdir}/infer.{job_id}'
            print(f'running job out of {jobdir}')
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

    args.chain = 1
    if args.sh is not None:
        if args.submit:
            msg = args.message
            for i in range(args.chain):
                with open(args.sh, 'w') as out:
                    job.write(out)
                jobid, jobdir, args.message = submit(job, args.sh, msg)
                msg = f'{args.message}, continue from {jobid}'
                job_dep = jobid
                if args.summit:
                    job_dep = f'ended({job_dep})'
                job.add_addl_jobflag(job.wait_flag, job_dep)
                job.set_env_var('CKPT', os.path.join(jobdir, 'last.ckpt'))
                if '-c' not in options:
                    job.set_env_var('OPTIONS', options + ' -c $CKPT')
                job.set_env_var('CMD', train_cmd)
                args.message = args.message[:args.message.find("resume")].strip().strip(',')
        else:
            with open(args.sh, 'w') as out:
                job.write(out)

    else:
        job.write(sys.stdout)
