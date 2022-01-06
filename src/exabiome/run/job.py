import re
import subprocess
from abc import abstractmethod, ABCMeta
from collections import OrderedDict

class AbstractJob(metaclass=ABCMeta):

    @property
    @abstractmethod
    def directive(self):
        pass

    @property
    @abstractmethod
    def queue_flag(self):
        pass

    @property
    @abstractmethod
    def wait_flag(self):
        pass

    @property
    @abstractmethod
    def project_flag(self):
        pass

    @property
    @abstractmethod
    def time_flag(self):
        pass

    @property
    @abstractmethod
    def output_flag(self):
        pass

    @property
    @abstractmethod
    def error_flag(self):
        pass

    @property
    @abstractmethod
    def jobname_flag(self):
        pass

    @property
    @abstractmethod
    def nodes_flag(self):
        pass

    @property
    @abstractmethod
    def submit_cmd(self):
        pass

    @property
    @abstractmethod
    def job_var(self):
        pass

    @property
    @abstractmethod
    def job_fmt_var(self):
        pass

    @property
    @abstractmethod
    def job_id_re(self):
        pass

    debug_queue = 'debug'


    def __init__(self, **kwargs):
        self.queue = kwargs.get('queue')
        self.project = kwargs.get('project')
        self.time = kwargs.get('time')
        self.output = kwargs.get('output')
        self.error = kwargs.get('error')
        self.gpus = kwargs.get('gpus')
        self.jobname = kwargs.get('jobname')
        self.nodes = kwargs.get('nodes')
        self.wait = kwargs.get('wait')
        self.conda_env = None
        self.modules = list()
        self.debug = False
        self.env_vars = OrderedDict()
        self.addl_job_flags = OrderedDict()
        self.set_env_var('JOB', f'${self.job_var}')
        self.commands = list()

    def set_conda_env(self, env):
        self.conda_env = env

    def add_modules(self, *module):
        for mod in module:
            self.modules.append(mod)

    def set_env_var(self, var, value):
        self.env_vars[var] = value

    def add_command(self, cmd, run=None, env_vars=None):
        if run is not None:
            cmd = f'{run} {cmd}'

        if env_vars:
            if not isinstance(env_vars, (list, tuple)):
                raise ValueError("'env_vars' must be a list or tuple. got %s" % type(env_vars))
            cmd = " ".join(f'{s}=${s}' for s in env_vars) + " " + cmd

        self.commands.append(cmd)

    def set_debug(self, debug):
        self.debug = debug

    def add_addl_jobflag(self, flag, val):
        self.addl_job_flags[flag] = val

    def submit_job(self, path):
        cmd = f'{self.submit_cmd} {path}'
        print(cmd)
        output = subprocess.check_output(
                    cmd,
                    stderr=subprocess.STDOUT,
                    shell=True).decode('utf-8')

        return self.extract_job_id(output)

    def extract_job_id(self, output):
        '''
        Example output:
        Job <338648> is submitted to queue <debug>.
        '''
        result = re.search(self.job_id_re, output)
        ret = None
        if result is not None:
            ret = int(result.groups(0)[0])
        else:
            print(f'Job submission failed: {output}')
        return ret

    def write_additional(self, f, options):
        pass

    @abstractmethod
    def write_run(self, f, command, command_options, options):
        pass

    def write_line(self, f, flag, value,):
        if value is not None:
            print(f'#{self.directive} -{flag} {value}', file=f)

    def _write_standard(self, f, options=None):
        if options is None:
            options = dict()
        self.write_line(f, self.queue_flag, self.debug_queue if self.debug else self.queue)
        self.write_line(f, self.project_flag, self.project)
        self.write_line(f, self.time_flag, self.time)
        self.write_line(f, self.nodes_flag, self.nodes)
        self.write_line(f, self.output_flag, self.output)
        self.write_line(f, self.error_flag, self.error)
        self.write_line(f, self.jobname_flag, self.jobname)
        if self.wait is not None:
            self.write_line(f, self.wait_flag, self.wait)
        for k, v in self.addl_job_flags.items():
            self.write_line(f, k, v)

    def write_header(self, f, options):
        print(f'#!/bin/bash', file=f)
        self._write_standard(f, options)
        self.write_additional(f, options)

    def write(self, f, options=None):
        self.write_header(f, options)
        print(file=f)
        for mod in self.modules:
            print(f'module load {mod}', file=f)
        print(file=f)
        if self.conda_env:
            print(f'conda activate {self.conda_env}', file=f)
        print(file=f)
        for k, v in self.env_vars.items():
            if isinstance(v, str):
                print(f'{k}="{v}"', file=f)
            else:
                print(f'{k}={v}', file=f)
        print(file=f)
        for c in self.commands:
            #if isinstance(c, dict):
            #    try:
            #        command = c['run']
            #        cmd_options = c.get('options')
            #        self.write_run(f, command, cmd_options, options)
            #    except KeyError as e:
            #        raise Exception("if using a dict as a command, it must have the key 'run' to indicate this is a resourced command")
            #else:
            #    print(c, file=f)
            print(c, file=f)
