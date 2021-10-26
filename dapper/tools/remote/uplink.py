"""Tools related to running experimentes remotely

Requires rsync, gcloud and ssh access to the DAPPER cluster.
"""

# TODO 9: use Fabric? https://www.fabfile.org/

import os
import subprocess
import tempfile
import time
from datetime import timedelta, timezone

from dateutil.parser import parse as datetime_parse
from patlib.std import sorted_human
from tqdm import tqdm

from dapper.dpr_config import rc


class SubmissionConnection:
    """Establish multiplexed ssh to a given submit-node for a given xps_path."""

    def __init__(self,
                 xps_path,
                 name="condor-submit",
                 zone="us-central1-f",
                 proj="mc-tut"):
        # Job info
        self.xps_path = xps_path
        self.nJobs    = len(list_job_dirs(xps_path))
        # Submit-node info
        self.name     = name
        self.proj     = proj
        self.zone     = zone
        self.host     = f"{name}.{zone}.{proj}"
        # instance name (as viewed by system ssh)
        self.ip       = get_ip(name)

        print("Preparing ssh connection")
        sub_run("gcloud compute config-ssh", shell=True)
        # Use multiplexing to enable simultaneous connections.
        # Possible alternative: alter MaxStartups in sshd_config,
        # or other configurations:
        # - https://stackoverflow.com/a/36654900/38281
        # - https://unix.stackexchange.com/a/226460
        # - https://superuser.com/a/1032667/142925
        self.ssh_M = (
            '''ssh -o ControlMaster=auto'''
            ''' -o ControlPath=~/.ssh/%r@%h:%p.socket -o ControlPersist=1m''')

        # print_condor_status()
        print("autoscaler.py%s detected" % ("" if _detect_autoscaler(self) else " NOT"))

    def remote_cmd(self, cmd_string, **kwargs):
        """Run command at self.host via multiplexed ssh."""
        # Old version (uses gcloud):
        #     command = """--command=""" + command
        #     connect = "gcloud compute ssh condor-submit".split()
        #     output = sub_run(connect + [command])
        return sub_run([*self.ssh_M.split(), self.ip, cmd_string], **kwargs)

    def rsync(self, src, dst, opts=(), rev=False, prog=False, dry=False, use_M=True):
        # Prepare: opts
        if isinstance(opts, str):
            opts = opts.split()

        # Prepare: src, dst
        src = str(src)
        dst = str(dst)
        dst = self.ip + ":" + dst
        if rev:
            src, dst = dst, src

        # Get rsync version
        v = sub_run("rsync --version", shell=True).splitlines()[0].split()
        i = v.index("version")
        v = v[i+1]  # => '3.2.3'
        v = [int(w) for w in v.split(".")]
        has_prog2 = (v[0] >= 3) and (v[1] >= 1)

        # Show progress
        if prog and has_prog2:
            prog = ("--info=progress2", "--no-inc-recursive")
        else:
            prog = []

        # Use multiplex
        multiplex = []
        if use_M:
            multiplex = "-e", self.ssh_M
        else:
            multiplex = []

        # Assemble command
        cmd = ["rsync", "-azh", *prog, *multiplex, *opts, src, dst]

        if dry:
            # Dry run
            return " ".join(cmd)
        else:
            # Sync
            subprocess.run(cmd, check=True)
            return None


def submit_job_GCP(xps_path, **kwargs):
    """GCP/HTCondor launcher"""
    sc = SubmissionConnection(xps_path, **kwargs)
    _sync_job(sc)
    _submit_job(sc)

    # Prepare download command
    print("To download results (before completion) use:")
    xcldd = ["xp.com", "DAPPER", "runlog", "err"]  # "out\\.*"
    xcldd = ["--exclude="+x for x in xcldd]
    print(sc.rsync(
        xps_path.parent, f"~/{xps_path.name}",
        rev=True, opts=xcldd, dry=True, use_M=False))

    try:
        _monitor_progress(sc)
    except (KeyboardInterrupt, Exception):
        inpt = input("Do you wish to clear the job queue? (Y/n): ").lower()
        if inpt in ["", "y", "yes"]:
            print("Clearing queue")
            _clear_queue(sc)
        raise
    else:
        print("Downloading results")
        sc.rsync(xps_path.parent, f"~/{xps_path.name}",
                 rev=True, opts=xcldd, prog=True)
    finally:
        # let user know smth's happenin
        # print("Checking for autoscaler cron job:")
        if not _detect_autoscaler(sc):
            print("Warning: autoscaler.py NOT detected!\n    "
                  "Shut down the compute nodes yourself using:\n    "
                  "gcloud compute instance-groups managed "
                  "resize condor-compute-pvm-igm --size 0")

        # Check for errors among jobs
        nHeld = _get_job_status(sc)["held"]
        if nHeld:
            if nHeld == sc.nJobs:
                print("NB: All jobs failed")
            else:
                print("NB: There were %d failed jobs" % nHeld)
            # Print path of error messages for a (the first found) failed job
            for d in list_job_dirs(sc.xps_path):
                if (d / "xp").stat().st_size == 0:
                    print(f"View error message at {d / 'out'}")
                    break


def _detect_autoscaler(self, minutes=10):
    """Grep syslog for autoscaler.

    Also get remote's date (time), to avoid another (slow) ssh.
    """
    command = """grep CRON /var/log/syslog | grep autoscaler | tail; date"""
    output  = self.remote_cmd(command).splitlines()
    recent_crons, now = output[:-1], output[-1]

    if not recent_crons:
        return False

    # Get timestamp of last cron job
    last_cron = recent_crons[-1].split(self.name)[0]
    log_time  = datetime_parse(last_cron).replace(tzinfo=timezone.utc)
    now       = datetime_parse(now)
    pause     = timedelta(minutes=minutes)

    if log_time + pause < now:
        return False

    return True


def _submit_job(self):
    print("Submitting jobs")
    xps_path = self.xps_path

    # I used to have a timeout in the remote_cmd() below,
    # because I thought that ssh would hang when the condor submission
    # was taking long. However, I think it is simply the submission
    # being slow, and we had better wait for it to finish,
    # because querying condor_q before then will fail,
    # causing _monitor_progress to crash.
    if self.nJobs > 4000:
        print("This might take a while")

    self.remote_cmd(
        f"""cd {xps_path.name}; condor_submit"""
        f""" -batch-name {xps_path.name} submit-description""")


def _sync_job(self):
    print("Syncing %d jobs" % self.nJobs)
    xps_path = self.xps_path

    # NB: --delete => Must precede other rsync's!
    self.rsync(xps_path, "~/", "--delete")

    htcondor = str(rc.dirs.DAPPER/"dapper"/"tools"/"remote"/"htcondor") + os.sep
    self.rsync(htcondor, "~/"+xps_path.name)

    _sync_DAPPER(self)


def _sync_DAPPER(self):
    """Sync DAPPER (as on work-tree, not a specific version) to GCP."""
    # Get list of files: whatever mentioned by .git
    repo  = f"--git-dir={rc.dirs.DAPPER}/.git"
    files = sub_run(f"git {repo} ls-tree -r --name-only HEAD", shell=True).split()

    def xcldd(f):
        return f.startswith("docs/") or f.endswith(".jpg") or f.endswith(".png")
    files = [f for f in files if not xcldd(f)]

    with tempfile.NamedTemporaryFile("w", delete=False) as synclist:
        print("\n".join(files), file=synclist)

    print("Syncing DAPPER")
    try:
        self.rsync(
            rc.dirs.DAPPER,
            f"~/{self.xps_path.name}/DAPPER",
            "--files-from="+synclist.name)
    except subprocess.SubprocessError as error:
        print(error.stderr)
        print("Did you mv/rm files (and not register it with `git`)?")
        raise


def print_condor_status(self):
    status = """`condor_status` -total"""
    status = self.remote_cmd(status)
    if status:
        print(status, ":")
        for line in status.splitlines()[::4]:
            print(line)
    else:
        print("[No compute nodes found]")


def _clear_queue(self):
    """Use `condor_rm` to clear the job queue of the submission."""
    try:
        batch = f"""-constraint 'JobBatchName == "{self.xps_path.name}"'"""
        self.remote_cmd(f"""condor_rm {batch}""")
        print("Queue cleared.")
    except subprocess.SubprocessError as error:
        if "matching" in error.args[0]:
            # Queue probably already cleared, as happens upon
            # KeyboardInterrupt, when there's also "held" jobs.
            pass
        else:
            raise


def _get_job_status(self):
    """Parse `condor_q` to get number idle, held, etc, jobs"""
    # The autoscaler.py script from Google uses
    # 'condor_q -totals -format "%d " Jobs -format "%d " Idle -format "%d " Held'
    # But in both condor versions I've tried, -totals does not mix well with -format,
    # and the ClassAd attributes ("Jobs", "Idle", "Held") are not available,
    # as listed by:
    #  - condor_q -l
    #  - Appendix "Job ClassAd Attributes" of the condor-manual (online).
    #  Condor version 8.6 (higher than 8.4 used by GCP tutorial)
    #  enables labelling jobs with -batch-name, and thus multiple jobs
    #  can be submitted and run (queried for progress, rm'd, downloaded) simultaneously.
    #  One alternative is to query job status with
    #  condor_q -constraint 'JobStatus == 5',
    #  but I prefer to parse the -totals output instead.

    batch = f"""-constraint 'JobBatchName == "{self.xps_path.name}"'"""
    qsum = self.remote_cmd(f"""condor_q {batch}""").split()
    status = dict(jobs="jobs;", completed="completed,", removed="removed,",
                  idle="idle,", running="running,", held="held,", suspended="suspended")
    # Another way to get total num. of jobs:
    # int(self.remote_cmd(
    #     f"""cd {self.xps_path.name}; ls -1 | grep -o '[0-9]*' | wc -l"""))
    # Another way to parse qsum:
    # int(re.search("""(\d+) idle""",condor_q).group(1))
    return {k: int(qsum[qsum.index(v)-1]) for k, v in status.items()}


def _monitor_progress(self):
    """Use condor_q to monitor job progress."""
    num_jobs = self.nJobs
    pbar = tqdm(total=num_jobs, desc="Processing jobs")
    try:
        unfinished = num_jobs
        while unfinished:
            job_status     = _get_job_status(self)
            unlisted       = num_jobs - job_status["jobs"]  # completed w/ success
            finished       = job_status["held"] + unlisted  # failed + suceeded
            unfinished_new = num_jobs - finished
            increment      = unfinished - unfinished_new
            unfinished     = unfinished_new
            # print(job_status)
            pbar.update(increment)
            time.sleep(1)  # dont clog the ssh connection
    except (KeyboardInterrupt, Exception):
        print("Some kind of exception occured,"
              " while %d jobs have not even run." % unfinished)
        raise
    finally:
        pbar.close()


def list_job_dirs(xps_path):
    dirs = sorted_human(os.listdir(xps_path))
    dirs = [xps_path/d for d in dirs]
    dirs = [d for d in dirs if d.is_dir() and d.stem.isnumeric()]
    return dirs


def get_ip(instance):
    """Get ip-address of instance.

    NB: the use of IP rather than the `Host` listed in `.ssh/config`
    (eg `condor-submit.us-central1-f.mc-tut`,
    as generated by `gcloud compute config-ssh`)
    requires `AddKeysToAgent yes` under `Host *` in `.ssh/config`,
    and that you've already logged into the instance once using (eg)
    `ssh condor-submit.us-central1-f.mc-tut`.
    """
    # cloud.google.com/compute/docs/instances/view-ip-address
    getip = 'get(networkInterfaces[0].accessConfigs[0].natIP)'
    ip = sub_run((f"gcloud compute instances describe {instance}"
                 f" --format={getip}").split())
    return ip.strip()

    # # Parse ssh/config for the "Host" of condor-submit.
    # # Q: how reliable/portable is it?
    # from pathlib import Path
    # with open(Path("~").expanduser()/".ssh"/"config") as ssh_config:
    #     for ln in ssh_config:
    #         if ln.startswith("Host condor-submit"):
    #             break
    #     else:
    #         raise RuntimeError(
    #             "Did not find condor-submit Host in .ssh/config.")
    #     return ln[ln.index("condor"):].strip()


def sub_run(*args, check=True, capture_output=True, text=True, **kwargs):
    r"""Do `subprocess.run`, with responsive defaults.

    Examples:
    >>> gitfiles = sub_run(["git", "ls-tree", "-r", "--name-only", "HEAD"])  # or:
    >>> # gitfiles = sub_run("git ls-tree -r --name-only HEAD", shell=True)
    """
    try:
        x = subprocess.run(
            *args, **kwargs,
            check=check, capture_output=capture_output, text=text)

    except subprocess.CalledProcessError as error:
        if capture_output:
            # error.args += (f"The stderr is: \n\n{error.stderr}",)
            # The above won't get printed because CalledProcessError.__str__
            # is non-standard. Instead, print stdout (on top of stack trace):
            print(error.stderr)
        else:
            pass  # w/o capture_output, error is automatically printed.
        raise

    if capture_output:
        return x.stdout
