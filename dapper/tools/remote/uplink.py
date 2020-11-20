"""Tools related to running experimentes remotely

Requires rsync, gcloud and ssh access to the DAPPER cluster."""

# TODO 5: use Fabric? https://www.fabfile.org/

import time
from tqdm import tqdm
from dapper.dpr_config import rc
import dapper.tools.utils as utils
from datetime import timezone, timedelta
from dateutil.parser import parse as datetime_parse
import os
import tempfile
import subprocess


class SubmissionConnection:
    """Establish multiplexed ssh to a given submit-node for a given xps_path."""

    def __init__(self,
                 xps_path,
                 name="condor-submit",
                 zone="us-central1-f",
                 proj="mc-tut"):
        # Job info
        self.xps_path = xps_path
        # Submit-node info
        self.name     = name
        self.proj     = proj
        self.zone     = zone
        self.host     = f"{name}.{zone}.{proj}"
        # instance name (as viewed by system ssh)
        self.ip       = get_ip(name)

        print("Preparing ssh connection")
        sys_cmd("gcloud compute config-ssh")
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
        print("autoscaler.py%s detected" % ("" if detect_autoscaler(self) else " NOT"))

    def remote_cmd(self, cmd_string):
        """Run command at self.host via multiplexed ssh."""
        # Old version (uses gcloud):
        #     command = """--command=""" + command
        #     connect = "gcloud compute ssh condor-submit".split()
        #     output = sys_cmd(connect + [command], split=False)
        return sys_cmd([*self.ssh_M.split(), self.ip, cmd_string], split=False)

    def rsync(self, src, dst, opts=[], rev=False, prog=False, dry=False, use_M=True):
        # Prepare: opts
        if isinstance(opts, str):
            opts = opts.split()

        # Prepare: src, dst
        src = str(src)
        dst = str(dst)
        dst = self.ip + ":" + dst
        if rev:
            src, dst = dst, src

        # Show progress
        if prog:
            # TODO 3: Implement rsync check for when new rsync
            # isnt available which supports --info=progress2
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
            _ = subprocess.run(cmd, check=True)
            return None


def submit_job_GCP(xps_path, **kwargs):
    """GCP/HTCondor launcher"""
    sc = SubmissionConnection(xps_path, **kwargs)
    sync_job(sc)

    # TODO 2: use subprocess.run(timeout=5*60) instead?
    try:
        # Timeout functionality for ssh -c submit
        def submit():
            print("Submitting jobs")
            sc.remote_cmd(
                f"""cd {xps_path.name}; condor_submit"""
                f""" -batch-name {xps_path.name} submit-description""")
        import multiprocessing_on_dill as mpd  # fails on GCP
        p = mpd.Process(target=submit)
        p.start()
        p.join(5*60)
        if p.is_alive():
            print('The ssh seems to hang, '
                  'but the jobs are probably submitted.')
            p.terminate()
            p.join()

        # Prepare download command
        print("To download results (before completion) use:")
        xcldd = ["xp.com", "DAPPER", "runlog", "err"]  # "out\\.*"
        xcldd = ["--exclude="+x for x in xcldd]
        print(sc.rsync(
            xps_path.parent, f"~/{xps_path.name}",
            rev=True, opts=xcldd, dry=True, use_M=False))

        monitor_progress(sc)
    except Exception:
        inpt = input("Do you wish to clear the job queue? (Y/n): ").lower()
        if inpt in ["", "y", "yes"]:
            print("Clearing queue")
            clear_queue(sc)
        raise
    else:
        print("Downloading results")
        sc.rsync(xps_path.parent, f"~/{xps_path.name}", rev=True, opts=xcldd, prog=True)
    finally:
        # print("Checking for autoscaler cron job:") # let user know smth's happenin
        if not detect_autoscaler(sc):
            print("Warning: autoscaler.py NOT detected!\n    "
                  "Shut down the compute nodes yourself using:\n    "
                  "gcloud compute instance-groups managed "
                  "resize condor-compute-pvm-igm --size 0")


def detect_autoscaler(self, minutes=10):
    """Grep syslog for autoscaler.

    Also get remote's date (time), to avoid another (slow) ssh."""

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


def sync_job(self):
    xps_path = self.xps_path

    jobs = list_job_dirs(xps_path)
    print("Syncing %d jobs" % len(jobs))

    # NB: --delete => Must precede other rsync's!
    self.rsync(xps_path, "~/", "--delete")

    htcondor = str(rc.dirs.DAPPER/"dapper"/"tools"/"remote"/"htcondor") + os.sep
    self.rsync(htcondor, "~/"+xps_path.name)

    print("Copying xp.com to initdirs")
    self.remote_cmd(
        f"""cd {xps_path.name}; for ixp in [0-999999];"""
        f""" do cp xp.com $ixp/; done""")

    sync_DAPPER(self)


def sync_DAPPER(self):
    """Sync DAPPER (as it currently exists, not a specific version)

    to compute-nodes, which don't have external IP addresses.
    """
    # Get list of files: whatever mentioned by .git
    repo  = f"--git-dir={rc.dirs.DAPPER}/.git"
    files = sys_cmd(f"git {repo} ls-tree -r --name-only HEAD").split()

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
        # Suggest common source of error in the message.
        msg = error.args[0] + \
            "\nDid you mv/rm files (and not registering it with .git)?"
        raise subprocess.SubprocessError(msg) from error


def print_condor_status(self):
    status = """condor_status -total"""
    status = self.remote_cmd(status)
    if status:
        print(status, ":")
        for line in status.splitlines()[::4]:
            print(line)
    else:
        print("[No compute nodes found]")


def clear_queue(self):
    """Use condor_rm to clear the job queue of the submission."""
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


def get_job_status(self):
    """Parse condor_q to get number idle, held, etc, jobs"""
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


def monitor_progress(self):
    """Use condor_q to monitor job progress."""
    num_jobs = len(list_job_dirs(self.xps_path))
    pbar = tqdm(total=num_jobs, desc="Processing jobs")
    try:
        unfinished = num_jobs
        while unfinished:
            job_status     = get_job_status(self)
            unlisted       = num_jobs - job_status["jobs"]  # completed w/ success
            finished       = job_status["held"] + unlisted  # failed + suceeded
            unfinished_new = num_jobs - finished
            increment      = unfinished - unfinished_new
            unfinished     = unfinished_new
            # print(job_status)
            pbar.update(increment)
            time.sleep(1)  # dont clog the ssh connection
    except Exception:
        print("Some kind of exception occured,"
              " while %d jobs were not even run." % unfinished)
        raise
    else:
        print("All jobs finished without failure.")
    finally:
        pbar.close()
        if job_status["held"]:
            print("NB: There were %d failed jobs" % job_status["held"])
            print(f"View errors at {self.xps_path}/JOBNUMBER/out")
            clear_queue(self) # TODO 3: this runs also if sucessfull. Ok?


def list_job_dirs(xps_path):
    dirs = [xps_path/d for d in utils.sorted_human(os.listdir(xps_path))]
    return [d for d in dirs if d.is_dir() and d.stem.isnumeric()]


def get_ip(instance):
    """Get ip-address of instance.

    NB: the use of IP rather than the ``Host`` listed in ``.ssh/config``
    (eg ``condor-submit.us-central1-f.mc-tut``,
    as generated by ``gcloud compute config-ssh``)
    requires ``AddKeysToAgent yes`` under ``Host *`` in ``.ssh/config``,
    and that you've already logged into the instance once using (eg)
    ``ssh condor-submit.us-central1-f.mc-tut``.
    """

    # cloud.google.com/compute/docs/instances/view-ip-address
    getip = 'get(networkInterfaces[0].accessConfigs[0].natIP)'
    ip = sys_cmd(f"gcloud compute instances describe {instance} --format={getip}")
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


def sys_cmd(args, split=True):
    """Run subprocess, capture output, raise exception."""
    if split:
        args = args.split()
    try:
        ps = subprocess.run(args, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        # CalledProcessError doesnt print its .stderr,
        # so we raise it this way:
        raise subprocess.SubprocessError(
            f"Command {error.cmd} returned non-zero exit status, "
            f"with stderr:\n{error.stderr.decode()}") from error
    output = ps.stdout.decode()
    return output
