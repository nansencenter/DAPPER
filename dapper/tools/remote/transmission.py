"""Tools related to running experimentes remotely

Requires rsync, gcloud and ssh access to the DAPPER cluster."""

from dapper import *
from pathlib import  Path
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse as _parse



def submit_job_GCP(xps_path):
    """GCP/HTCondor launcher"""
    xps_path = Path(xps_path)

    HOST = get_ip("condor-submit")
    # print_condor_status()
    print("autoscaler.py%s detected."%("" if detect_autoscaler() else " NOT"))
    sync_job(HOST,xps_path)

    print("Submitting jobs")
    try:
        remote_cmd(f"""cd {xps_path.name}; condor_submit submit-description""")
        remote_monitor_progress(xps_path.name)
    except KeyboardInterrupt as error:
        if input("Do you wish to clear the job queue? (Y/n): ").lower() in ["","y","yes"]:
            print("Clearing queue")
            remote_cmd("""condor_rm -a""")
        sys.exit(0)
    else:
        print("Downloading results")
        # Don't use individual scp's! (Pro: enables progbar. Con: it's SLOW!)
        nologs = " ".join("--exclude="+x for x in
                ["xp.com","out\\.*","err\\.*","run\\.*log","DAPPER"])
        sys_cmd(f"rsync -avz {nologs} {HOST}:~/{xps_path.name}/ {xps_path}")
    finally:
        # print("Checking for autoscaler cron job:") # let user know smth's happenin
        if not detect_autoscaler():
            print("Warning: autoscaler.py NOT detected!\n    "
                "Shut down the compute nodes yourself using:\n    "
                "gcloud compute instance-groups managed "
                "resize condor-compute-pvm-igm --size 0")


import subprocess
def sys_cmd(args,split=True):
    """Run subprocess, capture output, raise exception."""
    if split: args = args.split()
    try:
        ps = subprocess.run(args, check=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        # CalledProcessError doesnt print its .stderr, so we raise it this way:
        raise subprocess.SubprocessError(
            f"Command {error.cmd} returned non-zero exit status, "
            f"with stderr:\n{error.stderr.decode()}") from error
    output = ps.stdout.decode()
    return output

def remote_cmd(command):
    command = """--command=""" + command
    connect = "gcloud compute ssh condor-submit".split()
    output = sys_cmd(connect + [command], split=False)
    return output


def get_ip(instance):
    # cloud.google.com/compute/docs/instances/view-ip-address
    getip = 'get(networkInterfaces[0].accessConfigs[0].natIP)'
    ip = sys_cmd(f"gcloud compute instances describe {instance} --format={getip}")
    return ip.strip()

def print_condor_status():
    status = """condor_status -total"""
    status = remote_cmd(status)
    if status:
        print(status,":")
        for line in status.splitlines()[::4]: print(line)
    else:
        print("[No compute nodes found]")


def sync_DAPPER(HOST,xps_path):
    """DAPPER (as it currently exists, not a specific version)
    
    must be synced to the compute nodes, which don't have external IP addresses.
    """
    # Get list of files: whatever mentioned by .git
    repo  = f"--git-dir={dirs['DAPPER']}/.git"
    files = sys_cmd(f"git {repo} ls-tree -r --name-only HEAD").split()
    xcldd = lambda f: f.startswith("docs/") or f.endswith(".jpg") or f.endswith(".png")
    files = [f for f in files if not xcldd(f)]
    with open("synclist","w") as FILE:
        print("\n".join(files),file=FILE)

    print("Syncing DAPPER")
    try:
        sys_cmd(f"rsync -avz --files-from=synclist {dirs['DAPPER']} {HOST}:~/{xps_path.name}/DAPPER/")
    except subprocess.SubprocessError as error:
        # Suggest common source of error in the message.
        msg = error.args[0] + "\nDid you mv/rm files (and not registering it with .git)?"
        raise subprocess.SubprocessError(msg) from error
    finally:
        os.remove("synclist")

def sync_job(HOST,xps_path):

    jobs = [ixp for ixp in os.listdir(xps_path) if str(ixp).isnumeric()]

    print("Syncing %d jobs"%len(jobs))
    # NB: Note use of --delete. This rsync must come first!
    sys_cmd(f"rsync -avz --delete {dirs['DAPPER']}/dapper/tools/remote/htcondor/ {HOST}:~/{xps_path.name}")
    sys_cmd(f"rsync -avz {xps_path}/ {HOST}:~/{xps_path.name}")
    # print("Copying xp.com to initdir")
    remote_cmd(f"""cd {xps_path.name}; for ixp in [0-999999]; do cp xp.com $ixp/; done""")

    sync_DAPPER(HOST,xps_path)

    print("Syncing extra_files")
    extra_files = xps_path / "extra_files"
    sys_cmd(f"rsync -avz {extra_files}/ {HOST}:~/{xps_path.name}/extra_files/")


def detect_autoscaler(minutes=2):
    """Grep syslog for autoscaler.

    Also get remote's date (time), to avoid another (slow) ssh."""

    command = """grep CRON /var/log/syslog | grep autoscaler | tail; date"""
    output  = remote_cmd(command).splitlines()
    recent_crons, now = output[:-1], output[-1]

    if not recent_crons:
        return False

    # Get timestamp of last cron job
    last_cron = recent_crons[-1].split("condor-submit")[0]
    log_time = _parse(last_cron)
    now = _parse(now)
    # Make "aware" (assume /etc/timezone is utc):
    log_time = log_time.replace(tzinfo=timezone.utc)
    pause = timedelta(minutes=minutes)
    # spell_out(last_cron)
    # now = datetime.utcnow()
    # spell_out(log_time)
    # spell_out(pause)
    # spell_out(now)
    # spell_out(log_time < now-pause)


    if log_time + pause < now:
        return False
    else:
        return True

def remote_num_jobs(xps_path):
    return int(remote_cmd(f"""cd {xps_path}; ls -1 | grep -o '[0-9]*' | wc -l"""))

def remote_unfinished():
    condor_q = remote_cmd("""condor_q -totals""")
    return int(re.search("""(\d+) jobs;""",condor_q).group(1))

def remote_monitor_progress(xps_path):
    # Progress monitoring
    num_jobs = remote_num_jobs(xps_path)
    pbar = tqdm.tqdm(total=num_jobs,desc="Processing jobs")
    try:
        unfinished = num_jobs
        while unfinished:
            previously = unfinished
            unfinished = remote_unfinished()
            pbar.update(previously - unfinished)
            # print(num_jobs - unfinished, "jobs done")
            time.sleep(1) # ssh takes a while, so let's wait a bit extra
    finally:
        pbar.close()


