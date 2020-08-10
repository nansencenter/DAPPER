"""Tools related to running experimentes remotely"""

from dapper import *
from pathlib import  Path
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse as _parse

# TODO: use git ls-tree instead of xcldd
# TODO: allow multiple jobs simultaneously (by de-confusing progbar?)


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
    except KeyboardInterrupt as err:
        if input("Do you wish to clear the job queue? (Y/n): ").lower() in ["","y","yes"]:
            print("Clearing queue")
            remote_cmd("""condor_rm -a""")
        sys.exit(0)
    else:
        print("Downloading results")
        # Don't use individual scp's! (Pro: enables progbar. Con: it's SLOW!)
        nologs = " ".join("--exclude="+x for x in
                ["xp_com","out\\.*","err\\.*","run\\.*log"])
        cmd(f"rsync -avz {nologs} {HOST}:~/{xps_path.name}/ {xps_path}")
    finally:
        # print("Checking for autoscaler cron job:") # let user know smth's happenin
        if not detect_autoscaler():
            print("Warning: autoscaler.py NOT detected!\n    "
                "Shut down the compute nodes yourself using:\n    "
                "gcloud compute instance-groups managed "
                "resize condor-compute-pvm-igm --size 0")


import subprocess
def cmd(args,split=True):
    """Run subprocess and communicate."""
    if split:
        args = args.split()
    ps = subprocess.run(args, check=True, capture_output=True)
    output = ps.stdout.decode()
    return output

def remote_cmd(command):
    command = """--command=""" + command
    connect = "gcloud compute ssh condor-submit".split()
    output = cmd(connect + [command],split=False)
    return output


def get_ip(instance):
    # cloud.google.com/compute/docs/instances/view-ip-address
    getip = 'get(networkInterfaces[0].accessConfigs[0].natIP)'
    ip = cmd(f"gcloud compute instances describe {instance} --format={getip}")
    return ip.strip()

def print_condor_status():
    status = """condor_status -total"""
    status = remote_cmd(status)
    if status:
        print(status,":")
        for line in status.splitlines()[::4]: print(line)
    else:
        print("[No compute nodes found]")

def sync_job(HOST,xps_path):

    print("Syncing submission description to **submit node**")
    # NB: this rsync must come first coz it uses --delete which deletes all other contents job/
    cmd(f"rsync -avz --delete {dirs['DAPPER']}/dapper/tools/remote/htcondor/ {HOST}:~/{xps_path.name}")

    jobs = [ixp for ixp in os.listdir(xps_path) if ixp.startswith("ixp_")]
    print("Syncing %d jobs to **submit node**"%len(jobs))
    cmd(f"rsync -avz {xps_path}/ {HOST}:~/{xps_path.name}")

    
    print("Syncing DAPPER")
    files = cmd(f"git --git-dir={dirs['DAPPER']}/.git ls-tree -r --name-only HEAD")
    with open("dpr_files","w") as FILE:
        print(files,file=FILE)
    cmd(f"rsync -avz --files-from=dpr_files {dirs['DAPPER']} {HOST}:~/DAPPER/")

    # Sync via cloud-storage coz it's accessible to compute nodes 
    # even though they're not connected to the internet
    xcldd=".git|scripts|docs|.*__pycache__.*|.pytest_cache|DA_DAPPER.egg-info|old_version.zip" 
    cmd(f"gsutil -m rsync -r -d -x {xcldd} {dirs['DAPPER']} gs://pb2/DAPPER")

    # TODO: also uncomment corresponding stuff in run_job.sh
    # print("Syncing current dir to **cloud-storage**")
    # cmd(f"gsutil cp *.py gs://pb2/workdir/")

    print("Copying xp_com to initdir of each job")
    remote_cmd(f"""cd {xps_path.name}; for ixp in ixp_*; do cp xp_com $ixp/; done""")


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
    return int(remote_cmd(f"""cd {xps_path}; ls -1 | grep ixp | wc -l"""))

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


