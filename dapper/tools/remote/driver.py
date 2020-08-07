"""Tools related to running experimentes remotely"""

from dapper import *
from datetime import datetime, timezone, timedelta
from dateutil.parser import parse as _parse

def remote_work(curjob):
    """GCP/HTCondor launcher"""
    jobs = [ixp for ixp in os.listdir(curjob) if ixp.startswith("ixp_")]

    assert remote_unfinished()==0, "Cannot submit jobs (would overwite currently running experiments)"

    # cloud.google.com/compute/docs/instances/view-ip-address
    getip = 'get(networkInterfaces[0].accessConfigs[0].natIP)'
    ip_submit = cmd(f"gcloud compute instances describe condor-submit --format={getip}").strip()
    # print("IP:", ip_submit)

    status = """condor_status -total"""
    print(f"you@condor-submit({ip_submit})$ " + status)
    status = remote_cmd(status)
    if status:
        for line in status.splitlines()[::4]: print(line)
    else:
        print("[No compute nodes found]")

    # TODO: sync autoscaler.py

    switch = "" if autoscaler_cronjob_ran_recently() else " NOT"
    print("autoscaler.py%s detected."%switch)

    print(f"Syncing {len(jobs)} jobs to **submit node**")
    cmd(f"rsync -avz --delete {curjob}/ {ip_submit}:~/job")

    print("Syncing submission description to **submit node**")
    cmd(f"rsync -avz /home/pnr/GCP/cluster/htcondor/ {ip_submit}:~/job")

    print("Syncing DAPPER to **cloud-storage**")
    # Sync via cloud-storage coz it's accessible to compute nodes 
    # even though they're not connected to the internet
    xcldd=".git|scripts|docs|.*__pycache__.*|.pytest_cache|DA_DAPPER.egg-info" 
    DPR = dirs['DAPPER']
    cmd(f"gsutil -m rsync -r -d -x {xcldd} {DPR} gs://pb2/DAPPER")


    print("Syncing current dir to **cloud-storage**")
    cmd(f"gsutil cp *.py gs://pb2/workdir/")


    print("Copying common_input to initdir of each job")
    remote_cmd("""cd job; for f in ixp_*; do cp common_input $f/; done""")

    print("Submitting jobs")
    try:
        remote_cmd("""cd job; condor_submit submit-description""")
        remote_monitor_progress()
    except KeyboardInterrupt as err:
        if input("Do you wish to clear the job queue? (Y/n): ").lower() in ["","y","yes"]:
            print("Clearing queue")
            remote_cmd("""condor_rm -a""")
        sys.exit(0)
    else:
        print("Downloading results")
        # Don't use individual scp's! (Pro: enables progbar. Con: it's SLOW!)
        nologs = " ".join("--exclude="+x for x in
                ["common_input","out\\.*","err\\.*","run\\.*log"])
        cmd(f"rsync -avz {nologs} {ip_submit}:~/job/ {curjob}")
    finally:
        # print("Checking for autoscaler cron job:") # let user know smth's happenin
        if not autoscaler_cronjob_ran_recently():
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

def autoscaler_cronjob_ran_recently(minutes=2):

    # Grep syslog for autoscaler.
    # Also get remote's date (time), to avoid another (slow) ssh.
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

def remote_num_jobs():
    return int(remote_cmd("""cd job; ls -1 | grep ixp | wc -l"""))

def remote_unfinished():
    condor_q = remote_cmd("""condor_q -totals""")
    return int(re.search("""(\d+) jobs;""",condor_q).group(1))

def remote_monitor_progress():
    # Progress monitoring
    num_jobs = remote_num_jobs()
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


