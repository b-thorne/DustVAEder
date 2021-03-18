import os
import datetime
from pathlib import Path
import tensorflow as tf
import os, pwd
from tensorboard import notebook
import getpass
from IPython.core.display import display, HTML

def setup_vae_run_logging(chkpt_dir_name):
    """ Function to setup the logging for a given run. This requires the
    environment variable TF_LOGDIR to exist. If it does not exist an error
    will be thrown. 
    
    Parameters
    ----------
    chkpt_dir_name: str
        String identifying the name of the subdir in which to checkpoint.
        
    Notes
    -----
    Will create logging directory if it does not exist.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_dir = Path(os.environ['TF_LOGDIR']) 
    network_logdir = logging_dir / chkpt_dir_name
    network_logdir.mkdir(exist_ok=True)
    summary_writer = tf.summary.create_file_writer(str(network_logdir / current_time))
    return summary_writer

def get_pid_owner(pid):
    # the /proc/PID is owned by process creator
    proc_stat_file = os.stat("/proc/%d" % pid)
    # get UID via stat call
    uid = proc_stat_file.st_uid
    # look up the username from uid
    username = pwd.getpwuid(uid)[0]
    
    return username

def get_tb_port(username):
    
    for tb_nb in notebook.manager.get_all():
        if get_pid_owner(tb_nb.pid) == username:
            return tb_nb.port
    
def tb_address():
    
    username = getpass.getuser()
    tb_port = get_tb_port(username)
    
    address = "https://jupyter.nersc.gov" + os.environ['JUPYTERHUB_SERVICE_PREFIX'] + 'proxy/' + str(tb_port) + "/"

    address = address.strip()
    
    display(HTML('<a href="%s">%s</a>'%(address,address)))