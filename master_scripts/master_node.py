import os
import subprocess
import time 
import signal
import argparse
import shlex
from recovery_utils import recover_function
from recovery_utils import determine_gpus
def parse_train_args (cmd):
    tokens = shlex.split(cmd)
    train_file = tokens[0]
    args = tokens[1:]
    parsed = {}
    i = 0 
    while i<len(args):
        if args[i].startswith('--'):
            key = args[i]
            if i+1 < len(args) and not args[i+1].startswith('--'):
                parsed[key] = args[i+1]
                i+=2
            else:
                parsed[key] = None 
                i+=1
        else:
            i+=1
    return train_file, parsed
def rebuil_cmd(path,args):
    cmd = [path]
    for key, val in args.item():
        cmd.append(key)
        if val is not None:
            cmd.append(str(val))
    full_cmd = " ".join(cmd)
    return full_cmd
def add_main_args():
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',allow_abbrev=False)
    group  = parser.add_argument_group("Inputs")
    group.add_argument('--train-cmd', type = str,default=None,  required=True, help='Commmand to run training along with its arguements')
    group.add_argument('--recovery-path', default=None, required=False, help='Path to recovery script')
    group.add_argument('--monitor-path', default=None, required=True, help='Path to GPU monitoring script')
    group.add_argument('--use-avail-gpus-on-reboot',action='store_true' ,default=None, help='On failure restart training using available gpus. If None program will wait till all gpus are online again')
    return parser
def start_proc(cmd):
    return subprocess.Popen(cmd, shell= True, preexec_fn=os.setsid)
def is_proc_alive(proc):
    return proc.poll() is None
def main():
    parser = add_main_args()
    args = parser.parse_args()
    train_file, train_args = parse_train_args(args.train_cmd)
    log_path = "./logs/train_run.log"
    gpu_count_org = determine_gpus()
    #Start training process and log 
    train_proc = start_proc(f"python {args.train_cmd} 2>&1 | tee {log_path}")
    monitor_proc = start_proc(f"python {args.monitor_path}")

    try: 
        while True:
            print("Launched")
            if not is_proc_alive(train_proc):
                train_started = False
                train_done = False
                with open(log_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if "training ..." in line:
                            train_started = True
                        if "after training is done" in line:
                            print("Training has successfully completed shutting down all processes")
                            os.killpg(os.getpgid(monitor_proc.pid), signal.SIGTERM)
                            exit()

                if train_started:
                    #do recovery
                    print("Booting up recovery")
                    if (args.use_avail_gpus_on_reboot) is None:
                        while (determine_gpus()!=gpu_count_org):
                            pass
                        train_proc = start_proc(f"python {args.train_cmd} 2>&1 | tee {log_path}")
                            
                    else:
                        #reset pp and tp and build train cmd
                        #default recovery function
                        if args.recovery_path is None:
                            result =   recover_function(train_args["--pipeline-model-parallel-size"],train_args["--tensor-model-parallel-size"],train_args["--hidden-size"],train_args["--num-layers"],train_args["--num-attention-heads "])
                            pp = result[1][1]
                            tp = result[1][0]
                        else:
                            #recovery script should just print out the recalc tp and pp values in the form "tp pp"
                            result = subprocess.run(["python3", args.recovery_path], capture_output=True, text=True)
                            output = result.stdout.strip()  # e.g., "1 2"
                            tp, pp = output.split()
                        train_args["--tensor-model-parallel-size"] = int(tp)

                        train_args["--pipeline-model-parallel-size"] = int(pp)
                        train_cmd = rebuil_cmd(train_file,train_args)
                        train_proc = start_proc(f"python {train_cmd} 2>&1 | tee {log_path}")
                    
                else:
                    print("Training Processes died before training started. Likely not a GPU issue. Check logs for details.")
                           
    #need exception different path for non gpu errors
    except KeyboardInterrupt:
        print("Shutting down")
        os.killpg(os.getpgid(train_proc.pid), signal.SIGTERM)
        os.killpg(os.getpgid(monitor_proc.pid), signal.SIGTERM)

if __name__ == "__main__":
    main()
