"""
RUN THIS IF YOU EXPERIENCE XFOIL.EXE HANGING INDEFINITELY,
MAKE SURE THE TIMER IS >= TO THE TIMEOUT PARAMETER IN RUN_XFOIL.PY
"""
import subprocess
import time

import psutil


def main():
    print('Running xfoil monitoring system')
    timer = 50
    past_pids = []
    while True:
        past_pids, kill_pids = fetch_xfoil_pid(past_pids)
        for pid in kill_pids:
            subprocess.run(f'taskkill /f /pid {pid}')

        print(f'Sleeping {timer} sec ...')
        time.sleep(timer)


def fetch_xfoil_pid(past_pids):
    last_pids, kill_pids = [], []

    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == 'xfoil.exe':
            pid = proc.info['pid']
            last_pids.append(pid)
            if pid in past_pids:
                kill_pids.append(pid)

    return last_pids, kill_pids


if __name__ == "__main__":
    main()
