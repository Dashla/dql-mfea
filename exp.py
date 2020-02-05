#! /usr/bin/env python3

import task
import mfea
import signal
import pickle
import kerpy
import argparse
import DQLTask
from os import mkdir


def create_env(name):
    conf = {'verbose': 0, 'episodes': 50, 'max_steps': 300, 'visualize': False}
    return DQLTask.DQLTask(name, conf)


def handler(signum, frame):
    import pdb
    pdb.set_trace()


# EXP ------------------------------
# ! Activate to enable debug after CTRL+C
# signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":

    # + Take output folder from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='OUTPUT')
    parser.add_argument('--filename', type=str, default='environments.txt')
    args = parser.parse_args()

    try:
        mkdir(args.path)
    except FileExistsError:
        print("Warning: Directory exists")

    # + Definition of environments and configuration
    with open(args.filename, 'r') as f:
        names = list(f.read().split('\n')[0].split(','))
        names = [n.replace(' ', '') for n in names]

    # + Codify using the last layer raw scheme and run algorithm
    visualize = False
    cod_type = 1

    for run in range(5):
        envs = [create_env(x) for x in names]
        print(f"number of tasks: {len(envs)}, RUN{run}/{5}")
        DQLTask.compute_indexes(envs)
        tasks = [task.Task(x.D, x.fnc, 1, -1, x) for x in envs]

        res = mfea.mfea(tasks, reps=1, gen=60, pop=100, codification_type=cod_type)

        # + Save data
        pickle.dump(res, open(args.path+'/'+'res_'+str(run)+'.pickle', 'wb'))

        for i, cand in enumerate(res[2][0]):
            p = args.path+'/'+names[i]
            pickle.dump(envs[i].evhistory, open(p+'_'+str(run)+'.pickle', 'wb'))
            sol = tasks[i].decode(res[2][0][i])
            pickle.dump(sol, open(p+'_fnc_sol'+str(run)+'.pickle', 'wb'))

            if cod_type == 0:
                envs[i].save_processed_candidate(sol,
                                                 name=names[i]+'_'+str(run),
                                                 folder_name=args.path)
            else:
                envs[i].save_processed_candidate_1(sol,
                                                   name=names[i]+'_'+str(run),
                                                   folder_name=args.path)

            print(f"Data saved ({envs[i].fullname}). Press Enter to continue...")
            # input()

            # + Visualize results, can be deleted
            if visualize:
                envs[i].visualize = True
                tasks[i].dqltask.load_model(p+'_'+str(run)+'_weights.h5f')
                tasks[i].dqltask.test()
