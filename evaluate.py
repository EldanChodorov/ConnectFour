import argparse
import csv
import os
import shutil
import subprocess as sp
import sys
import time
import datetime
import pickle
from itertools import combinations
from collections import Counter


DETECT_PROBLEMS_DURATION = 200
SELFPLAY_DURATION = 100000
RANDOM_DURATION = 200
MINMAX_DURATION = 200
TOURNAMENT_DURATION = 200


# divert to the "right" interpreter
INTERPRETER = '/cs/usr/dsgissin/connect4/connect4env/bin/python3.5'
if not sys.executable == INTERPRETER:
    scriptpath = os.path.abspath(sys.modules[__name__].__file__)
    sp.Popen([INTERPRETER, scriptpath] + sys.argv[1:]).wait()
    exit()


code_path = os.path.abspath('/cs/usr/dsgissin/connect4/') + '/'
archive_path = os.path.abspath('/cs/usr/dsgissin/connect4/archive/') + '/'


def parse_policies(policies_file):
    return {p['id'] : (p['short_name'], p['paragraph']) for p in csv.DictReader(open(policies_file))}


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('policies_file', type=str, help="file name for a csv file with <id>,<short_name>,<paragraph>")
    p.add_argument('scoreboard_files', type=str, help="comma-sparated file names for the scoreboard to be written to")
    p.add_argument('tournament_scoreboard_files', type=str, help="comma-sparated file names for the scoreboard to be written to")
    p.add_argument('--score_archive', '-a', default=None, type=str,
                   help="file name to which results are archived after every stage")
    p.add_argument('--base_duration', '-d', type=int, help="number of rounds in basic game", default=50)
    p.add_argument('--handle_previous_run', '-r', choices=['rm','arch'], default='rm',
                   help="whether previous results should be removed or archived")
    args = p.parse_args()
    folders = [code_path + f for f in ['output','scripts','models']]
    if args.handle_previous_run == 'rm':
        for f in folders:
            try: shutil.rmtree(f)
            except IOError: pass
    else:
        prev = [int(x) for x in os.listdir(archive_path) if os.path.isdir(archive_path+x)]
        mvto = archive_path + str(1 + max(prev)) if prev else '0'
        for f in folders:
            try: shutil.move(f, mvto)
            except IOError: pass

    for f in folders: os.mkdir(f)
    return args.policies_file, args.scoreboard_files.split(','), args.tournament_scoreboard_files.split(','), args.score_archive, args.base_duration


def script_string(python_comm, output_file, ncpu=1, mem='1G', t='47:00:00'):
    """
    return the script string for the bash files.
    :param python_comm: the python command to run
    :param output_file: the output file for the SLURM log
    :param ncpu: number of cpus to use
    :param mem: amount of memory to allocate
    :return: the formatted string
    """
    return  """#! /bin/bash
#SBATCH --cpus-per-task={ncpu}
#SBATCH --output={ofile}
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={time}

{pcom}""".format(ncpu=ncpu, ofile=output_file,mem=mem, pcom=python_comm, time=t)


def test_vs_random(policies, best_model_files, script_folder=None, repeats=1, dur=int(100)):
    """
    test the policies against a random policy.
    :param policies: a dictionary of policies.
    :param script_folder: where to put the scripts for the cluster.
    :param repeats: number of tries for each policy.
    :param dur: length of session for each match.
    :return: list of file paths of results (to be parsed later).
    """
    if script_folder is None:
        script_folder = code_path + 'scripts/'
    result_files = []
    for r in range(repeats):
        for pid in policies:
            script_name = script_folder + pid + '.%i.random.bash' % r
            res_file = '%soutput/random_%s_%i.res' % (code_path, pid, r)
            out_file = '%soutput/random_%s_%i.out' % (code_path, pid, r)
            log_file = '%soutput/random_%s_%i.log' % (code_path, pid, r)
            model_folder = '%smodels' % code_path
            model_file = 'random_%s_%i.pkl' % (pid, r)
            result_files.append(res_file)
            pcom = ('python3 {p}Connect4.py '
                    '-A "{pid}(save_to={model_file},load_from={early_model});RandomAgent()" '
                    '-D {d} '
                    '-mf {model_folder} '
                    '-l {log_file} '
                    '-o {o} '
                    '-bi RandomBoard '
                    '-pat 0.1 '
                    '-t "test" ').format \
                (p=code_path,pid=pid,d=str(dur),o=res_file,model_file=model_file,
                 model_folder=model_folder, log_file=log_file,early_model=best_model_files[pid])
            with open(script_name, 'w') as script_file:
                script_file.write(script_string(pcom,out_file))
            out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
            jid = out[0].decode('utf').split(' ')[-1].strip()
            print("submitted job %s (%s) for Random test (%i) of policy %s" % (jid, script_name, r, pid))
    return result_files


def test_vs_minmax(policies, best_model_files, script_folder=None, repeats=1, dur=int(100), depth=1):
    """
    test the policies against a Minmax policy.
    :param policies: a dictionary of policies.
    :param best_model_files: a list of paths to the model files to be loaded.
    :param script_folder: where to put the scripts for the cluster.
    :param repeats: number of tries for each policy.
    :param dur: length of session for each match.
    :return: list of file paths of results (to be parsed later).
    """
    if script_folder is None:
        script_folder = code_path + 'scripts/'
    result_files = []
    for r in range(repeats):
        for pid in policies:
            script_name = script_folder + pid + '.%i.minmax.bash' % r
            res_file = '%soutput/minmax_%s_depth%i_%i.res' % (code_path, pid, depth, r)
            out_file = '%soutput/minmax_%s_depth%i_%i.out' % (code_path, pid, depth, r)
            log_file = '%soutput/minmax_%s_depth%i_%i.log' % (code_path, pid, depth, r)
            model_folder = '%smodels' % code_path
            model_file = 'minmax_%s_depth%i_%i.pkl' % (pid, depth, r)
            result_files.append(res_file)
            pcom = ('python3 {p}Connect4.py '
                    '-A "{pid}(save_to={model_file},load_from={early_model});MinmaxAgent(depth={depth})" '
                    '-D {d} '
                    '-mf {model_folder} '
                    '-l {log_file} '
                    '-o {o} '
                    '-bi RandomBoard '
                    '-pat 0.1 '
                    '-t "test" ').format \
                (p=code_path,pid=pid,d=str(dur),o=res_file,model_file=model_file, depth=depth,
                 model_folder=model_folder, log_file=log_file, early_model=best_model_files[pid])
            with open(script_name, 'w') as script_file:
                script_file.write(script_string(pcom, out_file))
            out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
            jid = out[0].decode('utf').split(' ')[-1].strip()
            print("submitted job %s (%s) for Minmax test (%i) of policy %s" % (jid, script_name, r, pid))
    return result_files


def test_tournament(policies, best_model_files, script_folder=None, dur=int(100)):
    """
    test all of the policies against each other!
    :param policies: a dictionary of policies.
    :param best_model_files: a list of paths to the model files to be loaded.
    :param script_folder: where to put the scripts for the cluster.
    :param dur: length of session for each match.
    :return: list of file paths of results (to be parsed later).
    """
    if script_folder is None:
        script_folder = code_path + '/scripts/'
    result_files = []
    for i, subp in enumerate(combinations(policies, 2)):
        pid1 = subp[0]
        pid2 = subp[1]
        script_name = script_folder + '%i.tournament.bash' % i
        res_file = '%s/output/tournament_%i.res' % (code_path, i)
        out_file = '%s/output/tournament_%i.out' % (code_path, i)
        log_file = '%soutput/tournament_%i.log' % (code_path, i)
        model_folder = '%smodels' % code_path
        model_file1 = 'tournament_%s_%i.pkl' % (pid1, i)
        model_file2 = 'tournament_%s_%i.pkl' % (pid2, i)
        result_files.append(res_file)

        pcom = ('python3 {p}Connect4.py '
                '-A "{pid1}(save_to={model_file1},load_from={early_model1});{pid2}(save_to={model_file2},load_from={early_model2})" '
                '-D {d} '
                '-mf {model_folder} '
                '-l {log_file} '
                '-o {o} '
                '-bi RandomBoard '
                '-pat 0.1 '
                '-t "test" ').format \
            (p=code_path,pid1=pid1,pid2=pid2,d=str(dur),o=res_file,model_file1=model_file1,model_file2=model_file2,
             model_folder=model_folder, log_file=log_file, early_model1=best_model_files[pid1],early_model2=best_model_files[pid2])
        with open(script_name, 'w') as script_file:
            script_file.write(script_string(pcom, out_file))
        out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
        jid = out[0].decode('utf').split(' ')[-1].strip()
        print("submitted job %s (%s) for tournament test %i of policies %s" % (jid, script_name, i, ','.join(subp)))
    return result_files


def detect_problems_train(policies, script_folder=None, dur=int(100)):
    """
    run a quick selfplay test to see if any submissions have critical problems.
    :param policies: a dictionary of policies.
    :param script_folder: where to put the scripts for the cluster.
    :param dur: length of session for each match.
    :return: list of file paths of results (to be parsed later).
    """
    if script_folder is None:
        script_folder = code_path + '/scripts/'
    result_files = []
    for pid in policies:
        script_name = script_folder + pid + '.detect_problems.bash'
        res_file = '%soutput/detect_problems_train_%s.res' % (code_path, pid)
        out_file = '%soutput/detect_problems_train_%s.out' % (code_path, pid)
        log_file = '%soutput/detect_problems_train_%s.log' % (code_path, pid)
        model_folder = '%smodels' % code_path
        model_file1 = 'detect_problems_train_%s_1.pkl' % (pid)
        model_file2 = 'detect_problems_train_%s_2.pkl' % (pid)
        model_path1 = model_folder + '/' + model_file1
        result_files.append(res_file)
        pcom = ('python3 {p}Connect4.py '
                '-A "{pid}(save_to={model_file1},load_from={model_path1});{pid}(save_to={model_file2},load_from={model_path1})" '
                '-D {d} '
                '-mf {model_folder} '
                '-l {log_file} '
                '-o {o} '
                '-bi RandomBoard '
                '-sp 1 '
                '-plt 0.1 '
                '-spt {d2} '
                '-pat 0.007 ').format \
            (p=code_path,pid=pid,d=str(dur),d2=str(int(dur/2)),o=res_file,model_file1=model_file1,model_file2=model_file2,model_path1=model_path1,
             model_folder=model_folder, log_file=log_file)
        with open(script_name, 'w') as script_file:
            script_file.write(script_string(pcom, out_file,t='01:00:00'))
        out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
        jid = out[0].decode('utf').split(' ')[-1].strip()
        print("submitted job %s (%s) for detect_problems_train of policy %s" % (jid, script_name, pid))
    return result_files


def detect_problems_test(policies, script_folder=None, dur=int(100)):
    """
    run a quick match vs RandomAgent in test mode to see if any submissions
    have critical problems.
    :param policies: a dictionary of policies.
    :param script_folder: where to put the scripts for the cluster.
    :param dur: length of session for each match.
    :return: list of file paths of results (to be parsed later).
    """
    if script_folder is None:
        script_folder = code_path + '/scripts/'
    result_files = []
    for pid in policies:
        script_name = script_folder + pid + '.detect_problems.bash'
        res_file = '%soutput/detect_problems_test_%s.res' % (code_path, pid)
        out_file = '%soutput/detect_problems_test_%s.out' % (code_path, pid)
        log_file = '%soutput/detect_problems_test_%s.log' % (code_path, pid)
        model_folder = '%smodels' % code_path
        model_file1 = 'detect_problems_test_%s_1.pkl' % (pid)
        model_path1 = model_folder + '/' + model_file1
        result_files.append(res_file)
        pcom = ('python3 {p}Connect4.py '
                '-A "{pid}(save_to={model_file1},load_from={model_path1});RandomAgent())" '
                '-D {d} '
                '-mf {model_folder} '
                '-l {log_file} '
                '-o {o} '
                '-bi RandomBoard '
                '-pat 0.1 '
                '-t "test" ').format \
            (p=code_path,pid=pid,d=str(dur),o=res_file,model_file1=model_file1,model_path1=model_path1,
             model_folder=model_folder, log_file=log_file)
        with open(script_name, 'w') as script_file:
            script_file.write(script_string(pcom, out_file,t='01:00:00'))
        out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
        jid = out[0].decode('utf').split(' ')[-1].strip()
        print("submitted job %s (%s) for detect_problems_test of policy %s" % (jid, script_name, pid))
    return result_files


def selfplay(policies, script_folder=None, dur=int(100000)):
    """
    run all policies against themselves in selfplay and save the learned model.
    :param policies: a dictionary of policies.
    :param script_folder: where to put the scripts for the cluster.
    :param dur: length of session for each match.
    :return: list of file paths of results (to be parsed later).
    """
    if script_folder is None:
        script_folder = code_path + '/scripts/'
    result_files = []
    for pid in policies:
        script_name = script_folder + pid + '.selfplay.bash'
        res_file = '%soutput/selfplay_%s.res' % (code_path, pid)
        out_file = '%soutput/selfplay_%s.out' % (code_path, pid)
        log_file = '%soutput/selfplay_%s.log' % (code_path, pid)
        model_folder = '%smodels' % code_path
        model_file1 = 'selfplay_%s_1.pkl' % (pid)
        model_file2 = 'selfplay_%s_2.pkl' % (pid)
        model_path1 = model_folder + '/' + model_file1
        result_files.append(res_file)
        pcom = ('python3 {p}Connect4.py '
                '-A "{pid}(save_to={model_file1},load_from={model_path1});{pid}(save_to={model_file2},load_from={model_path1})" '
                '-D {d} '
                '-mf {model_folder} '
                '-l {log_file} '
                '-o {o} '
                '-bi RandomBoard '
                '-sp 1 '
                '-plt 0.1 '
                '-pat 0.007 ').format \
            (p=code_path,pid=pid,d=str(dur),o=res_file,model_file1=model_file1,model_file2=model_file2,model_path1=model_path1,
             model_folder=model_folder, log_file=log_file)
        with open(script_name, 'w') as script_file:
            script_file.write(script_string(pcom, out_file))
        out = sp.Popen(['sbatch', script_name], stdout=sp.PIPE).communicate()
        jid = out[0].decode('utf').split(' ')[-1].strip()
        print("submitted job %s (%s) for selfplay of policy %s" % (jid, script_name, pid))
    return result_files


def merge_results(result_files, type=''):
    """
    merge the results from all of the result files into a single file, where
    we ignore the agents which are not from the students.
    :param result_files: all of the result files
    :param type: the type of test we are merging (random / minmax / selfplay...)
    :return: a list of the result strings
    """
    res_files = [r for r in result_files] # just a copy
    all = []
    while res_files:
        r = res_files.pop()
        try:
            with open(r) as R:
                for res in csv.DictReader(R):
                    if res['policy'] in ['RandomAgent','MinmaxAgent']: continue
                    res['type'] = type
                    all.append(res)
        except IOError:
            res_files.append(r)
            time.sleep(5)
    return all


def get_best_model_file(results):
    best_map = {}
    for r in results:
        if r['policy'] not in best_map: best_map[r['policy']] = ('', -1)
        if best_map[r['policy']][1] < float(r['score']):
            best_map[r['policy']] = (r['model_file_path'], float(r['score']))
    return {k : v[0] for k,v in best_map.items()}


def update_scoreboard(results, sb_files, archive_file):
    """
    create a scoreboard.
    """
    if archive_file:
        with open(archive_file, 'wb') as A: pickle.dump(results, A)
    fns, reformatted, tcnt = set([]), {}, Counter()
    for r in results:
        tcnt[(r['policy'], r['type'])] += 1
        if r['policy'] not in reformatted: reformatted[r['policy']] = {}
        fn = '%s%i' % (r['type'], tcnt[(r['policy'], r['type'])])
        reformatted[r['policy']][fn] = float(r['score'])
        fns.add(fn)
    fns = sorted(fns)
    for p, scores in reformatted.items():
        scores['average'] = sum(scores.values()) / len(scores)
    fns = ['policy'] + fns + ['average']
    policy_order = sorted(reformatted, key=lambda x:reformatted[x]['average'], reverse=True)
    for sb_file in sb_files:
        with open(sb_file, 'w') as SB:
            wrtr = csv.DictWriter(SB, fieldnames=fns)
            wrtr.writeheader()
            for p in policy_order:
                scores = reformatted[p]
                scores['policy'] = p
                wrtr.writerow(scores)


def update_tournament_scoreboard(results, sb_files, archive_file):
    """
    create a csv file with the results of all of the games in the tournament
    """
    if archive_file:
        with open(archive_file, 'wb') as A: pickle.dump(results, A)

    games = {}
    for r in results:
        game_id = r["game_id"]
        if game_id not in games:
            games[game_id] = {}
        games[game_id][r["player_num"]] = [r["policy"], r["score"]]

    scores = {}
    for game_id, game_dict in games.items():
        scores[game_id] = {}
        scores[game_id]['Game ID'] = game_id
        scores[game_id]['Player 1'] = game_dict['1'][0]
        scores[game_id]['Player 1 Score'] = game_dict['1'][1]
        scores[game_id]['Player 2'] = game_dict['2'][0]
        scores[game_id]['Player 2 Score'] = game_dict['2'][1]

    for sb_file in sb_files:
        with open(sb_file, 'w') as SB:
            wrtr = csv.DictWriter(SB, fieldnames=['Game ID','Player 1','Player 1 Score', 'Player 2','Player 2 Score'])
            wrtr.writeheader()
            for game_id, score_row in scores.items():
                wrtr.writerow(score_row)


if __name__ == '__main__':

    print('start time: %s' % str(datetime.datetime.now()))
    pfile, sb_files, tour_sb_files, afile, dur = parse_input()
    policies = parse_policies(pfile)

    # run demo to detect problematic submissions, then review out/log files:
    detect_problems_train(policies, dur=DETECT_PROBLEMS_DURATION)
    detect_problems_test(policies, dur=DETECT_PROBLEMS_DURATION)
    print('finished demo stage at time: %s' % str(datetime.datetime.now()))

    # run selfplay to become a master of connect four:
    selfplay_results = merge_results(selfplay(policies, dur=SELFPLAY_DURATION), 'selfplay')
    print('finished selfplay stage at time: %s' % str(datetime.datetime.now()))
    best_model_files = get_best_model_file(selfplay_results)

    # run the test against the random agent:
    results = merge_results(test_vs_random(policies, best_model_files, repeats=1, dur=RANDOM_DURATION), 'random')
    print('finished random stage at time: %s' % str(datetime.datetime.now()))
    update_scoreboard(results, sb_files, afile)

    # run the test againt the Minmax agent with depth 1:
    intermediate_results = merge_results(test_vs_minmax(policies, best_model_files, repeats=1, dur=MINMAX_DURATION, depth=1), 'minmax1')
    results.extend(intermediate_results)
    update_scoreboard(results, sb_files, afile)
    intermediate_results = merge_results(test_vs_minmax(policies, best_model_files, repeats=1, dur=MINMAX_DURATION, depth=2), 'minmax2')
    results.extend(intermediate_results)
    update_scoreboard(results, sb_files, afile)
    intermediate_results = merge_results(test_vs_minmax(policies, best_model_files, repeats=1, dur=MINMAX_DURATION, depth=3), 'minmax3')
    results.extend(intermediate_results)
    update_scoreboard(results, sb_files, afile)
    intermediate_results = merge_results(test_vs_minmax(policies, best_model_files, repeats=1, dur=MINMAX_DURATION, depth=4), 'minmax4')
    results.extend(intermediate_results)
    update_scoreboard(results, sb_files, afile)
    print('finished minmax stage at time: %s' % str(datetime.datetime.now()))

    # run the tournament between all of the agents:
    results = merge_results(test_tournament(policies, best_model_files, dur=TOURNAMENT_DURATION), 'tournament')
    print('finished tournament at time: %s' % str(datetime.datetime.now()))
    update_tournament_scoreboard(results, tour_sb_files, afile)

