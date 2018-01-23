import subprocess
from matplotlib import pyplot as plt


def single_run(args, log_file):

    cmdline = ['python3', 'Connect4.py'] + args
    out = []
    losses = []  # tuples (round, loss)

    with subprocess.Popen(cmdline, stdout=subprocess.PIPE, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
            out.append(line)

            if 'loss' in line and 'ROUND' in line:
                line = line.strip('\n')
                line_list = line.split(' ')
                val = line_list[line_list.index('loss') + 1]
                round = line_list[line_list.index('\tROUND') + 1]
                if line_list[line_list.index('\tROUND') - 1] == '1':
                    losses.append((round, val))

        if p.stderr:
            for line in p.stderr:
                print(line, end='')

    games, wins1, wins2 = get_final_result(out)
    msg1 = ', '.join(args)
    msg = msg1 + '\n' + 'Player 1 wins {}, Player 2 wins {}, out of {} games.'.format(wins1, wins2, games) + '\n'
    final_log(msg, log_file)
    plot_loss(losses)


def plot_loss(loss_list):
    x = [int(l[0]) for l in loss_list]
    y = [float(l[1]) for l in loss_list]
    plt.figure()
    plt.scatter(x, y)
    plt.xticks(x)
    plt.savefig('plots/{}_loss.png'.format(log_file))


def get_final_result(output):
    output.reverse()
    num_games, player1_wins, player2_wins = 0, 0, 0

    for i, line in enumerate(output):
        if 'Player 1 wins' in line:
            if player1_wins == 0:
                l = line.split()
                if l[-1] == '':
                    player1_wins = l[-2]
                else:
                    player1_wins = l[-1]

        if 'Player 2 wins' in line:
            if player2_wins == 0:
                l = line.split()
                if l[-1] == '':
                    player2_wins = l[-2]
                else:
                    player2_wins = l[-1]

        if 'STATUS' in line:
            if num_games == 0:
                l = line.split()
                if l[-1] == '':
                    l = l[:-1]
                num_games = l[-2]
    return num_games, player1_wins, player2_wins


def final_log(content, path):
    with open('session_logs/{}'.format(path), 'a') as f:
        f.write(content)
    print("Logged to {}".format(path))


if __name__ == '__main__':

    test_rounds = 1000
    log_file = 'fifth_epsilon'

    final_log('Starting log...      test rounds {}    topology net #{}'.format(test_rounds, 6), log_file)

    # train against self
    train_num = 'first_train'
    rounds = 100000
    first_model_save = 'SmartSmart_{}'.format(log_file)
    smart1_args = 'save_to={}1'.format(first_model_save)
    smart2_args = 'epsilon=0.85, epsilon_decay=0.001, save_to={}2'.format(first_model_save)
    train1_args = ['-D={}'.format(rounds), '-A=SmartPolicy({});SmartPolicy({})'.format(smart1_args, smart2_args), '-bi=RandomBoard']
    single_run(train1_args, log_file)

    # test against random
    smart1_args = 'load_from=models/{}1'.format(first_model_save)
    test1_args = ['-D={}'.format(test_rounds), '-A=SmartPolicy({});RandomAgent()'.format(smart1_args), '-bi=RandomBoard',
                  '-t=test', '-l=logs/SmartRandomTest2.log']
    single_run(test1_args, log_file)

    # train some more
    train_num = 'second_train'
    rounds = 10000
    second_model_save = 'SmartSmart_{}_2'.format(log_file)
    smart1_args = 'epsilon=0.85, epsilon_decay=0.001, save_to={}1, load_from=models/{}1'.format(second_model_save,
                                                                                               first_model_save)
    smart2_args = 'save_to={}2, load_From=models/{}2'.format(second_model_save, first_model_save)
    train2_args = ['-D={}'.format(rounds), '-A=SmartPolicy({});SmartPolicy({})'.format(smart1_args, smart2_args), '-bi=RandomBoard']
    single_run(train2_args, log_file)

    # test again against random
    smart1_args = 'load_from=models/{}1'.format(first_model_save)
    test1_args = ['-D={}'.format(test_rounds), '-A=SmartPolicy({});RandomAgent()'.format(smart1_args), '-bi=RandomBoard',
                  '-t=test', '-l=logs/SmartRandomTest3.log']
    single_run(test1_args, log_file)

    # test against minmax
    smart1_args = 'load_from=models/{}1'.format(first_model_save)
    test2_args = ['-D={}'.format(test_rounds), '-A=SmartPolicy({});MinmaxAgent(depth=1)'.format(smart1_args), '-bi=RandomBoard',
                  '-t=test', '-l=logs/MinmaxDepth2.log']
    single_run(test2_args, log_file)
