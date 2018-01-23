import subprocess


def single_run(args):

    cmdline = ['python3', 'Connect4.py'] + args
    p = subprocess.Popen(cmdline, stdout=subprocess.PIPE)
    out, err = p.communicate()
    if out:
        out = out.decode()
    if err:
        err = err.decode()
    output = out.split('\n')
    for line in output:
        print(line)
    print('Errors: {}'.format(err))


if __name__ == '__main__':

    # train against self
    rounds = '1000'
    first_model_save = 'SmartSmartD{}'.format(rounds)
    smart1_args = 'save_to={}1'.format(first_model_save)
    smart2_args = 'save_to={}2'.format(first_model_save)
    train1_args = ['-D={}'.format(rounds), '-A=SmartPolicy({});SmartPolicy({})'.format(smart1_args, smart2_args), '-bi=RandomBoard']
    single_run(train1_args)

    # test against random
    smart1_args = 'load_from={}1'.format(first_model_save)
    test1_args = ['-D={}'.format(rounds), '-A=SmartPolicy({});RandomAgent()'.format(smart1_args), '-bi=RandomBoard', '-t=test']
    single_run(test1_args)

    rounds = '20000'
    second_model_save = 'SmartSmartD'.format(rounds)
    smart1_args = 'save_to={}1'.format(second_model_save)
    smart2_args = 'save_to={}2'.format(second_model_save)
    train2_args = ['-D={}'.format(rounds), '-A=SmartPolicy({});SmartPolicy({})'.format(smart1_args, smart2_args), '-bi=RandomBoard']
    single_run(train2_args)

    # test against minmax
    smart1_args = 'load_from={}1'.format(second_model_save)
    test2_args = ['-D={}'.format(rounds), '-A=SmartPolicy({});MinmaxAgent(depth=1)'.format(smart1_args), '-bi=RandomBoard', '-t=test']
    single_run(test2_args)