from datetime import time
import random

from verify.verify import *
from verify.util import *
import os
import sys
import pandas as pd
# model params


BATCH_SIZE = 1000


#########################################


class QDModel:

    def __init__(self, is_male):
        self.is_male = is_male
        self.num_samples = 0
        # self.df_mys = pd.read_csv("main/final_labels.csv", sep=',', encoding='utf-8', dtype = str, keep_default_na=False, skip_blank_lines=False, converters={'body':lambda x:x.replace('/n','')})
        self.pop_model_other = pd.read_csv('python\\verifair\\main\\pop_model_2 (1).csv', engine='python')
        # all the comments with male and toxic
        self.male_df = self.pop_model_other.loc[(self.pop_model_other['male'] > 0.5) & (self.pop_model_other['target'] == True)]
        # all the comments with femal and toxic
        self.female_df = self.pop_model_other.loc[(self.pop_model_other['female'] > 0.5) & (self.pop_model_other['target'] == True)]

    def sample(self, n_samples):
        # time1 = time.time()
        k = random.randint(0, 1) # decide on k once
        ret = []
        # sample hate comment for gender
        while True:
           
          n = random.randint(0, len(self.pop_model_other.head()))
          label = self.pop_model_other.loc[self.pop_model_other.index[n], 'pred_log']
          if not self.is_male:
             n = random.randint(0, len(self.male_df.head()))
             label = self.pop_model_other.loc[self.male_df.index[n], 'pred_log']
             ret.append((int(False == label)))
             break
          else: 
             n = random.randint(0, len(self.female_df.head()))
             label = self.female_df.loc[self.female_df.index[n], 'pred_log']
             ret.append((int(False == label)))
             break

        # X = X[:n_samples]
        # time2 = time.time()
        # Y = predict(self.model, X, self.class_idx)
        # time3 = time.time()
        # print('Sample time: '+str(time2-time1))
        # print('Prediction time: '+str(time3-time2))
        # ret = []
        # for v in Y:
        #     pred = v['predictions']
        #     ret.append(int(pred == self.class_idx))
        # self.num_samples += n_samples
        return ret


def main(is_cat = True, c = 0.15, Delta = 0.005, delta = 0.5*1e-5, n_max = 100000, is_causal = False, log_iters = 10):

    log('Verifying fairness' , INFO)

    # Step 1: Load discriminative model
    log('Loading discriminative model...', INFO)
    # dis_model = load_dis_model(dis_model_path)
    log('Done!', INFO)

    # Step 2: Load generative models
    log('Loading sketch-rnn models'' as generative models', INFO)

    # Test generative model
    if False:
        test_gen_model(us_model, us_sess, prefix='us')
        test_gen_model(non_model, non_sess, prefix='nonus')
    # End

    log('Done!', INFO)

    if False: #test accuracy of the classifier
        X = sample_drawing(us_model, us_sess, 1)
        dis_model.evaluate(input_fn = get_input_fn(X,8,class_idx), steps = 1)
        exit(0)

    # # Step 4: Build model
    model_male = QDModel(is_male=True)
    model_female = QDModel(is_male=False)

    # runtime = time.time()

    # # Step 3: Run fairness
    result = verify(model_male, model_female, c, Delta, delta, 1, n_max, is_causal, log_iters)

    if result is None:
        log('RESULT: Failed to converge!', INFO)
        return

    # Step 3: Post processing
    is_fair, is_ambiguous, n_successful_samples, E = result
    # runtime = time.time() - runtime
    # n_total_samples = model_nonus.num_samples + model_us.num_samples

    log('RESULT: Pr[fair = {}] >= 1.0 - {}'.format(is_fair, 2.0 * delta), INFO)
    log('RESULT: E[ratio] = {}'.format(E), INFO)
    log('RESULT: Is fair: {}'.format(is_fair), INFO)
    log('RESULT: Is ambiguous: {}'.format(is_ambiguous), INFO)
    # log('RESULT: Successful samples: {} successful samples, Attempted samples: {}'.format(n_successful_samples, n_total_samples), INFO)
    # log('RESULT: Running time: {} seconds'.format(runtime), INFO)


if __name__ == '__main__':
    # c, Delta, delta

    baseline = [[0.2, 0, 0.5*1e-5]]

    vary_delta = []

    # for i in range(10):
    #     cur = baseline[0][:]
    #     cur[2] = 0.5*pow(10,-i-1)
    #     vary_delta.append(cur)

    vary_Delta = []
    
    # for i in range(5):
    #     cur = baseline[0][:]
    #     cur[0] = 0.4
    #     cur[1] = 5 * pow(10, -i-1)
    #     vary_Delta.append(cur)

    vary_c = []

    # for i in range(5):
    #     cur = baseline[0][:]
    #     cur[0] = 0.4 - 0.05 + i*0.01
    #     vary_c.append(cur)
    #     cur = baseline[0][:]
    #     cur[0] = 0.4 + 0.05 - i*0.01
    #     vary_c.append(cur)

    # vary_c.append([0.4, 0.1, 0.5*1e-5])


    settings = (
            ('baseline', baseline),
                ('vary_delta', vary_delta),
                ('vary_c', vary_c),
                # ('vary_Delta', vary_Delta)
                )

    for name, settings in settings:
        log('RESULT: running experiment '+name, INFO)
        for s in settings:
            log('RESULT: parameters delta: {}, Delta: {}, c: {}'.format(s[2]*2, s[1], s[0]), INFO)
            main(is_cat = False, c = s[0], Delta = s[1], delta = s[2], n_max = 500000, is_causal = False, log_iters = 10)
            log('\n', INFO)
            sys.stdout.flush()
        log('\n', INFO)
