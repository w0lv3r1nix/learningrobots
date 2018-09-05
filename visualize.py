import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
import pickle, os, argparse
import seaborn as sns
sns.set()
sns.set_context('talk')

# parse command line arguments
# example command: python3 visualize.py -n DRQN EffDRQN -m final_distance -div conf
parser = argparse.ArgumentParser(description='Train deepRL agent for cam-shift toy-problem.')
parser.add_argument('-n', metavar='networks', type=str, nargs='+', help='networks' )
parser.add_argument('-m', metavar='metrics', type=str, nargs='+', help='metrics to plot')
parser.add_argument('-div', metavar='diverse', type=str, nargs='*', help='whether to filter bad episodes (lost target), display confidence band, ')
args = parser.parse_args()
networks, metrics, div = args.n, args.m, args.div

def smooth(y, window_size=50):
    """
    Returns a smoothed series of measurements y.
    Smoothing by mean of sliding window of size window_size*2+1.
    """
    y = np.array(y).astype(float)
    y_smooth = np.zeros_like(y)
    y_pad = np.pad(y,window_size,'reflect')
    for i in range(len(y_smooth)):
        # take window_size values before and after datapoint into account
        window = y_pad[i:(i+2*window_size+1)]
        # drop nans
        window = window[~np.isnan(window)]

        y_smooth[i] = np.mean(window)
    return y_smooth

def mean_confidence_interval(data, confidence=0.95):
    """ Returns mean and confidence band of several measurement-series. """
    m, cb = np.zeros(np.shape(data)[1]),np.zeros(np.shape(data)[1])
    a = 1.0*np.array(data).astype(float)
    for i in range(np.shape(a)[1]):
        n = len(a[:,i])
        tmp = a[:,i].copy()
        tmp = tmp[~np.isnan(tmp)]
        mean, se = np.mean(tmp), scipy.stats.sem(tmp)
        h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
        m[i] = mean
        cb[i] = h
    return m, cb

plt.figure()
# go through each metric, network-architecture and run
for metric in metrics:
    for net in networks:
        l = os.listdir('./logs/%sAgent/' % (net))
        var,lengths,max = [],[],[]
        for run in l:
            if os.path.isfile('./logs/%sAgent/%s/log.pkl' % (net,run)):
                with open('./logs/%sAgent/%s/log.pkl' % (net,run),'rb') as f:
                    run_dict = pickle.load(f)

                    run_data = np.empty(int(run_dict['n_epochs']))
                    run_data[:] = np.nan

                    # handle/calculate specific metrics
                    if metric == 'hit_miss_ratio':
                        tmp = np.array(run_dict['hits_on_target']).astype(float) / (np.array(run_dict['hits_on_target']).astype(float) + np.array(run_dict['hits_on_last_target']).astype(float))
                    elif metric == 'steps':
                        tmp = np.array(run_dict[metric]).astype(float)
                        tmp = tmp/run_dict['max_steps']
                    elif metric == 'mean_reward_per_step':
                        tmp = np.array(run_dict['cumulative_reward']).astype(float)
                        tmp = tmp/run_dict['n_steps']
                    else:
                        tmp = np.array(run_dict[metric]).astype(float)

                    run_data[0:len(tmp)] = tmp
                    var.append(run_data)
                    lengths.append(len(tmp))
                    max.append(np.max(tmp))
                    print('Loaded %s - %s' %(net,run))

        m, cf = mean_confidence_interval(var)
        m = smooth(m[:np.max(lengths)])
        e = smooth(cf[:np.max(lengths)]/2)
        p = plt.plot(np.arange(0,len(m),1),m,label='%s - %s' % (net, metric))
        if 'conf' in div:
            plt.fill_between(np.arange(0,len(m),1),m-e,m+e,alpha=0.2, color = p[0].get_color())
        elif 'per_run' in div:
            for r in range(np.shape(var)[0]):
                tmp = var[r][:lengths[r]]
                plt.plot(np.arange(0,len(tmp),1),smooth(tmp),alpha=0.2, color = p[0].get_color(),linewidth=1)
        print('%s - Mean max: %.2f' %(net,np.mean(max)))

plt.plot(np.arange(0,plt.xlim()[1],1),np.zeros_like(np.arange(0,plt.xlim()[1],1)),'k--',linewidth=1)
plt.xlabel('Episode')
plt.ylabel('Hit-miss-ratio')
# plt.xlim([-50,10050])
# plt.legend(loc=1)
# plt.title("Final (euclidean) distance from agent to target position.")
plt.show()
