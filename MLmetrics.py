import numpy as np
from collections import defaultdict
import itertools

''' accuracy metrics '''

def MAE(predictions):
    __, __, true_r, predict_r = list(zip(*predictions))
    return np.mean(abs(np.array(true_r) - np.array(predict_r)))

def RMSE(predictions):
    __,__,true_r,predict_r = list(zip(*predictions))
    return np.sqrt(np.mean(np.square(np.array(true_r)-np.array(predict_r))))

''' beyond metrics - to be used with leave one out test/anti test set predictions, anti test set topN'''

def HitRate(topN,predictions):
    total = 0
    hits = 0
    for pred in predictions:
        lo_uid = pred[0]
        lo_iid = pred[1]
        hit = False
        for iid,predictRating in topN[int(lo_uid)]:
            if int(lo_iid) == int(iid):
                # print('--item hit:',lo_iid,',user:',lo_uid)
                hit = True
                break
        if hit:
            hits += 1
        total += 1
    # print('hits:', hits, ',', lo_uid, lo_iid)
    return hits/total

def CumulativeHitRate(topN,predictions,ratingCutoff = 0):
    total = 0
    hits = 0
    for pred in predictions:
        lo_uid = pred[0]
        lo_iid = pred[1]
        actual_rating = pred[2]
        if actual_rating >= ratingCutoff:
            hit = False
            for iid,predictRating in topN[int(lo_uid)]:
                if int(lo_iid) == int(iid):
                    # print('--item hit:', lo_iid, ',user:', lo_uid)
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1
    # print('hits:', hits, ',', lo_uid, lo_iid)
    return hits/total

def ARHR(topN,predictions):
    total = 0
    hits = 0
    for pred in predictions:
        hit_rank = 0
        rank = 0
        lo_uid = pred[0]
        lo_iid = pred[1]
        for iid,predictRating in topN[int(lo_uid)]:
            rank += 1
            if int(lo_iid) == int(iid):
                # print('--item hit:', lo_iid, ',user:', lo_uid,', rank:',rank)
                hit_rank = rank
                break
        if hit_rank>0:
            hits += 1.0 / int(hit_rank)
        total += 1
    return hits/total

def RatingHitRate(topN,predictions):
    total = defaultdict(float)
    hits = defaultdict(float)
    for pred in predictions:
        lo_uid = pred[0]
        lo_iid = pred[1]
        actual_rating = pred[2]
        hit = False
        for iid,predictRating in topN[int(lo_uid)]:
            if int(lo_iid) == int(iid):
                hit = True
                break
        if hit:
            hits[actual_rating] += 1
        total[actual_rating] += 1
    # print('hits:', hits, ',', lo_uid, lo_iid)
    for rating in sorted(hits.keys()):
        print(rating, hits[rating]/total[rating])

''' beyond metrics: to be used with full anti test set evaluations'''

def Spread(topN, predictions):
    P = defaultdict(float)
    total = sum([len(i) for i in topN.values()])
    for pred in predictions:
        pred_uid,pred_iid = pred[0],pred[1]
        for iid, predictRating in topN[int(pred_uid)]:
            if int(iid) == int(pred_iid):
                P[int(iid)] += 1.0 / total
    return -1.0 * sum([p * np.log(p) for p in P.values()])

def Coverage(predictions,testset):
    return len(predictions) / len(testset)

def Diversity(topN,simsAlgo):
    n = 0
    total = 0
    simMat = simsAlgo.compute_similarities()
    for userId in topN.keys():
        pairs = itertools.combinations(topN[userId],2)
        for pair in pairs:
            item1 = pair[0][0]
            item2 = pair[1][0]
            iid1 = simsAlgo.trainset.to_inner_iid(str(item1))
            iid2 = simsAlgo.trainset.to_inner_iid(str(item2))
            total += simMat[iid1][iid2]
            n += 1
    return 1 - (total / n)

def Novelty(topN,rankings):
    n = 0
    total = 0
    for uid in topN.keys():
        for iid,_ in topN[uid]:
            rank = rankings[iid]
            total += rank
            n += 1
    return total / n

