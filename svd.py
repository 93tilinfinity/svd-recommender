'''
Everyone wants to be a beast until it's time to do what beasts do.
'''

from surprise import SVD,NormalPredictor
from surprise.model_selection import GridSearchCV
from MLutils import MovieLens
from EvaluatorScript import Evaluator

def loadMovieLensData():
    ml = MovieLens()
    data = ml.loadData()
    rankings = ml.getPopularityRanking()
    return ml,data,rankings

ml, data, rankings = loadMovieLensData()

# Build evaluation object
evaluator = Evaluator(data,rankings)

# GridSearch SVD Tuning
param_grid = {'n_factors': [50,100,150],'n_epochs':[25,50,75],'lr_all': [0.005,0.01],'reg_all':[0.02,0.1,0.5]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse','mae'], cv=3)
gs.fit(data)

# Build tuned SVD, untuned SVD, random models
params = gs.best_params['rmse']
svdtuned = SVD(reg_all=params['reg_all'],n_factors=params['n_factors'],n_epochs=params['n_epochs'],lr_all=params['lr_all'])
svd = SVD()
random = NormalPredictor()

# Add models to evaluation object
evaluator.addModel(svdtuned,'SVDtuned')
evaluator.addModel(svd,'SVD')
evaluator.addModel(random,'Random')

# Evaluate object = fit models, build topN lists, run prediction/hitrate based metrics
evaluator.evaluateModel(True)

# Build topN list for target user 56
evaluator.sampleUser(ml)
