import numpy as np
import pandas as pd
import seaborn as sns

from .srs import Schedule, Card
from .mystery.mystery import MysteryLearner

def _sample_initial_rating(seed=None) -> int:
    rng = np.random.default_rng(seed) if seed is not None else None
    if rng is None:
        return int(np.random.randint(1, 5))
    return int(rng.integers(1, 5))


def simulate(Schedule: Schedule, seed=None, **kwargs):
    card = Card.sample(seed=seed)
    schedule = Schedule(**kwargs)
    learner = MysteryLearner(card, seed=seed)
    # initial_rating = Categorical(tensor([0.25, 0.25, 0.25, 0.25])).sample().item() + 1
    initial_rating = _sample_initial_rating(seed=seed)
    learner.initialize(initial_rating)
    schedule.initialize(initial_rating)

    data = []
    last = 0
    for t in range(1, 365+1):
        dt = t - last
        recall_prob = learner.recall_prob(dt)
        should_review = schedule.interval() == dt
        if should_review:
            rating = learner.review(dt)            
            schedule.update(rating, dt)
            last = t
        else:
            rating = None
        data.append({
            "time": t,
            "recall_prob": recall_prob,
            "review": should_review,
            "rating": rating
        })

    return pd.DataFrame(data)

def cheating_simulate(Schedule: Schedule, seed=None):
    card = Card.sample(seed=seed)
    learner = MysteryLearner(card, seed=seed)
    schedule = Schedule(MysteryLearner(card, seed=seed))
    #initial_rating = Categorical(tensor([0.25, 0.25, 0.25, 0.25])).sample().item() + 1
    initial_rating = _sample_initial_rating(seed=seed)
    learner.initialize(initial_rating)
    schedule.initialize(initial_rating)

    data = []
    last = 0
    for t in range(1, 365+1):
        dt = t - last
        recall_prob = learner.recall_prob(dt)
        should_review = schedule.interval() == dt
        if should_review:
            rating = learner.review(dt)            
            schedule.update(rating, dt)
            last = t
        else:
            rating = None
        data.append({
            "time": t,
            "recall_prob": recall_prob,
            "review": should_review,
            "rating": rating
        })

    return pd.DataFrame(data)

def simulate_many(Schedule: Schedule, N: int, seed=None, **kwargs):
    rows = []
    for i in range(N):
        data = simulate(Schedule, seed=seed, **kwargs)
        data["trace"] = i
        rows.append(data)
    return pd.concat(rows)

def cheating_simulate_many(Schedule: Schedule, N: int, seed=None):
    rows = []
    for i in range(N):
        data = cheating_simulate(Schedule, seed=seed)
        data["trace"] = i
        rows.append(data)
    return pd.concat(rows)

def eval_metrics(data, schedule, t_eval=100):
    mean_recall = data["recall_prob"].mean()
    num_reviews = len(data[data["review"]]) / data["trace"].max()
    exam_recall = data.loc[data["time"] == t_eval, "recall_prob"].mean()
    agg_score = min(max(mean_recall - 0.95, 0) * max(14 - num_reviews, 0) * 20, 1.0) * 40

    return pd.DataFrame([{
        "schedule": schedule,
        "avg_recall": mean_recall,
        "num_reviews": num_reviews,
        "score": agg_score,
        "exam_recall": exam_recall,        
    }])

def scatterplot_annot(data, x, y, label, **kwargs):
    ax = sns.scatterplot(data, x=x, y=y, **kwargs)

    for _, row in data.iterrows():
        ax.annotate(row[label], (row[x], row[y]), 
            textcoords='offset points', xytext=(5, 0))