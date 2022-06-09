from robust_speech.adversarial.utils import TargetGenerator
from speechbrain.utils.metric_stats import MetricStats
import torch


class BriefMetric(MetricStats):
    def summarize(self, field=None):
        """Summarize the metric scores, returning relevant stats.

        Arguments
        ---------
        field : str
            If provided, only returns selected statistic. If not,
            returns all computed statistics.

        Returns
        -------
        float or dict
            Returns a float if ``field`` is provided, otherwise
            returns a dictionary containing all computed stats.
        """
        min_index = torch.argmin(torch.tensor(self.scores))
        max_index = torch.argmax(torch.tensor(self.scores))
        self.summary = {
            "error_rate": float(sum(self.scores) / len(self.scores)),
            "min_score": float(self.scores[min_index]),
            "min_id": self.ids[min_index],
            "max_score": float(self.scores[max_index]),
            "max_id": self.ids[max_index],
        }

        if field is not None:
            return self.summary[field]
        else:
            return self.summary


class PrefixAccuracyMetric:
    def __init__(self, prefix, space=True):
        def metric(predictions, target, ** kwargs):
            res = []
            for pred in predictions:
                if (space and pred[0] == prefix) or ((not space) and pred[0].startswith(prefix)):
                    res.append(1)
                else:
                    res.append(0)
            return torch.tensor(res)
        self.custom_metric = metric

    def __call__(self, *args, **kwargs):
        return self.custom_metric(*args, **kwargs)


class SuffixAccuracyMetric:
    def __init__(self, suffix, space=True):
        def metric(predictions, target, ** kwargs):
            res = []
            for pred in predictions:
                if (space and pred[-1] == suffix) or ((not space) and pred[-1].endswith(suffix)):
                    res.append(1)
                else:
                    res.append(0)
            return torch.tensor(res)
        self.custom_metric = metric

    def __call__(self, *args, **kwargs):
        return self.custom_metric(*args, **kwargs)


class PrefixTarget(TargetGenerator):
    def __init__(self, prefix, space=True):
        self.prefix = prefix
        self.space = space

    def generate_targets(self, batch, hparams):
        return self.prefix+(" " if self.space else "")+batch.wrd[0]


class SuffixTarget(TargetGenerator):
    def __init__(self, suffix, space=True):
        self.suffix = suffix
        self.space = space

    def generate_targets(self, batch, hparams):
        newwrd = batch.wrd[0]+(" " if self.space else "")+self.suffix
        return newwrd
