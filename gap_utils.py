import numpy as np
import wandb
from scipy import stats

def get_margin(logits, target):
    logits = np.array(logits).copy()
    index = np.arange(len(logits))
    target_logits = logits[index, target]
    logits[index, target] = logits.min() - 10
    return target_logits - logits.max(1)

BASE_MARGIN_STD = 3.5

class Ens_gap_editor():
    def __init__(self, editor_type):

        if editor_type.find('norm_') == 0:
            self.__get_next_gap = getattr(self, editor_type.replace('norm_', ''))
            self.normalize_margin = True
        elif 'simple_gap_marginscale_' in editor_type:
            self._scale = float(editor_type.replace('simple_gap_marginscale_', ''))
            self.__get_next_gap = getattr(self, 'simple_gap_marginscale')
            self.normalize_margin = False
        else:
            self.__get_next_gap = getattr(self, editor_type)
            self.normalize_margin = False

    def __normalize_margin(self, m, num_model):
        if num_model == 0:
            self.__base_margin_std = m.std()
            return m
        return m * self.__base_margin_std / m.std()

    def get_next_gap(self, last_gap, logits, target, num_model):
        margin = get_margin(logits, target)
        if self.normalize_margin: margin = self.__normalize_margin(margin, num_model)

        val, logval = self.__get_next_gap(margin, last_gap, num_model, logits, target)
        if logval is not None:
            wandb.log({'mean_gap_size': logval})
        
        # val /= logits[np.arange(logits.shape[0]), target] # scale by true logit
        
        return val

    def save_gap(self, margin, last_gap, *args):
        return last_gap, None

    def simple_gap(self, margin, *args):
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin

    def simple_gap_center_by_mode(self, margin, *args):
        mean_margin = margin.mean()
        mode_margin = stats.mode(margin).mode
        return mode_margin - margin, mean_margin
    
    def simple_gap_marginscale(self, margin, *args):
        mean_margin = margin.mean()
        mode_margin = stats.mode(margin).mode
        margin = mode_margin - margin
        margin *= self._scale

        return margin, mean_margin
    
    def simple_gap_no_mean(self, margin, *args):
        mean_margin = margin.mean()
        return - margin, mean_margin
    
    def reverse_gap(self, margin, *args):
        mean_margin = margin.mean()
        return margin - mean_margin, mean_margin

    def reverse_gap_no_mean(self, margin, *args):
        mean_margin = margin.mean()
        return margin, mean_margin

    def cummulative_gap(self, margin, last_gap, *args):
        margin += last_gap
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin
    
    def cummulative_gap_no_mean(self, margin, last_gap, *args):
        margin += last_gap
        mean_margin = margin.mean()
        return - margin, mean_margin

    def mean_cummulative_gap(self, margin, last_gap, num_model, *args):
        # One model in ensemble means that num_model = 0
        margin = (last_gap * (num_model + 1)  + margin) / (num_model + 2)
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin

    def zeropos_no_mean(self, margin, last_gap, num_model):
        margin = np.clip(margin, None, 0)
        mean_margin = margin.mean()
        return - margin, mean_margin
    
    def zeropos(self, margin, last_gap, num_model):
        margin = np.clip(margin, None, 0)
        mean_margin = margin.mean()
        return mean_margin - margin, mean_margin
    
    def zeroneg_no_mean(self, margin, last_gap, num_model):
        margin = np.clip(margin, 0, None)
        mean_margin = margin.mean()
        return - margin, mean_margin
    
    def zeroneg(self, margin, last_gap, num_model):
        margin = np.clip(margin, 0, None)
        mean_margin = margin.mean()
        return mean_margin  - margin, mean_margin
    
    def simple_gap_all_logits(self, margin, last_gap, num_model, logits, target):
        mean_margin = margin.mean()
        out = logits - logits[np.arange(logits.shape[0]), target].reshape(-1, 1)
        out[np.arange(out.shape[0]), target] -= mean_margin

        return out.mean(1, keepdims=True) - out, mean_margin


# del loaders['train'].dataset.gap_size
# predictions_logits, targets = utils.predictions(loaders['train'], model, device)
# loaders['train'].dataset.gap_size = mean_gap_size - gap_size