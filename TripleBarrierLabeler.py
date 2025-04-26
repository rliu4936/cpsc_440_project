import pandas as pd
import numpy as np

class TripleBarrierLabeler:
    def __init__(self, close):
        self.close = close  # Series of prices

    def apply_pt_sl_on_t1(self, events, pt_sl, molecule):
        # Subset of events
        events = events.loc[molecule]
        out = events[['t1']].copy(deep=True)

        # Apply profit-taking and stop-loss thresholds
        if pt_sl[0] > 0:
            pt = pt_sl[0] * events['trgt']
        else:
            pt = pd.Series(index=events.index)

        if pt_sl[1] > 0:
            sl = pt_sl[1] * events['trgt']
        else:
            sl = pd.Series(index=events.index)

        for loc, t1 in events['t1'].fillna(self.close.index[-1]).items():
            df0 = self.close[loc:t1]  # Path prices
            df0 = df0 / self.close[loc] - 1  # Returns
            if events.at[loc, 'side'] == 1:
                out.at[loc, 'sl'] = df0[df0 < -sl[loc]].index.min()  # Stop-loss
                out.at[loc, 'pt'] = df0[df0 > pt[loc]].index.min()   # Profit-taking
            else:
                out.at[loc, 'sl'] = df0[df0 > sl[loc]].index.min()   # Stop-loss (short)
                out.at[loc, 'pt'] = df0[df0 < -pt[loc]].index.min()  # Profit-taking (short)

        return out

    def get_labels(self, barrier_touch_times):
        labels = pd.Series(index=barrier_touch_times.index, dtype='int8')

        for idx in barrier_touch_times.index:
            pt_touch = barrier_touch_times.at[idx, 'pt']
            sl_touch = barrier_touch_times.at[idx, 'sl']
            t1 = barrier_touch_times.at[idx, 't1']

            first_touch = min(
                [t for t in [pt_touch, sl_touch, t1] if pd.notna(t)],
                default=pd.NaT
            )

            if pd.isna(first_touch):
                labels[idx] = 0
            elif first_touch == pt_touch:
                labels[idx] = 1
            elif first_touch == sl_touch:
                labels[idx] = -1
            else:
                labels[idx] = 0  # vertical barrier hit first

        return labels
