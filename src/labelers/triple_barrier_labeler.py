
import pandas as pd
import numpy as np
from src.labelers.base_labeler import BaseLabeler

class TripleBarrierLabeler(BaseLabeler):
    def __init__(self, pt_sl=(2, 2), vertical_barrier_days=5):
        self.pt_sl = pt_sl
        self.vertical_barrier_days = vertical_barrier_days

    def apply_pt_sl_on_t1(self, close, events):
        out = events[['t1']].copy(deep=True)

        if self.pt_sl[0] > 0:
            pt = self.pt_sl[0] * events['trgt']
        else:
            pt = pd.Series(index=events.index)

        if self.pt_sl[1] > 0:
            sl = self.pt_sl[1] * events['trgt']
        else:
            sl = pd.Series(index=events.index)

        for loc, t1 in events['t1'].fillna(close.index[-1]).items():
            df0 = close[loc:t1]
            df0 = df0 / close[loc] - 1
            if events.at[loc, 'side'] == 1:
                out.at[loc, 'sl'] = df0[df0 < -sl[loc]].index.min()
                out.at[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
            else:
                out.at[loc, 'sl'] = df0[df0 > sl[loc]].index.min()
                out.at[loc, 'pt'] = df0[df0 < -pt[loc]].index.min()
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
                labels[idx] = 0
        return labels

    def label(self, df: pd.DataFrame) -> pd.Series:
        close = df['close']
        t1 = close.index.to_series().shift(-self.vertical_barrier_days)
        trgt = close.pct_change(self.vertical_barrier_days).abs()
        side = pd.Series(1, index=close.index)

        events = pd.DataFrame({
            't1': t1,
            'trgt': trgt,
            'side': side
        }).dropna()

        barrier_touch_times = self.apply_pt_sl_on_t1(close, events)
        labels = self.get_labels(barrier_touch_times)
        return labels