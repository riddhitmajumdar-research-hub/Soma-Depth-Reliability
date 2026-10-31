# Lane 6: window offset and the beat anchor

**Question.** Which feature of the synthetic windows can a classifier use without any signal shape at all?

**Methodology note.** [08](../../docs/methodology/08-lane6-beat-offset-mechanism.md).

**Contents.** `FACTS.md` and `FACTS_6c.md` hold the answers. Scripts `q1_q2_q3_demo.py`, `q7_offset_bayes.py`, `q8_beat_index_accuracy.py`, `q9_seed121_beat_index_sweep.py` with their stdout logs. The `_6c` versions of Q9 redid the earlier Q8 and Q9 scripts under a checkpoints-only rule.

`q9_seed121_beat_index_sweep_6c.py` reads Lane 5 weights and logs from [../lane5/recall_floor_main_run](../lane5/recall_floor_main_run/). Those weights are not in git.
