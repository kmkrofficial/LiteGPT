# Training

**Status: building blocks only.** Design: plan sections 2-5 and spec `training.*` (WSD schedule, pre-registered extension gate,
SFT, reasoning, DPO). Implemented so far: `litegpt/checkpoint.py` (atomic, resumable checkpoints with a bit-exact
resume test) and `litegpt/records.py` (run records). Training loops arrive in M2/M3.
