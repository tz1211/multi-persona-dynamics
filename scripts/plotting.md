# Single eval dir (unchanged)
python scripts/plot_projection_eval.py data/critical_projection_eval --save critical.png

# Sequential: base A + A_B (offset checkpoints, personas filtered to A_B)
python scripts/plot_projection_eval.py data/critical_projection_eval \
  --sequential-dir data/critical_pessimistic_projection_eval \
  --save crit_pess.png

# Combined plot variant
python scripts/plot_projection_eval.py data/critical_projection_eval \
  --sequential-dir data/critical_pessimistic_projection_eval \
  --combine --save crit_pess_combined.png

Added a finetune-switch marker for sequential runs and returned offset metadata:

summarize_projection_eval now returns (df, offset) and tags rows with stage (“base” or “sequential”). For sequential runs, checkpoints are still offset by the base max.
plot_projection_eval accepts the returned offset and, when --sequential-dir is used, draws a vertical dash-dot line at the transition checkpoint with a small label (“finetune switch”) on both plots. This applies to combined and split plots.
CLI unchanged except for the existing --sequential-dir flag; return type remains (summary_df, axes_list).

python scripts/plot_projection_eval.py data/critical_projection_eval \
  --sequential-dir data/critical_pessimistic_projection_eval \
  --save crit_pess.png        # writes _scores/_finetuning with markers at switch
