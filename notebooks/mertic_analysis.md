As we saw in EDA, the distribution of `u_in` is close to normal, so MAE is a reasonable baseline metric — but it has several weaknesses.

First, MAE only tells us that the model made a mistake, not what kind. For example, one large deviation (true pressure is high, prediction is low) and several small deviations of the same total size look identical to MAE — it cannot distinguish a single big miss from many small ones. To catch this, we also calculate RMSE, which penalizes large deviations more heavily than small ones.

Second, MAE does not show the direction of the error — whether the model is underestimating or overestimating pressure. To catch this shift, we calculate ME.

To compare models on overall performance, we also compute R², to check whether there is a meaningful improvement over a dummy baseline.

All of these metrics must be calculated using the same rule as the competition scoring: only on rows where `u_out == 0`.