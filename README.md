# Seoul Bike Sharing Demand

Predicting how many bikes will be rented each hour in Seoul, based on the time and the weather.

## The problem

Seoul runs a public bike rental system across the city. Bikes need to be available when and where people want them. Too few, and customers can't rent one. Too many, and bikes sit unused. Knowing how many bikes will be needed each hour helps the city plan how many to put out and when to move them between stations.

**Main question:** Can we predict how many bikes will be rented in a given hour, based on the time and the weather?

## Data

- Source: [Seoul Bike Sharing Demand Prediction](https://www.kaggle.com/datasets/saurabhshahane/seoul-bike-sharing-demand-prediction) (Kaggle).
- 8,760 rows, one per hour, from December 2017 to November 2018. No missing values.
- Time: date, hour, season and holiday.
- Weather: temperature, humidity, wind speed, visibility, dew point, solar radiation, rainfall and snowfall.
- Target: `Rented Bike Count`, the number of bikes rented in that hour. On average 729 bikes per hour (median 542, busiest hour 3,556).

This is a regression problem: we predict a number, not a category.

## What the notebook does

1. **Data check.** Looks at types, missing values and the text columns.
2. **Closed hours removed.** On 295 hours the system wasn't working and no bikes were rented. These rows are removed, which leaves 8,465. The `Functioning Day` column is then dropped, because it's always "Yes".
3. **Date features.** Adds day, month and day of the week from the date. `Year` is dropped, because the data covers only one year.
4. **Correlation check.** Dew point is almost the same as temperature, so it's dropped (it has the weaker link with the target).
5. **Encoding.** `Holiday` becomes 0/1. `Seasons` becomes dummy columns, with Autumn as the baseline (`drop_first=True`), to avoid the dummy trap.
6. **Exploration.** Total rentals by month, day of the month, hour and date. Demand peaks around 8am and 6pm, and is lowest in winter.
7. **Models.** Linear Regression, all on the same random 80/20 split:
   - hour as a number;
   - hour as dummy columns (midnight as the baseline);
   - hour dummies + temperature²;
   - hour dummies + temperature³.
8. **Temperature curves.** Compares a straight line with curves (temperature² up to temperature⁵), using temperature alone, to see whether the link with demand bends.

## Results (test set)

Baseline: a model that always predicts the average (729 bikes) has an R² of 0.

| Model | R² | MAE (bikes) |
|---|---|---|
| Hour as a number | 0.557 | 321 |
| Hour as dummies | 0.665 | 274 |
| Hour dummies + temperature² | 0.665 | 276 |
| **Hour dummies + temperature³** | **0.677** | **273** |

- **Turning the hour into dummies gave the biggest improvement.** Demand has two daily peaks, which a single number can't capture. RMSE fell from 417 to 363.
- **Curved temperature terms helped only a little** (R² 0.665 → 0.677).
- **Train and test scores are almost the same**, so the models are not overfitting.
- The best model is off by about 273 bikes per hour on average, about 37% of the average demand.

**What drives demand** (hour-dummy model):
- **Time of day matters most:** compared with midnight, about +508 bikes at 8am and +783 at 6pm, which points to commuting.
- **Temperature:** about +24 bikes per hour for each extra degree.
- **Rain:** about −65 bikes per hour for each mm.
- **Holidays:** about −131 bikes per hour.
- **Winter** has the lowest demand of all seasons.

**Temperature alone:** a straight line explains about 33% of demand (R² 0.325). Curves add very little (0.344 with temperature⁴ or temperature⁵).

## Known limitations

- A linear model applies the same hour effect to every day, but commuting peaks only happen on weekdays.
- The split is random. Neighbouring hours are very similar, so rows in the test set have near-copies in the training set, which may make the scores look better than they would on new days.
- Linear regression can predict fewer than 0 bikes, which is impossible.
- `Month` is used as a number, so December (12) and January (1) look far apart.
- The data covers one city and one year.

## Suggested next steps

**Fixes first**
1. Restart and run all cells, so the Seasons fix (`drop_first=True`) reaches every model.
2. Fix the "RMSE test" lines in the temperature² and temperature³ cells (they currently print the train RMSE).
3. Test temperature² and temperature³ together in the full model.
4. Update the notes so they match the new outputs, and remove leftover cells.

**Improve the model**

5. Add a weekend flag combined with the hour, so weekdays and weekends can have different daily patterns.
6. Try tree models (Random Forest, XGBoost), which find curves and interactions on their own.
7. Check for negative predictions, and try predicting the log of the target.
8. Test on whole days the model hasn't seen, instead of random hours.
9. Try the model with and without `Month`, since Seasons already covers the time of year.

## How to run

```
pip install pandas numpy matplotlib scikit-learn kagglehub
jupyter notebook BikeSharingDemand.ipynb
```

The notebook downloads the data from Kaggle with `kagglehub`. It was built in Google Colab.
