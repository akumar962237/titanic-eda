# Titanic Survival: Exploratory Data Analysis

Exploratory analysis of the Titanic passenger dataset (891 passengers) using Python, Pandas, Seaborn and Matplotlib. The project covers data inspection, cleaning, feature engineering, analysis and visualization.

## Tools
Python, Pandas, Seaborn, Matplotlib

## Data Cleaning
- **Missing values found:** `age` (177), `deck` (688), `embarked` / `embark_town` (2 each)
- **deck:** dropped (about 77% missing, not reliable)
- **embarked / embark_town:** filled with the most common value (mode)
- **age:** filled with the median age of the passenger's class and gender group, which is more accurate than one overall median
- **Identical-looking rows:** 107 rows look duplicated, but the dataset has no name/ID column, so they are different passengers. They were kept.
- Result: 0 missing values, all 891 rows retained

## Feature Engineering
- `family_size` = siblings/spouses + parents/children + 1
- `age_group` = Child, Teen, Young Adult, Adult, Senior

## Key Insights
1. **Overall survival rate was 38.4%.**
2. **Gender was the strongest factor:** 74.2% of women survived vs 18.9% of men.
3. **Class mattered a lot:** survival was 63.0% in 1st class, 47.3% in 2nd and 24.2% in 3rd.
4. **Children had the best chances (58.0%)**, while seniors had the lowest (22.7%).
5. **Medium-sized families did best:** people travelling with 1 to 3 family members (family size 2 to 4) had 55% to 72% survival, while those alone (30.4%) or in families of 5 or more did worse.
6. **Fares track class:** median fare was 60.29 (1st), 14.25 (2nd) and 8.05 (3rd). Passengers who boarded at Cherbourg survived more often (55.4%), because about 51% of Cherbourg passengers were 1st class (vs 20% at Southampton).

## Charts
All charts are in the `charts/` folder: survival by gender, class + gender, age group, family size, fare by class, and a correlation heatmap.

## How to Run
```
pip install pandas seaborn matplotlib
python eda_titanic.py
```

## Author
Ajay Kumar | github.com/akumar962237
