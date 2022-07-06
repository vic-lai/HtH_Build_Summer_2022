import pandas

pandas.options.display.max_columns = None
pandas.options.display.max_rows = None

ds_salary_data = pandas.read_csv("ds_salaries.csv")

print(ds_salary_data)