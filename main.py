import pandas as pd

df = pd.read_csv("data/student-mat.csv", sep=';')

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

#KPI Columns
df['attendance'] = 100 - df['absences']
df['study_hours'] = df['studytime'] * 2
df['marks'] = df['G3']

df = df[['attendance', 'study_hours', 'marks']]

print("\nTransformed Data:")
print(df.head())

#Data Cleaning
df = df.dropna()
df = df[df['attendance'] >= 0]

print("\nAfter cleaning:", df.shape)

#Performance score
df['performance_score'] = (
    0.4 * df['attendance'] +
    0.3 * df['study_hours'] * 10 +
    0.3 * df['marks'] * 5
)


print("\nWith Performance Score:")
print(df.head())

print(df['performance_score'].hist())

#At-risk students
threshold = df['performance_score'].mean() - df['performance_score'].std()

df['at_risk'] = df['performance_score'] < threshold

#Percentage of at-risk students
risk_percent = df['at_risk'].mean() * 100

print("\nAt-Risk Students %:", risk_percent)

print("\nCorrelation Matrix:")
print(df.corr())

import matplotlib.pyplot as plt

#Marks Distribution
plt.hist(df['marks'], bins=20)
plt.title("Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

#Attendance vs Marks
plt.scatter(df['attendance'], df['marks'])
plt.title("Attendance vs Marks")
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.show()

#At-Risk Students
df['at_risk'].value_counts().plot(kind='bar')
plt.title("At-Risk Students")
plt.show()

print("\n--- FINAL INSIGHTS ---")

print(f"Average Marks: {df['marks'].mean():.2f}")
print(f"Average Attendance: {df['attendance'].mean():.2f}")
print(f"At-Risk Students: {df['at_risk'].mean()*100:.2f}%")

print("\nKey Insight:")
print("Students with higher study hours and attendance tend to perform better academically.")

df.to_csv("processed_data.csv", index=False)
