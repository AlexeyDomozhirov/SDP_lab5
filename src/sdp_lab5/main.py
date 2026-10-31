import pandas as pd
import numpy as np

def swap_rows(df, pos_a, pos_b):
    """Меняет местами строки по их позициям (0-based) в датафрейме."""
    label_a = df.index[pos_a]   # метка строки на позиции pos_a
    label_b = df.index[pos_b]   # метка строки на позиции pos_b
    tmp = df.loc[label_a].copy()
    df.loc[label_a] = df.loc[label_b]
    df.loc[label_b] = tmp
    return df

def show(num, obj):
    print(f"\nЗадание {num}")
    print(obj)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Задание 1
df1 = pd.DataFrame({
    'Name': [
        'Strom, Mrs. Wilhelm (Elna Matilda Persson)',
        'Navratil, Mr. Michel ("Louis M Hoffman")',
        'Minahan, Miss. Daisy E'
    ],
    'Age': [29, 36.5, 33],
    'Sex': ['female', 'male', 'female']
})
show(1, df1)

# Задание 2
url2 = 'https://drive.google.com/uc?export=download&id=16gikf9vvMLyldkdHtoDYrPDbqTHP1BxG'
df2 = pd.read_csv(url2, sep=';', skipinitialspace=True)
show(2, df2.head())

# Задание 3
url3 = 'https://gist.githubusercontent.com/zaryanezrya/8b4ef51c707cb16d5e88a44dc00a1bb2/raw/41230f49c6268e072dbf102672f670be256922ab/gistfile1.txt'
df3 = pd.read_csv(url3, sep=',', skipinitialspace=True)
show(3, df3.head())

# Задание 4
df = pd.concat([df2, df3], ignore_index=True).drop_duplicates()
show(4, df.shape)

# Задание 5
df = df.set_index('PassengerId').sort_values(by='PassengerId')
show(5, df.head())

# Задание 6
print('Задание 6')
df.info()
print(df.describe())

# Задание 7
df = swap_rows(df, 0, 2)
show(7, df.head())

# Задание 8
df['Sex'] = df['Sex'].map({'female': 'f', 'male': 'm'})
show(8, df['Sex'].value_counts())

# Задание 9
ticket_counts = df.groupby('Ticket')['Name'].count()
big_tickets = ticket_counts[ticket_counts >= 6].index
show(9, df[df['Ticket'].isin(big_tickets)][['Name', 'Ticket']])

# Задание 10
names1 = df1['Name'].tolist()
if 'Cabin' in df.columns:
    cabins1 = df.loc[df['Name'].isin(names1), 'Cabin'].dropna().unique()
else:
    cabins1 = []

if len(cabins1) > 0:
    same_cabin = df[df['Cabin'].isin(cabins1) & ~df['Name'].isin(names1)]
    show(10, same_cabin[['Name', 'Cabin', 'Ticket']])
else:
    tickets1 = df.loc[df['Name'].isin(names1), 'Ticket'].dropna().unique()
    same_cabin = df[df['Ticket'].isin(tickets1) & ~df['Name'].isin(names1)]
    show(10, same_cabin[['Name', 'Ticket']])

# Задание 11
df['BirthYear'] = 1912 - df['Age']
show(11, df[['Name', 'Age', 'BirthYear']].head())

# Задание 12
if 'Cabin' in df.columns:
    df['Companion'] = (
        df.groupby('Cabin')['Cabin'].transform('count') - 1
    ).fillna(0).astype(int)
else:
    df['Companion'] = (
        df.groupby('Ticket')['Ticket'].transform('count') - 1
    ).fillna(0).astype(int)
show(12, df[['Name', 'Companion']].head())

# Задание 13
df = swap_rows(df, 0, 1)
show(13, df.head())

# Задание 14
df.to_csv('titanic_result.csv')
show(14, "Датафрейм сохранён в файл titanic_result.csv")

# Задание 15
show(15, df.nlargest(10, 'Fare')[['Name', 'Fare']])

# Задание 16
show(16, df.groupby(['Sex', 'Survived']).size().unstack())

# Задание 17
show(17, df.groupby(['Pclass', 'Survived']).size().unstack())
