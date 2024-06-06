import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def plot_active_member_distribution(data):
    active_counts = data['active_member'].value_counts()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    axs[0].pie(active_counts, labels=['Inactive', 'Active'], autopct='%1.1f%%', colors=['lightcoral', 'skyblue'], startangle=140)
    axs[0].set_title('Distribution of Active Members')
    axs[0].axis('equal')

    # Bar chart
    sns.countplot(x='active_member', data=data, palette=['lightcoral', 'skyblue'], ax=axs[1])
    for p in axs[1].patches:
        height = p.get_height()
        axs[1].annotate(f'{height}', xy=(p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axs[1].set_title('Count of Active vs Inactive Members')
    axs[1].set_xlabel('Active Member')
    axs[1].set_ylabel('Count')
    axs[1].set_xticks([0, 1])
    axs[1].set_xticklabels(['Inactive', 'Active'])

    plt.tight_layout()
    plt.show()



def plot_country_distribution(data):
    country_counts = data['country'].value_counts()
    unique_countries = country_counts.index

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Pie chart
    axs[0].pie(country_counts, labels=unique_countries, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel')[:len(unique_countries)])
    axs[0].set_title('Distribution of Countries')
    axs[0].axis('equal')

    # Bar chart
    sns.countplot(x='country', data=data, palette=sns.color_palette('pastel')[:len(unique_countries)], ax=axs[1])
    for p in axs[1].patches:
        height = p.get_height()
        axs[1].annotate(f'{height}', xy=(p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    axs[1].set_title('Count of Countries')
    axs[1].set_xlabel('Country')
    axs[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()
    

def plot_age_gender_distribution(data):
    age_gender_counts = data.groupby(['age', 'gender']).size().unstack().fillna(0)
    plt.figure(figsize=(14, 6))

    # Plotting with stacked bars to represent counts of each gender
    plt.bar(age_gender_counts.index, age_gender_counts['Male'], color='blue', label='Male', width=0.6)
    plt.bar(age_gender_counts.index, age_gender_counts['Female'], bottom=age_gender_counts['Male'], color='pink', label='Female', width=0.6)

    plt.title('Count of Each Age by Gender', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Gender', fontsize=12, title_fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def balance_age(data):
    plt.figure(figsize=(10, 6))
    # scatter plot
    sns.scatterplot(x='age', y='balance', data=data, hue='gender', palette='coolwarm')
    plt.title('Balance by Age')
    plt.xlabel('Age')
    plt.ylabel('Balance')
    plt.show()


def plot_boxplots(data):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(y=data['credit_score'], ax=axs[0], color='skyblue')
    axs[0].set_title('Boxplot of Credit Score')
    axs[0].set_yticks(axs[0].get_yticks())
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=90)

    sns.boxplot(y=data['balance'], ax=axs[1], color='salmon')
    axs[1].set_title('Boxplot of Balance')
    axs[1].set_yticks(axs[1].get_yticks())
    axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=90)

    sns.boxplot(y=data['estimated_salary'], ax=axs[2], color='green')
    axs[2].set_title('Boxplot of Estimated Salary')
    axs[2].set_yticks(axs[2].get_yticks())
    axs[2].set_yticklabels(axs[2].get_yticklabels(), rotation=90)

    plt.tight_layout()
    plt.show()