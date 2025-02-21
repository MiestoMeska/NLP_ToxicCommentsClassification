import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer

def plot_class_distribution(x):
    """
    Plots the distribution of class occurrences.

    Parameters:
     (pandas.Series): A Series containing the count of occurrences for each class.
    """
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=x.index, y=x.values, alpha=0.8)
    plt.title("Count per Class")
    plt.ylabel('Count of Occurrences', fontsize=12)
    plt.xlabel('Type', fontsize=12)

    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label, ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.show()

def plot_class_distribution_per_comment(data, labels):
    """
    Plots the distribution of the number of classes each comment has.

    Parameters:
     data (DataFrame): The dataset containing the comments and their labels.
     labels (list): List of column names corresponding to the labels.

    Returns:
    - None
    """

    num_classes_per_comment = data[labels].sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    
    counts, bins, patches = plt.hist(num_classes_per_comment, bins=range(len(labels) + 2), align='left', rwidth=0.8)
    

    plt.xlabel('Number of Classes')
    plt.ylabel('Number of Comments')
    plt.title('Distribution of the Number of Classes per Comment')
    plt.xticks(range(len(labels) + 1))


    for count, bin_position in zip(counts, bins[:-1]): 
        plt.text(bin_position, count + 0.5, str(int(count)), ha='center', fontsize=12)

    plt.show()

def plot_comment_lengths_with_stats(dataframe, title):
    """
    Plots a histogram of comment token lengths along with descriptive statistics and visual markers.
    
    Token length regions are highlighted with different background colors:
    - 0-128 tokens: No background
    - 128-256 tokens: Light yellow
    - 256-512 tokens: Light orange
    - >512 tokens: Light red (truncated region)
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the 'cleaned_text' column.
    - title (str): The title for the plot.
    
    Returns:
    - None: Displays the plot and prints the descriptive statistics for comment lengths.
    """

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    
    df_copy = dataframe.copy()
    df_copy['comment_length'] = df_copy['cleaned_text'].apply(lambda x: len(tokenizer.encode(x)))

    describe_table = df_copy['comment_length'].describe()
    print("Describe Table of Tokenized Comment Lengths:")
    print(describe_table)
    
    plt.figure(figsize=(10, 6))

    counts, bins, patches = plt.hist(df_copy['comment_length'], bins=100, color='blue', edgecolor='black', label='Comment Lengths')

    plt.axvspan(128, 256, color='yellow', alpha=0.2, label='128-256 Tokens (Light Yellow)')
    plt.axvspan(256, 512, color='orange', alpha=0.2, label='256-512 Tokens (Light Orange)')
    plt.axvspan(512, max(bins), color='red', alpha=0.3, label='Exceeds 512 Tokens (Light Red)')

    mean_length = df_copy['comment_length'].mean()
    median_length = df_copy['comment_length'].median()
    mode_length = df_copy['comment_length'].mode().iloc[0]

    plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_length:.0f}')
    plt.axvline(median_length, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_length:.0f}')
    plt.axvline(mode_length, color='orange', linestyle='dashed', linewidth=1, label=f'Mode: {mode_length:.0f}')

    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.show()
