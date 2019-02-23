from jira.client import JIRA
import pandas as pd
import re
import numpy as np
from datetime import datetime

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt
plt.style.use('ggplot')


import jira_creds

def fetch_comment_data_from_jira(jira_instance,project_key,start,stop):
    print('Begin comment fetch from JIRA')
    all_comments = range(start,stop+1)
    comment_output = []
    bad_comment = []
    for key_number in all_comments:
        issue_str = project_key+'-'+str(key_number)
        try:
            issue = jira_instance.issue(issue_str)
            comments = [comment for comment in issue.fields.comment.comments]
            for comment in comments:
                name, domain = comment.author.emailAddress.split('@')
                if (domain != 'companyname.com'):
                    comment_output.append([issue.key, comment.author.displayName, comment.author.emailAddress, comment.created, comment.body])
            print(issue_str, ' comment_output_count', len(comment_output))
        except:
            print('no bueno ' + issue_str)
            bad_comment.append(key_number)
        start += 1

    #put comment data in dataframe
    col_list = ['key', 'author', 'email', 'created_date', 'comment']
    comment_data = pd.DataFrame(comment_output, columns=col_list)

    return comment_data, bad_comment


def save_comment_data_to_csv(df, path, filename):
    complete_path = path + '\%s' % filename
    try:
        df.to_csv(complete_path, header=True, index=False, encoding='utf-8')
        print 'saved data to %s' % complete_path
    except:
        print 'something went wrong saving the csv to %s' % complete_path

def get_existing_comment_data(path,filename):
    complete_path = path + '\%s' % filename
    try:
        df = pd.read_csv(complete_path, index_col=False, encoding='utf-8')
        return df
    except:
        print('No Such File')


def comment_scrub(df, column, export=False, path=None, filename=None):
    # !2018-09-14_9-34-15.png|thumbnail!
    df["scrubbed"] = [re.sub("!.*!", "", text) for text in df[column]]

    # [^some_attachemnt.docx] _(1.25 MB)_
    df["scrubbed"] = [re.sub(r"\[\^.*\)_", "", text) for text in df["scrubbed"]]

    # DOBs (or any date)
    df["scrubbed"] = [re.sub("\d{0,4}[/-]\d{0,2}[/-]\d{0,4}", "", text) for text in df["scrubbed"]]

    # four to twelve digits in a row but not jira ticket numbers (ZZZZ-000)
    df["scrubbed"] = [re.sub("(?<![A-Z]{4}-)\d\d{3,10}", "", text) for text in df["scrubbed"]]

    # carriage returns
    df["scrubbed"] = [re.sub(r"\r", " ", text) for text in df["scrubbed"]]

    # line returns
    df["scrubbed"] = [re.sub(r"\n", " ", text) for text in df["scrubbed"]]

    # replace any white space with a single space and remove trailing whitespace
    df["scrubbed"] = [re.sub('\s+', ' ', text).strip() for text in df["scrubbed"]]

    # replace SQL statements
    df["scrubbed"] = [re.sub('(SELECT|select).*,.*(FROM|from).*', '', text, re.DOTALL) for text in df["scrubbed"]]

    # non alphanumeric and non punctuation characters
    df["scrubbed"] = [re.sub(r"""[^a-zA-Z0-9\.,\!\?\"\'\:\-/ ]""", "", text) for text in df["scrubbed"]]


    if export:
        try:
            complete_path = path + '\\' + filename + '.csv'
            df.to_csv(complete_path, header=True, index=False, encoding='utf-8')
            print('exported scrubbed data to ' + complete_path)

        except:
            print('something went wrong exporting scrubbed data')

    return df


def plot_sentiment_scatter(x_ax,y_ax,x_label,y_label,filename):
    fig, ax = plt.subplots(figsize=(11, 7))
    comment_data.plot(x=x_ax, y=y_ax, kind='scatter', ax=ax)
    ax.set_xticklabels([datetime.fromtimestamp(ts / 1e9).strftime("%Y-%m-%d") for ts in ax.get_xticks()])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.ylim(-1, 1)
    plt.xticks(rotation=90)
    plt.savefig(path + '\\' + filename + '.png')
    plt.show()


def group_by_week(df, datecolumn, week_column_name='week_starting'):
    df[datecolumn] = pd.to_datetime(df[datecolumn]) # ensure date column is a date
    df['day_of_week'] = df[datecolumn].dt.dayofweek #create new column with days to stubtract to get to monday
    df[week_column_name] = df[datecolumn] - pd.to_timedelta(df['day_of_week'], unit='d')
    df[week_column_name] = df[week_column_name].dt.normalize()
    df = df.drop('day_of_week', axis=1)
    return df

def plot_summed_components(df, x, y_columns_list, grouping_column, format_string):
    print('x shape: ', df[x].shape)
    print('y shape: ', df[y_columns_list[1]].shape)
    plt.figure(1)
    no_of_figs = str(len(y_columns_list))
    no_of_columns = '1'

    for column_name in y_columns_list:
        current_fig = str(y_columns_list.index(column_name) + 1)
        print (no_of_figs, no_of_columns, current_fig)
        subplotnumber = no_of_figs + no_of_columns + current_fig
        print (subplotnumber)
        plt.subplot(int(subplotnumber))
        print(x, grouping_column, column_name)
        plt.plot(df[x], df.groupby(grouping_column)[column_name].sum(), format_string)

    plt.show()

#ACTUAL SCRIPT START

path = '..\Data\\'
filename = 'comments.csv'
comment_data = get_existing_comment_data(path,filename)

#CHECK DATA LOOKS GOOD
print 'Collected {} row of comment data'.format(len('comment_data'))
print(comment_data.head())

comment_data = comment_scrub(comment_data,'comment', export=True, path=path, filename=filename)

# do sentiments analysis
sid = SentimentIntensityAnalyzer()
sentiment_output = pd.DataFrame([sid.polarity_scores(i) for i in comment_data.comment])
scrubbed_sentiment_output = pd.DataFrame([sid.polarity_scores(i) for i in comment_data.scrubbed])

# join sentiments to comments
comment_data = comment_data.join(sentiment_output)
comment_data = comment_data.join(scrubbed_sentiment_output, rsuffix='_scrubbed')

# see high and low sentiment values immediately
print (comment_data[comment_data['compound'] > 0.9 ].sort_values('compound', ascending=False).to_string())
print (comment_data[comment_data['compound'] < -0.5 ].sort_values('compound', ascending=True).to_string())

# plot compound sentiment score vs time
comment_data['created_timestamp'] = pd.to_datetime(comment_data.created_date)
comment_data['created_date_unix'] = comment_data.created_timestamp.astype(np.int64)

comment_data = group_by_week(comment_data, 'created_date')

plt.figure(1)
no_of_figs = '3'
no_of_columns = '1'

current_fig = '1'
subplotnumber = no_of_figs + no_of_columns + current_fig
print (subplotnumber)
plt.subplot(int(subplotnumber))
print('week_starting', 'week_starting', 'neu')
plt.plot(comment_data['week_starting'], comment_data.groupby('week_starting')['neu'].sum())

current_fig = '2'
subplotnumber = no_of_figs + no_of_columns + current_fig
print (subplotnumber)
plt.subplot(int(subplotnumber))
print('week_starting', 'week_starting', 'neg')
plt.plot(comment_data['week_starting'], comment_data.groupby('week_starting')['neg'].sum())

current_fig = '3'
subplotnumber = no_of_figs + no_of_columns + current_fig
print (subplotnumber)
plt.subplot(int(subplotnumber))
print('week_starting', 'week_starting', 'pos')
plt.plot(comment_data['week_starting'], comment_data.groupby('week_starting')['pos'].sum())

plt.show()