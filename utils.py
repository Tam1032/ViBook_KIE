import pandas as pd

def transform_tf_points(df, tf, size):
    train_points = df[['points', 'image']].copy()
    train_points = pd.merge(left=train_points, right=size, left_on='image', right_on='image')
    train_points['x_1'] = train_points.apply(lambda x: x['points'][0][0]/x['width'], axis=1)
    train_points['x_2'] = train_points.apply(lambda x: x['points'][1][0]/x['width'], axis=1)
    train_points['x_3'] = train_points.apply(lambda x: x['points'][2][0]/x['width'], axis=1)
    train_points['x_4'] = train_points.apply(lambda x: x['points'][3][0]/x['width'], axis=1)
    train_points['y_1'] = train_points.apply(lambda x: x['points'][0][1]/x['height'], axis=1)
    train_points['y_2'] = train_points.apply(lambda x: x['points'][1][1]/x['height'], axis=1)
    train_points['y_3'] = train_points.apply(lambda x: x['points'][2][1]/x['height'], axis=1)
    train_points['y_4'] = train_points.apply(lambda x: x['points'][3][1]/x['height'], axis=1)
    train_points.drop(labels=['points', 'image', 'width', 'height'], axis=1, inplace=True)
    train_text = pd.DataFrame(
    data=tf.toarray(),
    index=train_points.index,
    columns=tfidf_tokens
    )
    X_train = pd.concat([train_points, train_text], axis=1)
    return X_train

def return_text(df, col, label):
    return ' '.join(list(df[df[col]==label].text.values))

def convert_df_to_text(df):
    temp = df.sort_values(by=['label', 'link'])
    title = return_text(temp, 'label', 'title')
    author = return_text(temp, 'label', 'author')
    publisher = return_text(temp, 'label', 'publisher')
    other = return_text(temp, 'label', 'other')
    text = title + '|||' + author +  '|||' + publisher + '|||' + other
    return text

def convert_result_to_text(df):
    temp = df.sort_values(by=['pre_label', 'Mean_y'])
    title = return_text(temp, 'pre_label', 'title')
    author = return_text(temp, 'pre_label', 'author')
    publisher = return_text(temp, 'pre_label', 'publisher')
    other = return_text(temp, 'pre_label', 'other')
    text = title + '|||' + author +  '|||' + publisher + '|||' + other
    return text

def failed_examples(data, true, predict, y_true, y_pred):
    failed = (y_pred != y_true)
    data_failed = data[failed].reset_index(drop=True)
    return data_failed[np.logical_and((y_true[failed] == true), (y_pred[failed] == predict))]