import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class League:
    def __init__(self, folder_path):
        self.name, self.year = folder_path.split("_")[0], folder_path.split("_")[1]
        self.model = None
        self.path = folder_path
        self.look_back = None
        self.output_type =None
        self.initial_data = self.__create_init_data()
        self.X = None
        self.Y = None
        self.bet_odds = None


    @staticmethod
    def __str2match_week(df):
        df['Date'] = pd.to_datetime(df['Date']).dt.week
        match_week_dictionary = dict([(y, x) for x,y in enumerate(sorted(list(set(df['Date']))))])
        df['Date'] = df['Date'].apply(lambda x: match_week_dictionary[x])

        return df

    @staticmethod
    def __onehot_for_teams(df):
        pd_home = pd.get_dummies(df['HomeTeam'], prefix='home')
        df = pd.concat([df, pd_home], axis=1)
        pd_away = pd.get_dummies(df['AwayTeam'], prefix='away')
        df = pd.concat([df, pd_away], axis=1)
        return df

    @staticmethod
    def __what_to_keep(df):
        list_of_columns = list(df.columns.values)
        over = 'B365>2.5' if 'B365>2.5' in list_of_columns else None
        under = 'B365<2.5' if 'B365<2.5' in list_of_columns else None
        list_of_what_to_keep_main = ['FTR', 'avrg_goals_Home', 'avrg_goals_Away', 'avrg_goals_opp_Home', 'avrg_goals_opp_Away',
                                     'avrg_shoots_Home', 'avrg_shoots_Away', 'avrg_shoots_opp_Home', 'avrg_shoots_opp_Away',
                                     'points_Home', 'points_Away', 'points_opp_Home', 'points_opp_Away', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
        list_of_what_to_keep_bet = ['B365H', 'B365D', 'B365A', over, under]
        all_columns = list_of_what_to_keep_main + list_of_what_to_keep_bet
        all_columns_filtered_none = [col for col in all_columns if col is not None]
        return df[all_columns_filtered_none]

    @staticmethod
    def __create_output_parameter(df, output_parameter):
        if output_parameter == 'over/under':
            pass
        elif output_parameter == 'result classification':
            pass
        elif output_parameter == 'result regression':
            df['y'] = df['FTHG'] - df['FTAG']
            df = df.drop(['FTHG', 'FTAG', 'FTR', 'HomeTeam', 'AwayTeam'], axis=1)
        elif output_parameter == 'corners':
            pass
        else:
            raise ValueError('illegal argument for output_parameter')
        return df

    @staticmethod
    def __sum_points(list_of_result, home_or_away):
        if home_or_away == 'Home':
            result_dic = {'H': 3, 'D': 1, 'A': 0}
        elif home_or_away == 'Away':
            result_dic = {'H': 0, 'D': 1, 'A': 3}
        points = [result_dic[x] for x in list_of_result]
        return sum(points)

    def __feature_creator(self, team_name, week_day, home_or_away, n_last_games):
        data = self.initial_data.loc[(self.initial_data['HomeTeam'] == team_name) | (self.initial_data['AwayTeam'] == team_name)]
        data = data.loc[(data['Date'] < week_day) & (data['Date'] >= week_day - n_last_games)]
        data_for_home = data.loc[data['HomeTeam'] == team_name]
        data_for_away = data.loc[data['AwayTeam'] == team_name]
        new_data = {}
        new_data['avrg_goals_' + home_or_away] = np.mean(list(data_for_home['FTHG'].values) + list(data_for_away['FTAG'].values))
        new_data['avrg_goals_opp_' + home_or_away] = np.mean(list(data_for_home['FTAG'].values) + list(data_for_away['FTHG'].values))
        new_data['avrg_shoots_' + home_or_away] = np.mean(list(data_for_home['HS'].values) + list(data_for_away['AS'].values))
        new_data['avrg_shoots_opp_' + home_or_away] = np.mean(list(data_for_home['AS'].values) + list(data_for_away['HS'].values))
        new_data['points_' + home_or_away] = self.__sum_points(data_for_home['FTR'], 'Home') + self.__sum_points(data_for_away['FTR'], 'Away')
        new_data['points_opp_' + home_or_away] = self.__sum_points(data_for_home['FTR'], 'Away') + self.__sum_points(data_for_away['FTR'], 'Home')
        return new_data

    def __create_features_per_row(self, row, n_last_games):
        home_feat = self.__feature_creator(row['HomeTeam'], row['Date'], 'Home', n_last_games)
        away_feat = self.__feature_creator(row['AwayTeam'], row['Date'], 'Away', n_last_games)
        row['avrg_goals_Home'] = home_feat['avrg_goals_Home']
        row['avrg_goals_Away'] = away_feat['avrg_goals_Away']
        row['avrg_goals_opp_Home'] = home_feat['avrg_goals_opp_Home']
        row['avrg_goals_opp_Away'] = away_feat['avrg_goals_opp_Away']
        row['avrg_shoots_Home'] = home_feat['avrg_shoots_Home']
        row['avrg_shoots_Away'] = away_feat['avrg_shoots_Away']
        row['avrg_shoots_opp_Home'] = home_feat['avrg_shoots_opp_Home']
        row['avrg_shoots_opp_Away'] = away_feat['avrg_shoots_opp_Away']
        row['points_Home'] = home_feat['points_Home']
        row['points_Away'] = away_feat['points_Away']
        row['points_opp_Home'] = home_feat['points_opp_Home']
        row['points_opp_Away'] = away_feat['points_opp_Away']
        return row

    @staticmethod
    def __some_renaming(df):
        column_names = list(df.columns.values)
        if 'HG' in column_names:
            df = df.rename(index=str, columns={"HG": "FTHG"})
        if 'AG' in column_names:
            df = df.rename(index=str, columns={"AG": "ATHG"})
        if 'Res' in column_names:
            df = df.rename(index=str, columns={"Res": "FTR"})
        return df

    def __create_init_data(self):
        df = pd.read_csv(self.path)
        df = self.__str2match_week(df)
        df = self.__some_renaming(df)
        return df

    def create_dataset(self, look_back, output_parameter):
        df = self.initial_data.loc[self.initial_data['Date'] > look_back]
        df = df.apply(lambda row: self.__create_features_per_row(row, look_back), axis=1)
        df = self.__what_to_keep(df)
        df = self.__onehot_for_teams(df)
        data = self.__create_output_parameter(df, output_parameter)
        self.Y = data['y'].values
        self.bet_odds = data[['B365H', 'B365D', 'B365A']].values
        self.X = data.drop(['y', 'B365H', 'B365D', 'B365A'], axis=1).values
        self.look_back = look_back
        self.output_type = output_parameter
        pass

    @staticmethod
    def __prediction_metric(preds, dtrain):
        labels = [1.0 if x > 0.5 else 2.0 if x < -0.5 else 0.0 for x in dtrain.get_label()]
        predictions = [1.0 if x > 0.5 else 2.0 if x < -0.5 else 0.0 for x in  preds]
        score = accuracy_score(labels, predictions)
        return "match_score", score

    def train_model(self, test_size=0.2, max_depth=3, eta=1.0, n_rounds=100):
        data_train, data_test, labels_train, labels_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)
        dtrain = xgb.DMatrix(data_train, label=labels_train)
        dtest = xgb.DMatrix(data_test, label=labels_test)
        param = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'max_depth': max_depth, 'eta': eta, 'seed':0, 'silent': True}
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        #xgb_model = xgb.train(param, dtrain, n_rounds, evallist, feval=self.__prediction_metric, maximize=True, early_stopping_rounds=10)
        clf = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_rounds, learning_rate=eta, objective="reg:linear", seed=0)
        xgb_model = xgb.train(param, dtrain, n_rounds, evallist)
        cvresult = xgb.cv(clf.get_xgb_params(), dtrain, num_boost_round=100, nfold=15, metrics=['rmse'], stratified=True)

        return 1


    def save_moddel(self):
        pass

if __name__ == '__main__':
    testObj = League('data/england_17.csv')
    testObj.create_dataset(5, 'result regression')
    testObj.train_model()

#TODO: check what to do with first rounds.