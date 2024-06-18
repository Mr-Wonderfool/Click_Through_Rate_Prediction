import warnings as w
w.filterwarnings('ignore')
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from utils.utils import F1_accuracy, thresholding
from sklearn.preprocessing import LabelEncoder
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, get_feature_names
from tensorflow.python.keras.optimizers import Adam

class DeepCTR:
    def __init__(self):
        self.sparse_features = None
        self.dense_features = None
        self.train_, self.test_ = None, None
        self.dnn_feature_columns, self.linear_feature_columns = None, None
        self.feature_names = None
        self.model = None
        self.target = None
        # hyperparameter tuning
        self.batch_size = 1024
        self.embedding_dims = 15
        self.hidden_size = [512, 256, 256, 128]
        self.l2_reg_linear = 0.000001
        self.l2_reg_embedding = 0.000001
        self.l2_reg_dnn = 0.001

    def _preprocess(self, data_path, is_test):
        data = pd.read_csv(data_path).drop(columns=['id', 'user_id'])
        data.replace('Male', 1., inplace=True)
        data.replace('Female', 0., inplace=True)
        if not is_test:
            data.dropna(inplace=True)
            data['date'] = pd.to_datetime(data['date'], format="%Y/%m/%d %H:%M")
        else:
            data.fillna(method='ffill', inplace=True)
            data['date'] = pd.to_datetime(data['date'], format="%m-%d %H:%M")
        data['date'] = data['date'].dt.hour
        return data
    def _balance_data(self, data: pd.DataFrame):
        """
        Perform Synthetic Minority Over-Sampling (SMOTE) Technique 
        to balance positive and negative data

        Returns
        -------
        balanced_train: pd.DataFrame
            hstack the return value from SMOTE.fit_transform (after shuffling)
            and form dataframe with the original columns
        """
        data_all = data.values
        train, labels = data_all[..., :-1], data_all[..., -1]
        sm = SMOTE(sampling_strategy=.7, random_state=42)
        X_res, y_res = sm.fit_resample(train, labels)
        combined = np.array(np.hstack((X_res, y_res[:, np.newaxis])), dtype=np.int32)
        np.random.shuffle(combined)
        balanced_train = pd.DataFrame(combined, columns=data.columns)
        return balanced_train
    def get_data(self, train_path='../data/train.csv', test_path='../data/test.csv'):
        """Read training data and adjust to DeepFM input style
        Assign self.train, self.test according to specified path
        """
        # change date value to hour
        train, test = self._preprocess(train_path, is_test=False), self._preprocess(test_path, is_test=True)
        train = self._balance_data(train)
        encoder = LabelEncoder()
        sparse_features = train.columns # assume no dense features
        self.target = ['isClick']
        self.sparse_features = sparse_features.delete(sparse_features.get_loc(self.target[0]))
        for feat in self.sparse_features:
            train[feat] = encoder.fit_transform(train[feat])
            test[feat] = encoder.fit_transform(test[feat])
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=train[feat].max()+1, embedding_dim=self.embedding_dims)
            for feat in self.sparse_features]
        self.dnn_feature_columns = fixlen_feature_columns
        self.linear_feature_columns = fixlen_feature_columns
        # TODO: add dense feature columns
        self.feature_names = get_feature_names(self.linear_feature_columns + self.dnn_feature_columns)
        self.train_, self.test_ = train, test

    def fit(self, pretrained_path=None):
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            from tensorflow.python.keras.models import load_model
            from deepctr.layers import Linear, DNN, FM, NoMask, _Add, Concat, PredictionLayer
            pretrained = load_model(pretrained_path, custom_objects=
                {'Concat': Concat, 'Linear': Linear, 'DNN': DNN, 'NoMask': NoMask, 'FM': FM, 
                '_Add': _Add, 'PredictionLayer': PredictionLayer})
            # pretrained.summary()
            self.model = pretrained
            return None
        from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
        train_model_input = {name: self.train_[name] for name in self.feature_names}
        checkpoints = ModelCheckpoint(filepath='../model_data/Deep_fm.h5', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # TODO: dropout necessary for sparse features? (maybe not when data size is sparse in the first place)
        model = DeepFM(self.linear_feature_columns, self.dnn_feature_columns, 
                dnn_hidden_units=self.hidden_size, 
                l2_reg_linear=self.l2_reg_linear, l2_reg_embedding=self.l2_reg_embedding, l2_reg_dnn=self.l2_reg_dnn)
        # model = DeepFM(self.linear_feature_columns, self.dnn_feature_columns)
        model.compile(loss="binary_crossentropy", metrics=['AUC', 'binary_crossentropy'], optimizer=Adam(lr=0.00001))
        model.fit(
            train_model_input, self.train_[self.target].values, 
            batch_size=self.batch_size,
            epochs=10,
            verbose=2, # change to 1 for bar information
            validation_split=.1,
            callbacks=[checkpoints, early_stopping]
        )
        self.model = model
    
    def predict(self):
        test_model_input = {name: self.test_[name] for name in self.feature_names}
        prediction = self.model.predict(test_model_input)
        true = self.test_[self.target].values
        return prediction, true
    def evaluate(self):
        prediction, true = self.predict()
        for i in range(int(len(prediction) / 1e3)):
            print(f"True label: {true[i]}, predicted as: {prediction[i]}")
        out = thresholding(prediction, .5)
        F1, acc = F1_accuracy(out, true)
        return F1, acc

if __name__ == '__main__':
    model = DeepCTR()
    train_path='../data/train.csv'
    test_path='../data/test.csv'
    # pretrained_weights = '../model_data/Deep_fm.h5'
    model.get_data(train_path=train_path, test_path=test_path)
    model.fit()
    F1, acc = model.evaluate()
    print(f"F1 score: {F1}, accuracy: {acc}")