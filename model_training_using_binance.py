import ccxt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class data_download :
    def ccxt_download(self):
        symbol = 'BTC/USDT'
        timeframe = '1h'
        limit = 1000  
        binance = ccxt.binance()
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df

class create_feature:
    def signal(self,df):
        df['price_change'] = df['close'].diff().shift(-1)
        df.dropna(inplace=True)
        df['signal'] = (df['price_change'] > 0).astype(int)
        return df
    
class create_model:
    def split_data(self):
        features = df[['open', 'high', 'low', 'close', 'volume']]
        labels = df['signal']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_model(self,X_train,y_train):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        return rf_model
    
    def accuracy(self,model,X_test,y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy}')

if __name__ == '__main__':
    
    df = data_download().ccxt_download()
    df = create_feature().signal(df)
    X_train, X_test, y_train, y_test = create_model().split_data()
    model = create_model().train_model(X_train,y_train)
    report = create_model().accuracy(model,X_test,y_test)