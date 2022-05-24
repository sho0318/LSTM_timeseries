import pandas as pd
import datetime
import numpy as np

class DataHundle():
    def __init__(self,DataDirectory,DataNum):
        self.DataDirectory = DataDirectory
        self.DataNum = DataNum
    
    def data_normalize(self,df):
        mean = df.mean()
        std = df.std()
        df = (df - mean) / std

        df = df.reset_index(drop=True)
        return df

    def day_devied(self,data):
        cols = ['Date Time', 'njobs']
        df = pd.DataFrame(index=[], columns=cols)
        
        n = 0
        start = datetime.datetime(2021,4,6)
        for d in range(0,42):
            start = start + datetime.timedelta(days=1)
            for num in range(0,24):
                dt1 = start + datetime.timedelta(seconds=3600*num)
                dt2 = dt1 + datetime.timedelta(seconds=3599)
                ddd = data[(data['que_time'] >= dt1) & (data['que_time'] <= dt2)]
                if len(ddd) > 0:
                    k = len(ddd)
                else:
                    k = 0

                record = pd.Series([dt1, k], index=cols)
                df = df.append(record, ignore_index=True)

        date_time = pd.to_datetime(df.pop('Date Time'), format='%Y-%m-%d %H:%M:%S')
        timestamp = date_time.map(pd.Timestamp.timestamp)

        day = 24*60*60
        year = (365.2425)*day

        df['Day sin'] = np.sin(timestamp * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

        df = self.data_normalize(df)
        return df

    def read_data(self,use_data):
        data_box = []
        start_day = (use_data-1)*7
        for i in range(self.DataNum):
            data_name = "data" + str(i)
            name = '../data/'+ self.DataDirectory + "/no_" + str(use_data) + '/' + data_name + ".csv"
            data = pd.read_csv(name, parse_dates=['que_time'])
            data = self.day_devied(data)
            data = data.drop(range(start_day,start_day+7))
            data = data.reset_index(drop=True)
            data_box.append(data)
        return data_box


class RealDataHundle():
    def __init__(self,RealBox):
        self.real_box = RealBox
    
    def data_normalize(self,df):
        mean = df.mean()
        std = df.std()
        df = (df - mean) / std

        df = df.reset_index(drop=True)
        return df
    
    def day_devied(self,data,start_num):
        cols = ['Date Time', 'njobs']
        df = pd.DataFrame(index=[], columns=cols)
        
        n = 0
        tmp = start_num * 7 + 6
        month = tmp // 30 + 4
        day = tmp % 30
        start = datetime.datetime(2021,month,day)
        for d in range(0,7):
            start = start + datetime.timedelta(days=1)
            for num in range(0,24):
                dt1 = start + datetime.timedelta(seconds=3600*num)
                dt2 = dt1 + datetime.timedelta(seconds=3599)
                ddd = data[(data['que_time'] >= dt1) & (data['que_time'] <= dt2)]
                if len(ddd) > 0:
                    k = len(ddd)
                else:
                    k = 0

                record = pd.Series([dt1, k], index=cols)
                df = df.append(record, ignore_index=True)

        date_time = pd.to_datetime(df.pop('Date Time'), format='%Y-%m-%d %H:%M:%S')
        timestamp = date_time.map(pd.Timestamp.timestamp)

        day = 24*60*60
        year = (365.2425)*day

        df['Day sin'] = np.sin(timestamp * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp * (2 * np.pi / day))
        df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

        df = self.data_normalize(df)
        return df

    def change_realdata(self):
        return_box = []
        for i in range(len(self.real_box)):
            box = self.real_box[i]
            return_box.append(self.day_devied(box,i))
        return return_box



if __name__ == "__main__":
    DataDirectory = "0.7"
    DataNum = 2
    fake_data = []

    datahundle = DataHundle(DataDirectory,DataNum)
    for use_data in range(3):
        fake_data.append(datahundle.read_data(int(use_data+1)))
    
    print(fake_data)

